import re
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents._base_chat_agent import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.mas_graph.math_graph.math_env import MathEnv, MathEnvBatch


PLANNER_USER_INSTRUCTION = (
    "The full math problem is already present earlier in this conversation. "
    "Produce a plan only. Do not solve the problem and do not give the final answer. "
    "Your output must use exactly these section headers:\n"
    "Goal:\n"
    "Given:\n"
    "Unknowns:\n"
    "Method:\n"
    "Steps:\n"
    "Checks:\n"
    "Keep each section concise. "
    "In Method, name the main strategy. "
    "In Steps, give 3 to 6 numbered steps. "
    "In Checks, mention the main failure modes or edge cases to verify."
)

SOLVER_USER_INSTRUCTION = (
    "The full math problem and any plan are already present earlier in this conversation. "
    "Solve the problem now by following the available plan. "
    "Do not ask the user to restate the problem. "
    "Your output must use exactly these section headers:\n"
    "PLAN_USED:\n"
    "DERIVATION:\n"
    "CHECKS:\n"
    "FINAL_ANSWER:\n"
    "In FINAL_ANSWER, write only the final result. "
    "End with a final line exactly in the form: Final Answer: <your answer>."
)

CRITIQUE_USER_INSTRUCTION = (
    "The full math problem, candidate plan, and candidate solution are already present earlier in this conversation. "
    "Critique both the plan and the solution. "
    "Do not solve the problem yourself. "
    "Use exactly these section headers:\n"
    "PLAN_CRITIQUE:\n"
    "SOLUTION_CRITIQUE:\n"
    "FIRST_ERROR:\n"
    "REVISION_FOCUS:\n"
    "In PLAN_CRITIQUE, identify the main weakness in the plan, or say the plan is acceptable if it is sound. "
    "In SOLUTION_CRITIQUE, identify the main weakness in the solution, or say the solution is acceptable if it is sound. "
    "FIRST_ERROR must be a one-sentence natural-language description, not a label. "
    "In REVISION_FOCUS, state exactly what the reviser must fix first."
)

REVISE_USER_INSTRUCTION = (
    "The full math problem, prior plan, prior solution, and critique are already present earlier in this conversation. "
    "Write a revised version of the whole attempt: revise both the plan and the solution. "
    "Do not ask the user to restate the problem. "
    "Your output must use exactly these section headers:\n"
    "REVISED_PLAN:\n"
    "REVISED_SOLUTION:\n"
    "CHECKS:\n"
    "FINAL_ANSWER:\n"
    "In FINAL_ANSWER, write only the final result. "
    "Start by writing a brief corrected plan, then write the corrected full solution that follows that plan. "
    "If the critique points out plan weaknesses, fix the plan before repairing the derivation. "
    "Focus on repairing derivation, algebra, arithmetic, case handling, omitted justifications, and any local plan issues. "
    "Do not preserve prior equations or assumptions unless you independently verify they are valid. "
    "End with a final line exactly in the form: Final Answer: <your answer>."
)

VERIFIER_USER_INSTRUCTION = (
    "The full math problem, revised plan, and revised solution are already present earlier in this conversation. "
    "Judge only whether the revised solution should be accepted. "
    "Perform an independent check instead of trusting the prior reasoning. "
    "Do not approve a solution just because it looks plausible or polished. "
    "Recompute the key equations, substitutions, and final numeric result for yourself. "
    "If you cannot independently justify the conclusion, do not approve. "
    "Your output must use exactly these section headers:\n"
    "DECISION:\n"
    "INDEPENDENT_CHECK:\n"
    "FIRST_FAILURE:\n"
    "ACTION:\n"
    "In DECISION, write exactly one of: APPROVE or REPLAN. "
    "In ACTION, write exactly one of: END or REPLAN. "
    "If DECISION is APPROVE, ACTION must be END. "
    "If DECISION is REPLAN, ACTION must be REPLAN."
)


class InstructionAssemblingAgent(BaseChatAgent):
    """Wrap an AssistantAgent and assemble the current turn into one user message."""

    def __init__(self, inner: AssistantAgent, per_turn_instruction: str) -> None:
        super().__init__(name=inner.name, description=inner.description)
        self._inner = inner
        self._per_turn_instruction = per_turn_instruction
        self._last_input_prompt = ""
        self._root_problem = ""

    @property
    def produced_message_types(self):
        return self._inner.produced_message_types

    def _assemble_user_message(self, messages):
        sections = []
        for msg in messages:
            if not isinstance(msg, BaseChatMessage):
                continue
            try:
                content = msg.to_model_text()
            except Exception:
                content = str(msg)
            content = (content or "").strip()
            if not content:
                continue
            if msg.source == "user":
                if not self._root_problem:
                    self._root_problem = content
                title = "Problem"
            else:
                title = f"Message from {msg.source}"
            sections.append(f"{title}:\n{content}")

        if self._root_problem:
            sections = [f"Problem:\n{self._root_problem}"] + [
                section for section in sections if section != f"Problem:\n{self._root_problem}"
            ]
        sections.append(f"Current task:\n{self._per_turn_instruction}")
        assembled = "\n\n".join(sections).strip()
        self._last_input_prompt = assembled
        return [TextMessage(content=assembled, source="user")]

    async def on_messages(self, messages, cancellation_token: CancellationToken) -> Response:
        return await self._inner.on_messages(self._assemble_user_message(messages), cancellation_token)

    async def on_messages_stream(self, messages, cancellation_token: CancellationToken):
        async for message in self._inner.on_messages_stream(
            self._assemble_user_message(messages), cancellation_token
        ):
            yield message

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._last_input_prompt = ""
        self._root_problem = ""
        await self._inner.on_reset(cancellation_token)

    @property
    def last_input_prompt(self) -> str:
        return self._last_input_prompt


def classify_verifier_response(text: str) -> str:
    normalized = (text or "").strip()
    decision_match = re.search(r"DECISION:\s*(APPROVE|REPLAN)\b", normalized)
    action_match = re.search(r"ACTION:\s*(END|REPLAN)\b", normalized)
    decision = decision_match.group(1) if decision_match else ""
    action = action_match.group(1) if action_match else ""

    if decision == "APPROVE" or action == "END" or normalized == "APPROVE":
        return "approve"
    if decision == "REPLAN" or action == "REPLAN" or normalized.startswith("REPLAN:"):
        return "replan"
    return "replan"


def classify_critique_response(text: str) -> str:
    normalized = (text or "").strip()
    decision_match = re.search(r"DECISION:\s*(FUNDAMENTAL|LOCAL)\b", normalized)
    action_match = re.search(r"ACTION:\s*(REPLAN|REVISE)\b", normalized)

    decision = decision_match.group(1) if decision_match else ""
    action = action_match.group(1) if action_match else ""

    if decision == "FUNDAMENTAL" or action == "REPLAN":
        return "replan"
    if decision == "LOCAL" and action == "REVISE":
        return "revise"
    return "replan"


def extract_answer(text: str) -> str:
    """
    Extract the final answer from solution text.
    Looks for patterns like:
    - "The answer is X"
    - "Final answer: X"
    - "Answer: X"
    - Last boxed expression \\boxed{X}
    """
    if not text:
        return ""

    def _cleanup_answer_text(answer_text: str) -> str:
        cleaned = (answer_text or "").strip().rstrip(".")
        cleaned = re.sub(r'\\[A-Za-z]+\{', '', cleaned)
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = cleaned.strip().strip("$").strip()
        cleaned = cleaned.strip("()[]")
        cleaned = cleaned.strip()
        return cleaned

    def _postprocess_explicit_answer(answer_text: str) -> str:
        cleaned = _cleanup_answer_text(answer_text)
        if "=" in cleaned:
            rhs = _cleanup_answer_text(cleaned.split("=")[-1])
            if rhs:
                return rhs
        return cleaned

    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, text)
    if boxed_matches:
        return _postprocess_explicit_answer(boxed_matches[-1])

    answer_patterns = [
        r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)',
        r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
        r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return _postprocess_explicit_answer(matches[-1])

    tail_number_patterns = [
        r'=\s*(-?\d+\.?\d*)\s*\.?\s*$',
        r'[Ii]s\s+(-?\d+\.?\d*)\s*\.?\s*$',
        r'[Aa]nswer[^\\d-]*(-?\d+\.?\d*)\s*\.?\s*$',
    ]

    for pattern in tail_number_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return _postprocess_explicit_answer(match.group(1))

    all_numbers = re.findall(r'-?\d+\.?\d*', text)
    if all_numbers:
        return _postprocess_explicit_answer(all_numbers[-1])

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if re.search(r'\d', last_line):
            return _postprocess_explicit_answer(last_line)

    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    """
    if not answer:
        return ""

    normalized = answer.lower().strip()
    normalized = re.sub(r'[,\$\s]+', '', normalized)

    numeric_match = re.search(r'-?\d+\.?\d*', normalized)
    if numeric_match:
        return numeric_match.group(0)

    return normalized


def check_answer_correctness(generated_answer: str, ground_truth_answer: str) -> bool:
    """
    Check if generated answer matches ground truth.
    """
    if not generated_answer or not ground_truth_answer:
        return False

    extracted_ground_truth = extract_answer(ground_truth_answer)
    if extracted_ground_truth:
        ground_truth_answer = extracted_ground_truth

    gen_norm = normalize_answer(generated_answer)
    gt_norm = normalize_answer(ground_truth_answer)

    if gen_norm == gt_norm:
        return True

    try:
        gen_float = float(gen_norm)
        gt_float = float(gt_norm)
        return abs(gen_float - gt_float) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def select_final_solution(messages, allowed_sources=None):
    """Select the latest non-empty final-solution message from allowed sources."""
    if allowed_sources is None:
        allowed_sources = {"revise", "solver"}

    for msg in reversed(messages or []):
        if not isinstance(msg, BaseChatMessage):
            continue
        if msg.source not in allowed_sources:
            continue

        try:
            message_text = msg.to_model_text()
        except Exception:
            message_text = str(msg)

        message_text = (message_text or "").strip()
        if not message_text:
            continue

        extracted_answer = extract_answer(message_text)
        return message_text, msg.source, extracted_answer

    return None, None, ""


async def math_graph_multi(
    env: Optional[MathEnv] = None,
    model_client_dict: dict = None,
    model_client: OpenAIChatCompletionClient = None,
    multi_logger=None,
    mode: Optional[str] = None,
    env_idx: Optional[int] = None,
    rollout_idx: Optional[int] = None,
):
    """
    Math problem solving workflow with graph:
        planner -> solver -> critique -> revise -> verifier -> (terminate or planner)

    Roles:
    - planner: make a solution plan
    - solver: solve according to the plan
    - critique: analyze the solution and identify issues
    - revise: fix the solution based on critique
    - verifier: decide whether to APPROVE or REPLAN

    Returns:
        env with final_reward assigned if env is provided
    """

    if env is None:
        raise ValueError("math_graph_multi requires a non-empty env")

    task = env.state.problem

    if multi_logger is not None and mode is not None and env_idx is not None and rollout_idx is not None:
        multi_logger.log_env_agent_info(
            mode=mode,
            env_idx=env_idx,
            rollout_idx=rollout_idx,
            turn_idx=-1,
            agent_name="user",
            message="Graph rollout started",
            extra_data={"problem": task},
        )

    # Client wiring
    default_client = model_client
    planner_client = model_client_dict.get("planner") if model_client_dict else None
    solver_client = model_client_dict.get("solver") if model_client_dict else None
    critique_client = model_client_dict.get("critique") if model_client_dict else None
    revise_client = model_client_dict.get("revise") if model_client_dict else None
    verifier_client = model_client_dict.get("verifier") if model_client_dict else None

    # Backward-compatible fallbacks
    if planner_client is None:
        planner_client = solver_client or default_client
    if solver_client is None:
        solver_client = model_client_dict.get("reasoning_generator") if model_client_dict else None
        solver_client = solver_client or default_client
    if critique_client is None:
        critique_client = model_client_dict.get("tool_generator") if model_client_dict else None
        critique_client = critique_client or default_client
    if revise_client is None:
        revise_client = solver_client or default_client
    if verifier_client is None:
        verifier_client = critique_client or default_client

    # -------------------------
    # Agent definitions
    # -------------------------
    planner_inner = AssistantAgent(
        "planner",
        model_client=planner_client,
        system_message=(
            "You are a planner for mathematical problem solving. "
            "You are not the solver. "
            "Your job is to convert the problem into a structured solving blueprint. "
            "Be strategic, method-focused, and concise. "
            "Never compute the final answer."
        ),
    )
    planner = InstructionAssemblingAgent(planner_inner, PLANNER_USER_INSTRUCTION)

    solver_inner = AssistantAgent(
        "solver",
        model_client=solver_client,
        system_message=(
            "You are a math solver. "
            "Be analytical, precise, and concise. "
            "Follow the required output schema exactly."
        ),
    )
    solver = InstructionAssemblingAgent(solver_inner, SOLVER_USER_INSTRUCTION)

    critique_inner = AssistantAgent(
        "critique",
        model_client=critique_client,
        system_message=(
            "You are a strict mathematics critic. "
            "Your job is to critique the candidate plan and the candidate solution together. "
            "Detect whether the first serious problem is a fundamental plan/modeling flaw or only a local fixable issue. "
            "Be skeptical, concrete, and actionable. "
            "Prioritize the first error that invalidates the downstream reasoning. "
            "Output a routing decision that distinguishes REPLAN from REVISE."
        ),
    )
    critique = InstructionAssemblingAgent(critique_inner, CRITIQUE_USER_INSTRUCTION)

    revise_inner = AssistantAgent(
        "revise",
        model_client=revise_client,
        system_message=(
            "You are a reviser. "
            "Be corrective, coherent, and precise. "
            "Your job is to revise the whole attempt, including the plan and the solution, when revision is requested. "
            "Follow the required output schema exactly."
        ),
    )
    revise = InstructionAssemblingAgent(revise_inner, REVISE_USER_INSTRUCTION)

    verifier_inner = AssistantAgent(
        "verifier",
        model_client=verifier_client,
        system_message=(
            "You are a strict final mathematics verifier. "
            "Do an independent check of the revised solution instead of relying on its presentation quality. "
            "Be protocol-following, skeptical, and focused on accept-vs-replan decisions. "
            "If any key step is not independently justified, reject it with REPLAN. "
            "Follow the required output schema exactly."
        ),
    )
    verifier = InstructionAssemblingAgent(verifier_inner, VERIFIER_USER_INSTRUCTION)

    end_agent = AssistantAgent(
        "end",
        model_client=solver_client,
        system_message="You are a completion marker. Just acknowledge the approved solution.",
    )

    # -------------------------
    # Graph definition
    # planner -> solver -> critique -> revise -> verifier -> (terminate / planner)
    # -------------------------
    builder = DiGraphBuilder()

    builder.add_node(planner)
    builder.add_node(solver)
    builder.add_node(critique)
    builder.add_node(revise)
    builder.add_node(verifier)
    builder.add_node(end_agent)

    builder.set_entry_point(planner)

    builder.add_edge(planner, solver)
    builder.add_edge(solver, critique)
    builder.add_edge(revise, verifier)

    def approved(msg) -> bool:
        try:
            return classify_verifier_response(msg.to_model_text()) == "approve"
        except Exception:
            return False

    def replan(msg) -> bool:
        try:
            return classify_verifier_response(msg.to_model_text()) == "replan"
        except Exception:
            return True

    builder.add_edge(critique, revise)
    builder.add_edge(verifier, end_agent, condition=approved)
    builder.add_edge(verifier, planner, condition=replan)

    graph = builder.build()

    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(20)
        | TextMentionTermination("APPROVE", sources=[verifier.name]),
    )

    # Run the workflow
    run_result = await Console(team.run_stream(task=task))

    # -------------------------
    # Logging transcript
    # -------------------------
    if multi_logger is not None and mode is not None and env_idx is not None and rollout_idx is not None:
        prompt_by_agent = {
            planner.name: planner,
            solver.name: solver,
            critique.name: critique,
            revise.name: revise,
            verifier.name: verifier,
        }
        transcript_messages = list(getattr(run_result, "messages", []) or [])
        for turn_idx, msg in enumerate(transcript_messages):
            if not isinstance(msg, BaseChatMessage):
                continue

            try:
                message_text = msg.to_model_text()
            except Exception:
                message_text = str(msg)

            input_messages = []
            for prev_msg in transcript_messages[:turn_idx]:
                if not isinstance(prev_msg, BaseChatMessage):
                    continue
                try:
                    prev_text = prev_msg.to_model_text()
                except Exception:
                    prev_text = str(prev_msg)
                input_messages.append(
                    {
                        "source": prev_msg.source,
                        "text": prev_text,
                    }
                )

            multi_logger.log_env_agent_info(
                mode=mode,
                env_idx=env_idx,
                rollout_idx=rollout_idx,
                turn_idx=turn_idx,
                agent_name=msg.source,
                message=f"{msg.source} output",
                extra_data={
                    "message_type": type(msg).__name__,
                    "input": input_messages,
                    "input_prompt": getattr(prompt_by_agent.get(msg.source), "last_input_prompt", ""),
                    "output": message_text,
                },
            )

    # -------------------------
    # Extract final solution
    # Only extract an answer from the latest solver/revise message.
    # This avoids verifier/critique text being mistaken for the final answer.
    # -------------------------
    final_solution: Optional[str] = None
    final_solution_source: Optional[str] = None
    extracted_answer = ""

    final_solution, final_solution_source, extracted_answer = select_final_solution(
        getattr(run_result, "messages", []) or [],
        allowed_sources={revise.name, solver.name},
    )

    # -------------------------
    # Evaluate final solution
    # -------------------------
    try:
        ground_truth = env.state.ground_truth_answer or ""
        ground_truth_extracted = extract_answer(ground_truth) if ground_truth else ""
        is_correct = False

        if final_solution:
            is_correct = check_answer_correctness(extracted_answer, ground_truth_extracted or ground_truth)

            env.state.reasoning_generated_solution = final_solution
            env.state.reasoning_generated_solution_history.append(final_solution)
            env.state.reasoning_extracted_answer = extracted_answer
            env.state.reasoning_extracted_answer_history.append(extracted_answer)
            env.state.reasoning_is_correct = is_correct

        final_reward = 1.0 if is_correct else 0.0
        env.state.final_reward = final_reward
        env.final_reward = final_reward

        if multi_logger is not None and mode is not None and env_idx is not None and rollout_idx is not None:
            multi_logger.log_env_agent_info(
                mode=mode,
                env_idx=env_idx,
                rollout_idx=rollout_idx,
                turn_idx=9998,
                agent_name="evaluator",
                message="Graph rollout evaluated",
                extra_data={
                    "final_solution_source": final_solution_source,
                    "final_solution": final_solution,
                    "extracted_answer": extracted_answer,
                    "ground_truth": ground_truth,
                    "ground_truth_extracted": ground_truth_extracted,
                    "is_correct": is_correct,
                    "final_reward": final_reward,
                },
            )

    except Exception as e:
        env.final_reward = 0.0

        if hasattr(env, "state"):
            try:
                env.state.final_reward = 0.0
            except Exception:
                pass

        if multi_logger is not None and mode is not None and env_idx is not None and rollout_idx is not None:
            multi_logger.log_env_agent_info(
                mode=mode,
                env_idx=env_idx,
                rollout_idx=rollout_idx,
                turn_idx=9999,
                agent_name="evaluator",
                message="Graph rollout evaluation failed",
                extra_data={
                    "error": str(e),
                    "final_solution_source": final_solution_source,
                    "final_solution": final_solution,
                },
            )

    return env


# Backward-compatible alias for older imports.
math_graph = math_graph_multi
