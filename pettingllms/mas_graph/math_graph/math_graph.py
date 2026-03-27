import re
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.agents._base_chat_agent import BaseChatAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.mas_graph.math_graph.math_env import MathEnv, MathEnvBatch


def extract_answer(text: str) -> str:
    """
    Extract the final answer from solution text.
    Looks for patterns like:
    - "The answer is X"
    - "Final answer: X"
    - "Answer: X"
    - Last boxed expression \\boxed{X}
    
    Args:
        text: Solution text
        
    Returns:
        Extracted answer or empty string
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

    # Try to find boxed answer (LaTeX style)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, text)
    if boxed_matches:
        return _postprocess_explicit_answer(boxed_matches[-1])
    
    # Try to find explicit answer statements
    answer_patterns = [
        r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)',
        r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
        r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return _postprocess_explicit_answer(matches[-1])

    # Prefer a numeric value explicitly stated at the end of the text.
    tail_number_patterns = [
        r'=\s*(-?\d+\.?\d*)\s*\.?\s*$',
        r'[Ii]s\s+(-?\d+\.?\d*)\s*\.?\s*$',
        r'[Aa]nswer[^\\d-]*(-?\d+\.?\d*)\s*\.?\s*$',
    ]
    for pattern in tail_number_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return _postprocess_explicit_answer(match.group(1))

    # Fallback to the last standalone numeric token in the text.
    all_numbers = re.findall(r'-?\d+\.?\d*', text)
    if all_numbers:
        return _postprocess_explicit_answer(all_numbers[-1])

    # If no pattern matched, try to extract last line with numbers
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Check if last line contains numbers
        if re.search(r'\d', last_line):
            return _postprocess_explicit_answer(last_line)
    
    return ""


def classify_verifier_response(text: str) -> str:
    """
    Normalize verifier output into one of:
    - "approve"
    - "needs_revision"

    Any non-compliant verifier output is treated as NEEDS_REVISION so the
    graph loops back to the solver instead of silently falling through.
    """
    normalized = (text or "").strip()
    if normalized == "APPROVE":
        return "approve"
    if normalized.startswith("NEEDS_REVISION:"):
        return "needs_revision"
    return "needs_revision"


SOLVER_USER_INSTRUCTION = (
    "The full math problem is already present earlier in this conversation. "
    "Solve that problem now; do not ask the user to provide the problem again. "
    "If you are revising, use the verifier feedback from the conversation above and fix the first concrete issue. "
    "You must end with a final line exactly in the form: Final Answer: <your answer>. "
    "The Final Answer line must be the last line of the response, with no text after it."
)


VERIFIER_USER_INSTRUCTION = (
    "The full math problem and the candidate solution are already present earlier in this conversation. "
    "Judge only the previous solution in the conversation above. "
    "Do not solve the problem again. "
    "Do not ask the user to provide the problem again. "
    "Do not restate the problem. "
    "Do not provide a new full derivation. "
    "Do not include markdown, LaTeX, bullet points, or boxed answers. "
    "Reply with exactly one of these formats only: "
    "APPROVE "
    "or "
    "NEEDS_REVISION: <brief explanation of the first concrete issue>."
)


class InstructionAppendingAgent(BaseChatAgent):
    """Wrap an AssistantAgent and assemble a single user prompt on every turn."""

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
        injected_messages = self._assemble_user_message(messages)
        return await self._inner.on_messages(injected_messages, cancellation_token)

    async def on_messages_stream(self, messages, cancellation_token: CancellationToken):
        injected_messages = self._assemble_user_message(messages)
        async for message in self._inner.on_messages_stream(injected_messages, cancellation_token):
            yield message

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._last_input_prompt = ""
        self._root_problem = ""
        await self._inner.on_reset(cancellation_token)

    @property
    def last_input_prompt(self) -> str:
        return self._last_input_prompt


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    - Remove extra whitespace
    - Convert to lowercase
    - Remove common punctuation
    - Extract numbers if present
    
    Args:
        answer: Answer string
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase and strip
    normalized = answer.lower().strip()
    
    # Remove common punctuation and symbols
    normalized = re.sub(r'[,\$\s]+', '', normalized)
    
    # Try to extract numeric value if present
    numeric_match = re.search(r'-?\d+\.?\d*', normalized)
    if numeric_match:
        return numeric_match.group(0)
    
    return normalized


def check_answer_correctness(generated_answer: str, ground_truth_answer: str) -> bool:
    """
    Check if generated answer matches ground truth.
    
    Args:
        generated_answer: Generated answer string
        ground_truth_answer: Ground truth answer string
        
    Returns:
        True if answers match, False otherwise
    """
    if not generated_answer or not ground_truth_answer:
        return False

    extracted_ground_truth = extract_answer(ground_truth_answer)
    if extracted_ground_truth:
        ground_truth_answer = extracted_ground_truth
    
    # Normalize both answers
    gen_norm = normalize_answer(generated_answer)
    gt_norm = normalize_answer(ground_truth_answer)
    
    # Direct comparison
    if gen_norm == gt_norm:
        return True
    
    # Try comparing as floats if both are numeric
    try:
        gen_float = float(gen_norm)
        gt_float = float(gt_norm)
        # Allow small floating point differences
        return abs(gen_float - gt_float) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


async def math_graph(
    env: Optional[MathEnv] = None,
    model_client_dict: dict = None,
    model_client: OpenAIChatCompletionClient = None,
    multi_logger=None,
    mode: Optional[str] = None,
    env_idx: Optional[int] = None,
    rollout_idx: Optional[int] = None,
):
    """
    Main function for math problem solving workflow using autogen.

    This workflow:
    1. Math solver generates a step-by-step solution
    2. Verifier checks the solution and provides feedback
    3. Loop continues until solution is approved or max iterations reached
    4. Extract final answer and compare with ground truth
    5. Assign final_reward (1.0 if correct, 0.0 otherwise)

    Args:
        env: Optional MathEnv instance with problem and ground truth
        model_client_dict: Dictionary of model clients for each agent {agent_name: client}
        model_client: Single model client (fallback for backward compatibility)

    Returns:
        env: Updated environment with final_reward
    """

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

    solver_client = model_client_dict.get("reasoning_generator") if model_client_dict else None
    verifier_client = model_client_dict.get("tool_generator") if model_client_dict else None

    # Define solver agent
    solver_inner = AssistantAgent(
        "reasoning_generator",
        model_client=solver_client,
        system_message=(
            "You are an expert mathematician. "
            "Be analytical, precise, and concise."
        ),
    )
    solver = InstructionAppendingAgent(solver_inner, SOLVER_USER_INSTRUCTION)

    # Define verifier agent
    verifier_inner = AssistantAgent(
        "tool_generator",
        model_client=verifier_client,
        system_message=(
            "You are a strict mathematics verifier. "
            "Be skeptical, protocol-following, and focused on finding the first concrete issue."
        ),
    )
    verifier = InstructionAppendingAgent(verifier_inner, VERIFIER_USER_INSTRUCTION)

    # Define a simple end agent to mark completion
    end_agent = AssistantAgent(
        "end",
        model_client=solver_client,
        system_message="You are a completion marker. Just acknowledge the approved solution.",
    )

    # Build graph: solver -> verifier -> (solver [if needs revision] or end [if approved])
    builder = DiGraphBuilder()

    # Add nodes
    builder.add_node(solver)
    builder.add_node(verifier)
    builder.add_node(end_agent)

    # Set solver as the entry point (required for graphs with cycles)
    builder.set_entry_point(solver)

    # Add edge from solver to verifier (unconditional)
    builder.add_edge(solver, verifier)

    # Define condition functions that accept BaseChatMessage
    def needs_revision(msg) -> bool:
        """Check if verifier requests revision"""
        try:
            return classify_verifier_response(msg.to_model_text()) == "needs_revision"
        except Exception:
            return True

    def approved(msg) -> bool:
        """Check if verifier approves the solution"""
        try:
            return classify_verifier_response(msg.to_model_text()) == "approve"
        except Exception:
            return False

    # Add conditional edges from verifier:
    # - If needs revision -> back to solver (creates a loop)
    # - If approved -> to end_agent (exit the loop)
    builder.add_edge(verifier, solver, condition=needs_revision)
    builder.add_edge(verifier, end_agent, condition=approved)

    # Build the graph
    graph = builder.build()
    
    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(15),
    )
    
    # Run the workflow and capture the autogen TaskResult (holds all messages)
    run_result = await Console(team.run_stream(task=task))
    
    # Persist the full graph transcript to per-rollout logs for easier debugging.
    if multi_logger is not None and mode is not None and env_idx is not None and rollout_idx is not None:
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
                    "input_prompt": getattr(
                        {"reasoning_generator": solver, "tool_generator": verifier}.get(msg.source, None),
                        "last_input_prompt",
                        "",
                    ),
                    "output": message_text,
                },
            )

    # Extract the final solution from the solver's last recorded message
    final_solution: Optional[str] = None
    for msg in reversed(getattr(run_result, "messages", []) or []):
        if isinstance(msg, BaseChatMessage) and msg.source == solver.name:
            final_solution = msg.to_model_text()
            break
    
    # If env is provided, evaluate the solution
    if env is not None:
        try:
            ground_truth = env.state.ground_truth_answer or ""
            ground_truth_extracted = extract_answer(ground_truth) if ground_truth else ""
            extracted_answer = ""
            is_correct = False
            
            if final_solution:
                # Extract answer from solution
                extracted_answer = extract_answer(final_solution)
                
                # Check correctness
                is_correct = check_answer_correctness(extracted_answer, ground_truth_extracted or ground_truth)
                
                # Update env state
                env.state.reasoning_generated_solution = final_solution
                env.state.reasoning_generated_solution_history.append(final_solution)
                env.state.reasoning_extracted_answer = extracted_answer
                env.state.reasoning_extracted_answer_history.append(extracted_answer)
                env.state.reasoning_is_correct = is_correct
            
            # Assign final reward: 1.0 if correct, 0.0 otherwise
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
                        "final_solution": final_solution,
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "ground_truth_extracted": ground_truth_extracted,
                        "is_correct": is_correct,
                        "final_reward": final_reward,
                    },
                )
            
        except Exception as e:
            # In case of any evaluation failure, assign zero reward
            env.final_reward = 0.0
            if multi_logger is not None and mode is not None and env_idx is not None and rollout_idx is not None:
                multi_logger.log_env_agent_info(
                    mode=mode,
                    env_idx=env_idx,
                    rollout_idx=rollout_idx,
                    turn_idx=9999,
                    agent_name="evaluator",
                    message="Graph rollout evaluation failed",
                    extra_data={"error": str(e), "final_solution": final_solution},
                )
    
    # Return env with final_reward
    if env is not None:
        return env
