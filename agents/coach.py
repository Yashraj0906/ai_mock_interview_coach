"""Coach Agent — generates comprehensive feedback after the interview ends."""

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage

from prompts.coach_prompt import get_coach_prompt


# Model: 70B for highest quality feedback
COACH_MODEL = "llama-3.3-70b-versatile"


def coach_node(state: dict) -> dict:
    """LangGraph node: generates a structured feedback report.

    Triggered when turn_count reaches the planned number of turns.
    Synthesizes ALL evaluations and the full conversation into actionable feedback.
    """
    messages = state.get("messages", [])
    evaluations = state.get("evaluations", [])
    interview_plan = state.get("interview_plan", {})

    role = interview_plan.get("role", "General")

    # Step 1: Build evaluation summary for the Coach's context
    eval_summary = "## Evaluation Data (from silent Evaluator agent)\n\n"
    for e in evaluations:
        eval_summary += (
            f"- Turn {e.get('turn', '?')}: "
            f"Overall={e.get('overall_score', '?')}/5, "
            f"Clarity={e.get('clarity_score', '?')}/5, "
            f"Depth={e.get('depth_score', '?')}/5, "
            f"Recommendation={e.get('recommendation', '?')}, "
            f"Notes: {e.get('notes', 'N/A')}\n"
        )

    # Step 2: Build prompt
    system_prompt = get_coach_prompt(role, interview_plan)
    full_prompt = f"{system_prompt}\n\n{eval_summary}"

    # Step 3: Call LLM
    try:
        llm = ChatGroq(model=COACH_MODEL, temperature=0.5)
        response = llm.invoke(
            [SystemMessage(content=full_prompt)] + messages
        )
        feedback = response.content.strip()
    except Exception as e:
        # Fallback feedback
        avg_score = (
            sum(e.get("overall_score", 3) for e in evaluations) / len(evaluations)
            if evaluations
            else 3
        )
        feedback = (
            f"## 📋 Interview Feedback Report — {role}\n\n"
            f"### Overall Performance: {avg_score:.1f}/5\n\n"
            f"We encountered an issue generating detailed feedback ({e}). "
            f"Based on your {len(evaluations)} responses, your average score was "
            f"{avg_score:.1f}/5. Please try again for a detailed breakdown."
        )

    return {
        "messages": [AIMessage(content=feedback)],
        "current_phase": "completed",
    }
