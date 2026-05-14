"""Interviewer Agent — asks adaptive questions every turn, for ANY role."""


from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage

from prompts.interviewer_prompt import get_interviewer_prompt
from rag.retriever import get_question_patterns


# Model: 70B for best conversational quality
INTERVIEWER_MODEL = "llama-3.3-70b-versatile"


def interviewer_node(state: dict) -> dict:
    """LangGraph node: generates the next interview question.

    Reads evaluator feedback to adapt difficulty dynamically.
    Retrieves question patterns from RAG and adapts them to the specific role.
    """
    interview_plan = state.get("interview_plan", {})
    evaluations = state.get("evaluations", [])
    messages = state.get("messages", [])
    turn_count = state.get("turn_count", 0)

    role = interview_plan.get("role", "General")
    categories = interview_plan.get("relevant_categories", ["behavioral"])
    total_turns = interview_plan.get("num_turns", 6)

    # Step 1: Get last evaluator recommendation (if any)
    last_recommendation = None
    last_score = None
    if evaluations:
        last_eval = evaluations[-1]
        last_recommendation = last_eval.get("recommendation")
        last_score = last_eval.get("overall_score")

    # Step 2: Determine which category to pull questions from
    # Cycle through categories based on turn count
    category_index = turn_count % len(categories)
    current_category = categories[category_index]

    # If probing deeper, stay on the same category as last turn
    if last_recommendation == "probe_deeper" and turn_count > 0:
        prev_category_index = (turn_count - 1) % len(categories)
        current_category = categories[prev_category_index]

    # Step 3: Retrieve question patterns via RAG
    difficulty = interview_plan.get("difficulty", "mid-level")
    difficulty_map = {
        "entry-level": "easy",
        "mid-level": "medium",
        "senior-level": "hard",
    }
    rag_difficulty = difficulty_map.get(difficulty, "medium")

    # Adjust difficulty based on evaluator feedback
    if last_score and last_score >= 4 and last_recommendation == "increase_difficulty":
        rag_difficulty = "hard"
    elif last_score and last_score <= 2:
        rag_difficulty = "easy"

    try:
        rag_context = get_question_patterns(
            category=current_category, difficulty=rag_difficulty, top_k=3
        )
    except Exception as e:
        rag_context = f"(RAG unavailable: {e}. Generate questions from your knowledge.)"

    # Step 4: Build prompt
    prompt = get_interviewer_prompt(
        role=role,
        interview_plan=interview_plan,
        rag_context=rag_context,
        last_recommendation=last_recommendation,
        last_score=last_score,
        turn_count=turn_count,
        total_turns=total_turns,
    )

    # Step 5: Call LLM
    try:
        llm = ChatGroq(model=INTERVIEWER_MODEL, temperature=0.7)
        response = llm.invoke(
            [SystemMessage(content=prompt)] + messages
        )
        question = response.content.strip()
    except Exception as e:
        question = (
            "I appreciate your patience. Let me ask you another question — "
            f"could you tell me about a challenge you've faced in your work? "
            f"(Note: connection issue occurred: {e})"
        )

    return {
        "messages": [AIMessage(content=question)],
    }
