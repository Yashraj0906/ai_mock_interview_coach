"""Evaluator Agent — silently scores every answer. Never visible to user."""

import json

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from prompts.evaluator_prompt import get_evaluator_prompt
from rag.retriever import get_rubric


# Model: 8B for speed — structured JSON output only
EVALUATOR_MODEL = "llama-3.3-70b-versatile"


class EvaluationResult(BaseModel):
    """Pydantic schema for structured evaluation output."""

    clarity_score: int = Field(ge=1, le=5, description="Clarity of the answer (1-5)")
    depth_score: int = Field(ge=1, le=5, description="Depth and specificity (1-5)")
    relevance_score: int = Field(ge=1, le=5, description="Relevance to the question (1-5)")
    overall_score: int = Field(ge=1, le=5, description="Holistic assessment (1-5)")
    recommendation: str = Field(
        description="One of: probe_deeper, move_on, increase_difficulty"
    )
    notes: str = Field(description="Brief evaluator notes for the Coach")


def evaluator_node(state: dict) -> dict:
    """LangGraph node: evaluates the latest candidate answer.

    Runs SILENTLY after every candidate response. Never produces
    a message visible to the user — only writes to state["evaluations"].
    """
    messages = state.get("messages", [])
    interview_plan = state.get("interview_plan", {})
    turn_count = state.get("turn_count", 0)

    role = interview_plan.get("role", "General")
    role_context = interview_plan.get("role_context", "")
    categories = interview_plan.get("relevant_categories", ["behavioral"])

    # Determine current category for rubric retrieval
    category_index = turn_count % len(categories) if categories else 0
    current_category = categories[category_index] if categories else "behavioral"

    # Step 1: Retrieve scoring rubric via RAG
    try:
        rag_context = get_rubric(category=current_category, top_k=3)
    except Exception as e:
        rag_context = f"(Rubric unavailable: {e}. Use standard scoring criteria: 1=poor, 3=average, 5=excellent.)"

    # Step 2: Build prompt
    prompt = get_evaluator_prompt(rag_context, role, role_context)

    # Step 3: Call LLM with structured output
    try:
        llm = ChatGroq(model=EVALUATOR_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(EvaluationResult)
        evaluation = structured_llm.invoke(
            [SystemMessage(content=prompt)] + messages
        )
        eval_dict = evaluation.model_dump()

    except Exception:
        # Fallback: try raw JSON parsing
        try:
            llm = ChatGroq(model=EVALUATOR_MODEL, temperature=0)
            response = llm.invoke(
                [SystemMessage(content=prompt)] + messages
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            eval_dict = json.loads(raw)
            # Validate with Pydantic
            validated = EvaluationResult(**eval_dict)
            eval_dict = validated.model_dump()
        except Exception:
            # Ultimate fallback — neutral evaluation
            eval_dict = {
                "clarity_score": 3,
                "depth_score": 3,
                "relevance_score": 3,
                "overall_score": 3,
                "recommendation": "move_on",
                "notes": "Evaluation fallback — could not parse LLM response.",
            }

    # Ensure recommendation is valid
    valid_recs = {"probe_deeper", "move_on", "increase_difficulty"}
    if eval_dict.get("recommendation") not in valid_recs:
        eval_dict["recommendation"] = "move_on"

    # Add turn number
    eval_dict["turn"] = turn_count + 1

    return {
        "evaluations": [eval_dict],
        "turn_count": turn_count + 1,
    }
