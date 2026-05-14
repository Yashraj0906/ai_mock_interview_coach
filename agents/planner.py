"""Session Planner Agent — runs once at start, creates interview plan for ANY role."""

import json

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from prompts.planner_prompt import get_planner_prompt
from rag.retriever import get_frameworks


# Model: 70B for high reasoning quality (runs only once)
PLANNER_MODEL = "llama-3.3-70b-versatile"


def planner_node(state: dict) -> dict:
    """LangGraph node: generates an interview plan based on candidate info.

    Uses RAG to retrieve universal frameworks, then leverages the LLM's
    world knowledge to understand ANY role and map it to interview categories.
    """
    candidate_info = state.get("candidate_info", {})
    role = candidate_info.get("role", "General")
    background = candidate_info.get("background", "")
    focus_area = candidate_info.get("focus_area", "mixed")

    # Step 1: Retrieve universal interview frameworks via RAG
    try:
        rag_context = get_frameworks(top_k=5)
    except Exception as e:
        rag_context = f"(RAG unavailable: {e}. Use your knowledge of interview best practices.)"

    # Step 2: Build the planner prompt
    prompt = get_planner_prompt(role, background, focus_area, rag_context)

    # Step 3: Call LLM
    try:
        llm = ChatGroq(model=PLANNER_MODEL, temperature=0.3)
        response = llm.invoke([SystemMessage(content=prompt)])
        plan_text = response.content.strip()

        # Clean potential markdown formatting
        if plan_text.startswith("```"):
            plan_text = plan_text.split("```")[1]
            if plan_text.startswith("json"):
                plan_text = plan_text[4:]
            plan_text = plan_text.strip()

        interview_plan = json.loads(plan_text)

    except json.JSONDecodeError:
        # Fallback plan if LLM doesn't produce valid JSON
        interview_plan = {
            "role": role,
            "relevant_categories": ["behavioral", "situational"],
            "topics": ["general competency", "problem solving", "teamwork"],
            "difficulty": "mid-level",
            "question_types": ["behavioral", "situational"],
            "num_turns": 6,
            "role_context": f"Interview for the {role} role, assessing general competencies.",
        }
    except Exception as e:
        interview_plan = {
            "role": role,
            "relevant_categories": ["behavioral", "situational"],
            "topics": ["general competency", "problem solving", "teamwork"],
            "difficulty": "mid-level",
            "question_types": ["behavioral", "situational"],
            "num_turns": 6,
            "role_context": f"Interview for {role}. (Planner error: {e})",
        }

    return {
        "interview_plan": interview_plan,
        "rag_context": rag_context,
        "current_phase": "interviewing",
    }
