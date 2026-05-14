"""
agents/ — The 4 AI agents that power the interview system.

Each agent is a LangGraph node (a Python function that reads state and returns updates).

- planner.py   → Runs ONCE at start. Takes any role, creates interview plan.
- interviewer.py → Runs EVERY turn. Asks adaptive questions.
- evaluator.py  → Runs EVERY turn (silently). Scores answers as JSON.
- coach.py      → Runs ONCE at end. Generates feedback report.
"""

from agents.planner import planner_node
from agents.interviewer import interviewer_node
from agents.evaluator import evaluator_node
from agents.coach import coach_node

__all__ = ["planner_node", "interviewer_node", "evaluator_node", "coach_node"]
