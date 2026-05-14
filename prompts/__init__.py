"""
prompts/ — System prompts for each agent.

Each file exports a function that builds the prompt string with dynamic variables.
Prompts are kept separate from agent logic for clarity and maintainability.

- planner_prompt.py      → Tells the LLM to create an interview plan for any role
- interviewer_prompt.py  → Tells the LLM to act as interviewer, handle edge cases
- evaluator_prompt.py    → Tells the LLM to score answers as strict JSON
- coach_prompt.py        → Tells the LLM to generate a feedback report
"""

from prompts.planner_prompt import get_planner_prompt
from prompts.interviewer_prompt import get_interviewer_prompt
from prompts.evaluator_prompt import get_evaluator_prompt
from prompts.coach_prompt import get_coach_prompt

__all__ = [
    "get_planner_prompt",
    "get_interviewer_prompt",
    "get_evaluator_prompt",
    "get_coach_prompt",
]
