# Difficulty Calibration Framework

## Purpose

This framework guides how to adjust interview difficulty based on the candidate's 
background, experience level, and performance during the interview.

## Initial Difficulty Setting

Based on candidate's background information:

### Entry-Level / Intern
- **Default difficulty:** Easy to Medium
- **Question focus:** Foundational knowledge, learning ability, enthusiasm, basic problem-solving
- **STAR expectations:** Okay to draw from academic projects, internships, personal projects, volunteer work
- **Technical depth:** Focus on fundamentals, not advanced concepts
- **Example roles:** Intern, Junior Developer, Entry-Level Analyst, New Grad

### Mid-Level (2-5 years experience)
- **Default difficulty:** Medium
- **Question focus:** Independent execution, collaboration, handling complexity, growing influence
- **STAR expectations:** Professional examples expected. Should demonstrate ownership and impact.
- **Technical depth:** Should understand trade-offs, not just implementations
- **Example roles:** Software Engineer, Product Manager, Data Analyst, Designer

### Senior-Level (5+ years experience)
- **Default difficulty:** Medium to Hard
- **Question focus:** Leadership, strategic thinking, mentoring, system-level decisions, ambiguity tolerance
- **STAR expectations:** Should demonstrate cross-functional impact, mentoring, and strategic contributions
- **Technical depth:** Should discuss architecture-level decisions, long-term implications, and industry context
- **Example roles:** Senior Engineer, Lead Designer, Senior PM, Engineering Manager, Director

## Dynamic Difficulty Adjustment (During Interview)

Based on the Evaluator's recommendations:

### When to Increase Difficulty
- **Signal:** `recommendation: "increase_difficulty"` or `overall_score >= 4`
- **How:** Move to "Hard" tier questions in the next topic, or probe deeper on the current topic
- **Example transition:** "That's a great answer. Let me push a bit harder — [harder follow-up]"

### When to Probe Deeper (Maintain Difficulty)
- **Signal:** `recommendation: "probe_deeper"` or `overall_score == 2-3`
- **How:** Ask a follow-up on the SAME topic to give the candidate another chance
- **Example transition:** "Can you tell me more about the specific approach you used?" or "What metrics supported that decision?"

### When to Simplify
- **Signal:** `recommendation: "probe_deeper"` AND `overall_score == 1`
- **How:** Rephrase the question more simply, offer a hint, or move to an easier topic
- **Example transition:** "Let me rephrase that — [simpler version]" or "Here's a hint to consider: [hint]. Now, how would you approach this?"

### When to Move On
- **Signal:** `recommendation: "move_on"` or topic has been explored for 2+ turns
- **How:** Transition to the next topic from the interview plan
- **Example transition:** "Great, let's shift gears. I'd like to explore [next topic]."

## Difficulty Should Never:
- Make the candidate feel attacked or belittled
- Jump from Easy directly to Hard without Medium
- Stay at Easy for the entire interview (even for interns — push them a bit)
- Remain static — the whole point is adaptive calibration
