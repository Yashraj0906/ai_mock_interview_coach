"""AI Mock Interview Coach — Streamlit UI.

Run with:  streamlit run main.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from graph import compile_graph, AgentState
from agents.planner import planner_node
from agents.interviewer import interviewer_node
from agents.evaluator import evaluator_node
from agents.coach import coach_node

load_dotenv()

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Mock Interview Coach",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stExpander"] {
        border: 2px solid #667eea;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ────────────────────────────────
if "phase" not in st.session_state:
    st.session_state.phase = "intake"  # intake → interviewing → completed
    st.session_state.state = None
    st.session_state.messages = []
    st.session_state.interview_plan = None
    st.session_state.evaluations = []
    st.session_state.turn_count = 0


# ═══════════════════════════════════════════════════════════════
# PHASE 1: INTAKE FORM
# ═══════════════════════════════════════════════════════════════
if st.session_state.phase == "intake":
    st.markdown('<p class="main-title">🎯 AI Mock Interview Coach</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Practice for any role. Get real feedback. Improve faster.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    with st.form("intake_form"):
        role = st.text_input(
            "🎯 Target Role",
            placeholder="e.g., Product Manager, UX Designer, HR Business Partner, DevOps Engineer...",
            help="Type any role — the system adapts to it dynamically.",
        )

        background = st.text_area(
            "📝 Your Background (optional)",
            placeholder="e.g., 2 years as a frontend developer, CS degree, led a 3-person team...",
            help="A brief 2-3 line summary helps calibrate difficulty.",
            height=100,
        )

        focus_area = st.radio(
            "🎯 Focus Area",
            options=["Mixed", "Behavioral", "Technical", "Case Study"],
            horizontal=True,
            index=0,
            help="Choose what type of questions you want to practice.",
        )

        submitted = st.form_submit_button(
            "🚀 Start Interview", use_container_width=True
        )

    if submitted:
        if not role.strip():
            st.error("⚠️ Please enter a target role to begin.")
        else:
            with st.spinner("🧠 Planning your interview session..."):
                # Initialize the graph state
                initial_state = {
                    "messages": [],
                    "candidate_info": {
                        "role": role.strip(),
                        "background": background.strip(),
                        "focus_area": focus_area.lower().replace(" ", "_"),
                    },
                    "turn_count": 0,
                    "evaluations": [],
                    "interview_plan": {},
                    "rag_context": "",
                    "current_phase": "planning",
                }

                # Run Planner
                planner_result = planner_node(initial_state)
                initial_state.update(planner_result)

                st.session_state.interview_plan = initial_state["interview_plan"]

                # Run Interviewer (first question)
                interviewer_result = interviewer_node(initial_state)
                first_question = interviewer_result["messages"][0].content

                # Save state
                st.session_state.messages = [
                    {"role": "assistant", "content": first_question}
                ]
                st.session_state.state = initial_state
                st.session_state.state["messages"] = [AIMessage(content=first_question)]
                st.session_state.phase = "interviewing"
                st.session_state.evaluations = []
                st.session_state.turn_count = 0

            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PHASE 2: INTERVIEW CHAT
# ═══════════════════════════════════════════════════════════════
elif st.session_state.phase == "interviewing":
    plan = st.session_state.interview_plan
    role = plan.get("role", "")
    max_turns = plan.get("num_turns", 6)

    st.markdown(f"### 🎙️ Mock Interview — {role}")

    # Progress indicator
    progress = min(st.session_state.turn_count / max_turns, 1.0)
    st.progress(progress, text=f"Turn {st.session_state.turn_count}/{max_turns}")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Type your answer..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add to LangGraph state
        st.session_state.state["messages"].append(HumanMessage(content=user_input))

        with st.spinner("🤔 Evaluating your answer..."):
            # Step 1: Run Evaluator (silent — no UI output)
            eval_result = evaluator_node(st.session_state.state)
            st.session_state.state["evaluations"] = (
                st.session_state.state.get("evaluations", []) + eval_result["evaluations"]
            )
            st.session_state.state["turn_count"] = eval_result["turn_count"]
            st.session_state.evaluations = st.session_state.state["evaluations"]
            st.session_state.turn_count = eval_result["turn_count"]

        # Step 2: Check if interview should end
        if st.session_state.turn_count >= max_turns:
            with st.spinner("📝 Generating your feedback report..."):
                coach_result = coach_node(st.session_state.state)
                feedback = coach_result["messages"][0].content

                st.session_state.messages.append(
                    {"role": "assistant", "content": feedback}
                )
                st.session_state.state["messages"].append(
                    AIMessage(content=feedback)
                )
                st.session_state.phase = "completed"
        else:
            with st.spinner("🎤 Preparing next question..."):
                # Step 3: Run Interviewer for next question
                interviewer_result = interviewer_node(st.session_state.state)
                next_question = interviewer_result["messages"][0].content

                st.session_state.messages.append(
                    {"role": "assistant", "content": next_question}
                )
                st.session_state.state["messages"].append(
                    AIMessage(content=next_question)
                )

        st.rerun()


# ═══════════════════════════════════════════════════════════════
# PHASE 3: FEEDBACK & RESULTS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.phase == "completed":
    plan = st.session_state.interview_plan
    role = plan.get("role", "")

    st.markdown(f"### ✅ Interview Complete — {role}")
    st.progress(1.0, text="Interview finished!")

    # Display full chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Score visualization
    st.markdown("---")
    with st.expander("📊 Turn-by-Turn Scores", expanded=True):
        if st.session_state.evaluations:
            import pandas as pd

            eval_data = []
            for e in st.session_state.evaluations:
                eval_data.append({
                    "Turn": f"Turn {e.get('turn', '?')}",
                    "Overall": e.get("overall_score", 0),
                    "Clarity": e.get("clarity_score", 0),
                    "Depth": e.get("depth_score", 0),
                    "Relevance": e.get("relevance_score", 0),
                })

            df = pd.DataFrame(eval_data)
            st.bar_chart(df.set_index("Turn"))

            # Average scores
            avg = df[["Overall", "Clarity", "Depth", "Relevance"]].mean()
            cols = st.columns(4)
            for i, (metric, val) in enumerate(avg.items()):
                cols[i].metric(metric, f"{val:.1f}/5")

    # Restart button
    st.markdown("---")
    if st.button("🔄 Start a New Interview", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
