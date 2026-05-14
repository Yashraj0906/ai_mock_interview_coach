"""AI Mock Interview Coach — Streamlit UI.

Run with:  streamlit run main.py
"""

# ─── Suppress noisy library warnings BEFORE any imports ────────
import warnings
import os
import logging

# Silence transformers __path__ deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")

# Suppress transformers & sentence_transformers log noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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

# ─── Constants ──────────────────────────────────────────────────
LOGO_PATH = Path(__file__).parent / "logo.jpg"

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="UpGrad Interview Assessment",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        max-width: 840px;
        margin: 0 auto;
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .logo-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .logo-row img {
        height: 48px;
        border-radius: 6px;
    }
    .brand-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06B6D4 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .brand-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.1rem;
        margin-bottom: 1.8rem;
    }

    /* ── Form card ── */
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(6,182,212,0.04) 0%, rgba(139,92,246,0.06) 100%);
        border: 1px solid rgba(139,92,246,0.18);
        border-radius: 14px;
        padding: 1.5rem;
    }

    /* ── Expander ── */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 12px;
    }

    /* ── Submit button ── */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #8B5CF6 0%, #06B6D4 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(139,92,246,0.3) !important;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: rgba(139,92,246,0.05);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 12px;
        padding: 0.7rem;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(139,92,246,0.12) !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
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
    # ── Logo + Title ──
    if LOGO_PATH.exists():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.image(str(LOGO_PATH), width="stretch")

    st.markdown(
        '<p class="brand-title">Interview Assessment</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="brand-sub">AI-Powered Mock Interview Coach · Practice any role · Get real feedback</p>',
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

    # Footer
    st.markdown(
        '<p class="app-footer">Built with LangGraph · Groq · ChromaDB — UpGrad Capstone Project</p>',
        unsafe_allow_html=True,
    )
