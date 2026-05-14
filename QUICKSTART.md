# COMPLETE PROJECT GUIDE — AI Mock Interview Coach

> Everything explained: what each file does, how RAG works, how agents connect, and how to run it.

---

## PART 1: What Each File Does

### Core Files

| File | Purpose | When It Runs |
|------|---------|:---:|
| `main.py` | **Streamlit UI** — the app the user sees. Has 3 screens: intake form, chat, feedback panel. Manages `st.session_state` to keep the interview alive across reruns. | When you do `streamlit run main.py` |
| `graph.py` | **LangGraph state machine** — defines the `AgentState` (what data flows between agents) and wires the 4 agents together with edges and conditional routing. | Imported by `main.py` |

### Agents (agents/)

| File | Agent | Model | What It Does | Output |
|------|-------|-------|-------------|--------|
| `planner.py` | Session Planner | `llama-3.3-70b-versatile` | Takes ANY role the user types → uses LLM's world knowledge to figure out what topics/skills matter → outputs a JSON plan | `{"role": "HR Manager", "topics": [...], "difficulty": "mid-level", ...}` |
| `interviewer.py` | Interviewer | `llama-3.3-70b-versatile` | Reads the plan + evaluator feedback → generates the next interview question → adapts difficulty dynamically | A natural-sounding interview question |
| `evaluator.py` | Evaluator | `llama-3.1-8b-instant` | Silently scores every answer → uses Pydantic `with_structured_output()` to force valid JSON → sends recommendation to Interviewer | `{"overall_score": 3, "recommendation": "probe_deeper", ...}` |
| `coach.py` | Coach | `llama-3.3-70b-versatile` | Runs at the end → reads ALL evaluations + full chat → generates detailed Markdown feedback report | Formatted feedback with strengths, gaps, practice tips |

### Prompts (prompts/)

| File | What It Contains |
|------|-----------------|
| `planner_prompt.py` | System prompt telling the LLM: "You are an Expert Interview Session Planner. Given this role, determine topics, difficulty, categories." Forces JSON-only output. |
| `interviewer_prompt.py` | System prompt telling the LLM: "You are a senior interviewer for {role}." Includes ALL edge case handling (vague answers, "I don't know", off-topic, rambling). Reads evaluator recommendation. |
| `evaluator_prompt.py` | System prompt telling the LLM: "Score this answer on clarity, depth, relevance. Output JSON ONLY." References the Pydantic schema. |
| `coach_prompt.py` | System prompt telling the LLM: "Synthesize all turns into a feedback report. Be specific, not generic." Forces Markdown format. |

### RAG Pipeline (rag/)

| File | What It Does |
|------|-------------|
| `ingest.py` | **Run ONCE.** Reads all 14 markdown files from `knowledge_base/` → splits into chunks → embeds with `all-MiniLM-L12-v2` → stores in ChromaDB. Creates the vector database. |
| `retriever.py` | **Called by agents during interviews.** Has 3 functions: `get_frameworks()`, `get_question_patterns()`, `get_rubric()`. Each queries ChromaDB with metadata filters and returns relevant text. |

### Knowledge Base (knowledge_base/) — THE RAG DATA

| Folder | Files | What They Contain |
|--------|-------|-------------------|
| `question_patterns/` | 6 files | Universal interview question TEMPLATES organized by category (behavioral, technical, situational, case, leadership, culture_fit). NOT role-specific — the LLM adapts them to any role. |
| `rubrics/` | 5 files | Scoring criteria with 1-5 scales. Each score level is described in detail. Used by the Evaluator to ground its scoring. |
| `frameworks/` | 3 files | STAR method explanation, competency-to-category mapping table, difficulty calibration rules. Used by the Planner to structure interviews. |

---

## PART 2: How RAG Works (Step by Step)

### What Is RAG?

RAG = **Retrieval-Augmented Generation**. Instead of relying only on what the LLM knows, we give it specific documents to reference. This makes answers more grounded and consistent.

### Where Is the Data?

The data is the **14 markdown files** you can see in `knowledge_base/`. These are hand-written documents containing:
- Question templates (like "Tell me about a time you [X]")
- Scoring rubrics (like "Score 1 = poor, Score 5 = excellent")
- Interview frameworks (like "Use the STAR method")

**There is no external data source.** The knowledge base IS the documents we wrote. They are universal (not role-specific), so they work for ANY role.

### How Data Gets INTO ChromaDB (Ingestion)

When you run `python -m rag.ingest`, this happens:

```
knowledge_base/behavioral_patterns.md (raw text, ~2000 words)
         |
         v
    [CHUNKING] — split into ~300 word pieces with 50 word overlap
         |
         v
    Chunk 1: "## Pattern: Leadership ### Easy - Tell me about..."
    Chunk 2: "### Medium - Give me an example of a time you..."
    Chunk 3: "### Hard - Tell me about a time you made an..."
         |
         v
    [EMBEDDING] — sentence-transformers converts each chunk to a 384-number vector
         |
         v
    Chunk 1 → [0.023, -0.156, 0.089, ..., 0.045]  (384 numbers)
    Chunk 2 → [0.112, -0.034, 0.201, ..., -0.078]  (384 numbers)
         |
         v
    [METADATA] — each chunk gets tagged:
    {"category": "behavioral", "doc_type": "question_pattern", "difficulty": "easy"}
         |
         v
    [CHROMADB] — stored in a local database file (chroma_db/ folder)
    Total: 31 chunks from 14 files
```

### How Data Gets OUT of ChromaDB (Retrieval)

When an agent needs context during an interview:

```
Interviewer needs a behavioral question pattern
         |
         v
    retriever.get_question_patterns(category="behavioral", difficulty="medium")
         |
         v
    [QUERY EMBEDDING] — "behavioral interview questions medium level" → [0.045, ...]
         |
         v
    [COSINE SIMILARITY] — ChromaDB compares this vector to all 31 stored vectors
         |
         v
    [METADATA FILTER] — only chunks where category="behavioral" AND doc_type="question_pattern"
         |
         v
    [TOP 3] — returns the 3 most similar chunks as plain text
         |
         v
    Agent receives:
    "--- Context 1 [behavioral | behavioral_patterns.md] ---
     ## Pattern: Leadership ### Medium - Tell me about a time you had to
     influence a decision without having formal authority..."
```

### Why RAG Instead of Just Using the LLM?

| Without RAG | With RAG |
|---|---|
| LLM invents question structures from scratch each time | LLM adapts proven STAR-format templates |
| Evaluator scores based on vibes | Evaluator scores against documented 1-5 rubrics |
| Planner guesses difficulty calibration | Planner follows documented difficulty framework |
| Inconsistent across sessions | Consistent quality grounded in documents |

---

## PART 3: How Everything Connects (Flow)

```
USER types "HR Manager" + clicks Start
    |
    v
[main.py] creates initial state → calls planner_node()
    |
    v
[planner.py] → calls retriever.get_frameworks() → gets STAR/competency docs from ChromaDB
            → sends to LLM with prompt: "Plan interview for HR Manager"
            → LLM returns JSON: {topics: ["talent acquisition", "employee relations", ...]}
    |
    v
[main.py] calls interviewer_node() for first question
    |
    v
[interviewer.py] → calls retriever.get_question_patterns("behavioral") → gets templates
                → sends to LLM with prompt: "You are an interviewer for HR Manager. Ask Q1."
                → LLM returns: "Welcome! Let's start. Tell me about a time you..."
    |
    v
[main.py] shows question in chat UI → user types answer
    |
    v
[main.py] calls evaluator_node()
    |
    v
[evaluator.py] → calls retriever.get_rubric("behavioral") → gets scoring criteria
              → sends to LLM: "Score this answer. Output JSON only."
              → LLM returns: {"overall_score": 2, "recommendation": "probe_deeper"}
              → THIS IS NEVER SHOWN TO THE USER — only stored in state
    |
    v
[main.py] checks: turn_count < 6? → YES → calls interviewer_node() again
    |
    v
[interviewer.py] reads recommendation="probe_deeper" from evaluator
               → asks a follow-up: "Can you give me a more specific example?"
    |
    v
    ... (repeats for 5-7 turns) ...
    |
    v
[main.py] checks: turn_count >= 6? → YES → calls coach_node()
    |
    v
[coach.py] → reads ALL evaluations + full chat history
           → sends to LLM: "Generate feedback report"
           → returns Markdown with strengths, gaps, practice tips, turn-by-turn table
    |
    v
[main.py] shows feedback in expandable panel + bar chart of scores
```

---

## PART 4: Setup & Run (with uv)

### Step 1: Go to project folder
```powershell
cd "C:\Users\Yashraj Kumar\Downloads\UpGrad\ai_interview_coach"
```

### Step 2: Create uv environment
```powershell
uv venv
```

### Step 3: Activate it
```powershell
.venv\Scripts\activate
```

### Step 4: Install dependencies
```powershell
uv pip install -r requirements.txt
```

### Step 5: Create .env file
```powershell
copy .env.example .env
```
Open `.env` in any editor and replace with your actual Groq key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```
Get free key: https://console.groq.com/keys

### Step 6: Run RAG ingestion (ONCE)
```powershell
python -m rag.ingest
```
Expected output:
```
[*] Loading embedding model: all-MiniLM-L12-v2...
[+] Processing: frameworks/star_method.md
[+] Processing: question_patterns/behavioral_patterns.md
... (14 files total)
[*] Embedding 31 chunks...
[OK] Ingestion complete! 31 chunks stored in 'interview_knowledge'
```

### Step 7: Launch the app
```powershell
streamlit run main.py
```
Opens at http://localhost:8501

---

## PART 5: What You Still Need to Do

1. **Test with 3 different roles** — PM, SWE intern, and a non-standard role (HR/UX)
2. **Copy those 3 transcripts** into the README
3. **Write README.md** — I can help you generate this after testing
4. **Push to GitHub** — see below

---

## PART 6: GitHub Setup

Your repo: https://github.com/Yashraj0906/ai_mock_interview_coach.git

```powershell
cd "C:\Users\Yashraj Kumar\Downloads\UpGrad\ai_interview_coach"
git init
git remote add origin https://github.com/Yashraj0906/ai_mock_interview_coach.git
git add .
git commit -m "feat: complete AI Mock Interview Coach - multi-agent system with RAG"
git branch -M main
git push -u origin main
```

> Make sure your `.gitignore` excludes `.env` and `chroma_db/` (it does).
