# 🏆 FIFA World Cup Analyst Chatbot

> A RAG-powered conversational AI for World Cup history, team statistics, and match predictions.
> Built with LangChain · LangGraph · Pinecone · OpenAI · Streamlit | Data: 1872–2025

**Live App:** [footballchatbot-genai.streamlit.app](https://footballchatbot-genai.streamlit.app)

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture Overview](#architecture-overview)
3. [Data Sources](#data-sources)
4. [Data Pipeline](#data-pipeline)
5. [Vector Store & Pinecone Indexes](#vector-store--pinecone-indexes)
6. [The 5 Tools](#the-5-tools)
7. [LangGraph ReAct Agent](#langgraph-react-agent)
8. [Prompts](#prompts)
9. [Memory System](#memory-system)
10. [Caching Notes](#caching-notes)
11. [UI Modes](#ui-modes)
12. [Setup & Deployment](#setup--deployment)

---

## What It Does

Instead of relying on the LLM's training data, this chatbot retrieves real match records from a Pinecone vector database at query time (RAG) and generates grounded answers via GPT-4o-mini.

**Capabilities:**
- Match results and goalscorers (1930–2022)
- Team statistics and all-time records
- Head-to-head comparisons across all international matches
- Match predictions with Plotly H2H bar charts
- 2026 World Cup preview (qualified teams, host cities, format)
- Multi-language responses and persistent user preferences

---

## Architecture Overview

```
User Question
     ↓
LangGraph ReAct Agent  (GPT-4o-mini as reasoning backbone)
     ↓ selects tool based on docstring descriptions
     ├── match_retrieval_tool   → worldcup-matches Pinecone index
     ├── team_stats_tool        → worldcup-team-stats Pinecone index
     ├── head_to_head_tool      → international-matches Pinecone index
     ├── match_prediction_tool  → GPT-4o-mini + H2H context + Plotly chart
     └── wc2026_tool            → worldcup-2026 Pinecone index
     ↓
Retrieved context injected into LLM prompt
     ↓
GPT-4o-mini generates grounded answer
     ↓
User sees response (+ optional Plotly chart)
```

**Stack:**
| Component | Technology |
|---|---|
| Agent framework | LangGraph (ReAct pattern) |
| LLM | GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536 dims) |
| Vector DB | Pinecone (AWS us-east-1, cosine similarity) |
| UI (notebook) | ipywidgets |
| UI (web) | Streamlit |
| Charts | Plotly |
| Data processing | pandas, numpy |

---

## Data Sources

| File | Source | Records | Purpose |
|---|---|---|---|
| `results.csv` | [Kaggle — martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 49,016 rows | All international matches 1872–2025 |
| `goalscorers.csv` | Kaggle — martj42 | ~1M rows | Every goal with scorer name and type |
| `shootouts.csv` | Kaggle — martj42 | 665 rows | Penalty shootout winners |
| `former_names.csv` | Kaggle — martj42 | ~100 rows | Historical team name mappings |
| Wikipedia 2026 | Scraped (cached locally) | ~50 facts | Qualified teams, host cities, format |

---

## Data Pipeline

### Step 1 — Name Normalization
`former_names.csv` is loaded first to build a normalization dictionary. All team names across every file are mapped to their current FIFA name before any processing begins.

```
"West Germany"  → "Germany"
"Soviet Union"  → "Russia"
"Zaire"         → "DR Congo"
```

This ensures consistent team tracking across the full 153-year dataset.

### Step 2 — Load & Filter
- `results.csv` loaded and all team names normalized
- Filtered to `tournament == "FIFA World Cup"` only — excludes qualifiers and friendlies
- Two DataFrames created: `df_wc` (~850 WC matches) and `df_all` (all 49,016 matches)

### Step 3 — Enrich with Goalscorers
`goalscorers.csv` is parsed into a lookup dictionary keyed by `(date, home_team, away_team)`. Each entry stores home and away scorer lists with annotations:

```python
"Müller"       # regular goal
"Müller (pen)" # penalty
"Müller (OG)"  # own goal
```

### Step 4 — Enrich with Shootouts
`shootouts.csv` is parsed into a lookup dictionary by match. Critical for matches like the 2022 Final where Argentina vs France ended 3-3 but Argentina won on penalties.

### Step 5 — Compute Team Stats
For each of ~80 teams, pre-computed aggregates include: appearances, matches played, wins/draws/losses, win rate %, goals scored/conceded, average goals per game, best stage reached, top 5 all-time WC scorers, and penalty shootout record.

### Step 6 — Scrape Wikipedia 2026
See [Caching Notes](#caching-notes) for scraping behavior.

### Step 7 — Sanity Checks
Automated assertions verify:
- Germany appears only as "Germany" (not "West Germany") ✅
- 2022 World Cup shows ~64 matches ✅
- Germany vs Brazil 2014 has scorer data ✅
- Brazil shows ~22 appearances, ~70 wins ✅

---

## Vector Store & Pinecone Indexes

### Hybrid Document Format
Every match is converted into a LangChain `Document` with two components:

**`page_content`** — natural language sentence for semantic embedding:
```
"In the 2014 FIFA World Cup Semi-Final in Belo Horizonte,
Germany defeated Brazil 7–1 in a dominant performance.
Scorers — Germany: Thomas Müller, Miroslav Klose, Toni Kroos (x2),
Sami Khedira, André Schürrle (x2). Brazil: Oscar."
```

**`metadata`** — structured dictionary for exact filtering:
```python
{
  "year": 2014, "stage": "Semi-Final",
  "home_team": "Germany", "away_team": "Brazil",
  "home_goals": 7, "away_goals": 1,
  "winner": "Germany", "total_goals": 8,
  "had_shootout": False, "is_wc": True
}
```

This hybrid approach enables both semantic search ("find exciting finals") and exact filtering (`stage == "Final" AND year >= 2010`).

### Indexes

| Index | Documents | Contents |
|---|---|---|
| `worldcup-matches` | ~850 | All FIFA World Cup matches with enriched scorer data |
| `international-matches` | ~49,016 | All international matches for H2H lookups |
| `worldcup-team-stats` | ~80 | Pre-computed per-team statistics |
| `worldcup-2026` | ~50 | 2026 tournament preview and qualified teams |

**All indexes:** dimension=1536, metric=cosine, cloud=AWS us-east-1

Documents are uploaded in batches of 100. Data persists in Pinecone permanently — skip the upload section on all subsequent sessions.

---

## The 5 Tools

Each tool is a Python function decorated with `@tool`. The LangGraph agent reads each tool's docstring to decide when to call it — so docstring clarity directly affects routing accuracy.

### `match_retrieval_tool`
- **Index:** `worldcup-matches`
- **Returns:** Top 8 semantically similar match documents
- **Handles:** "Who won X?", "What was the score when Y played Z?", "Who scored in the 2022 final?"

### `team_stats_tool`
- **Index:** `worldcup-team-stats`
- **Returns:** Pre-computed statistics document for the queried team
- **Handles:** "How many times did Brazil win?", "What is Germany's win rate?", "Who are Brazil's top scorers?"

### `head_to_head_tool`
- **Index:** `international-matches`
- **Returns:** Win/draw/loss counts + last 5 meetings + recent form for both teams
- **Handles:** Team comparisons, H2H record, recent form
- **Note:** Always called automatically **before** `match_prediction_tool`. Caches H2H counts for Plotly chart rendering.

### `match_prediction_tool`
- **Uses:** Output from `head_to_head_tool` + GPT-4o-mini
- **Returns:** 3-paragraph match preview with predicted winner and scoreline
- **Also:** Generates and displays a Plotly H2H bar chart inline
- **Handles:** "Predict Brazil vs Germany", "Who would win X vs Y?"

### `wc2026_tool`
- **Index:** `worldcup-2026`
- **Returns:** Relevant 2026 tournament documents
- **Handles:** "Who qualified for 2026?", "Where is the 2026 final?", "How many teams in 2026?"

### Bonus: Preference Tools
| Tool | Purpose |
|---|---|
| `set_preference_tool` | Saves language, favorite team, detail level, format to JSON |
| `get_preferences_tool` | Reads saved preferences and injects into agent context |

---

## LangGraph ReAct Agent

### Why LangGraph (not LangChain AgentExecutor)
LangChain 1.x removed `AgentExecutor` and `create_openai_functions_agent`. LangGraph is the official modern replacement, providing explicit state management and multi-step reasoning loops.

### ReAct Loop Example
```
User: "Predict Argentina vs France"
  ↓
Thought: I need H2H data first
Action: head_to_head_tool("Argentina vs France")
Observation: W12 / D8 / L9 · Last 5 meetings: [...]
  ↓
Thought: Now I can generate the prediction
Action: match_prediction_tool("Argentina vs France")
Observation: [3-paragraph preview returned]
  ↓
Final Answer: Preview shown to user + Plotly chart rendered
```

The agent autonomously chains tools — calling `head_to_head_tool` before `match_prediction_tool` without being explicitly told to, because the prediction tool's docstring signals it requires H2H context.

---

## Prompts

### System Prompt (Agent)
The agent system prompt instructs the model to:
- Always retrieve data using the appropriate tool before answering — never answer from training memory alone
- Call `head_to_head_tool` before `match_prediction_tool` for prediction queries
- Respect user preferences (language, detail level, format) when formatting responses
- Gracefully handle cases where retrieved data is sparse or ambiguous

### match_prediction_tool Prompt
```
You are a football analyst. Given the following head-to-head data:
{h2h_context}

Write a 3-paragraph match preview for {team1} vs {team2}:
1. Historical H2H record and what it tells us
2. Recent form and key players to watch
3. Prediction with scoreline and reasoning

Respond in {language}. Detail level: {detail_level}.
```

### Embedding Strategy
All documents use `text-embedding-3-small` (1536 dimensions). The natural language `page_content` is what gets embedded — the `metadata` is stored alongside for filtering but not embedded. This means semantic queries like "Brazil's biggest ever loss" surface the right match even without exact keyword overlap.

---

## Memory System

### Session Memory (Conversation Context)
Uses LangGraph's `MemorySaver` with a unique `thread_id` per conversation. All messages are stored automatically — follow-up questions work without repeating team names.

```python
# Each session gets a fresh thread
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# New conversation resets the thread
def new_session():
    st.session_state.thread_id = str(uuid.uuid4())
```

### Persistent Preferences (Cross-Session)
User preferences are saved to a local JSON file and survive across sessions.

```json
{
  "language": "Spanish",
  "favorite_team": "Brazil",
  "detail_level": "high",
  "format": "paragraph"
}
```

**Colab path:** `/content/user_prefs.json`
**Streamlit deployment path:** `user_prefs.json` (relative, in app directory)

---

## Caching Notes

### Wikipedia 2026 Scrape
The scraper applies a **2-second rate limit** between requests (responsible scraping). The raw HTML is cached locally on first run — subsequent sessions load from cache without hitting the network.

```python
CACHE_PATH = "wiki_2026_cache.html"

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        html = f.read()
else:
    response = requests.get(WIKI_URL, headers=HEADERS)
    time.sleep(2)  # rate limit
    html = response.text
    with open(CACHE_PATH, "w") as f:
        f.write(html)
```

Falls back to static hardcoded data if the scrape fails.

### Pinecone Uploads
Data is uploaded to Pinecone **once** and persists permanently in the cloud. On subsequent Colab or Streamlit sessions, skip Section 4 (vector store upload) entirely. Re-running it will upsert duplicate documents.

### H2H Chart Cache
`head_to_head_tool` caches the win/draw/loss counts in a module-level variable so that `match_prediction_tool` (called immediately after) can render the Plotly chart without making a second Pinecone query.

---

## UI Modes

### Colab Notebook (ipywidgets)
Runs directly inside Google Colab — no external tools or deployment needed.
- Reads from local CSV files (`df_all` DataFrame in memory)
- Chat history with colored messages
- Preferences sidebar
- Plotly charts render inline below predictions
- `new_session()` clears memory

### Streamlit Web App (Deployed)
Key difference: **reads H2H data from Pinecone** instead of `df_all` — works anywhere without CSV files present.

**Deployment steps:**
1. Push `app.py` + `requirements.txt` to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add `OPENAI_API_KEY` and `PINECONE_API_KEY` as secrets
4. Deploy → permanent shareable URL

**Features:** Full chat interface · Sidebar preferences · Plotly H2H charts · New conversation button · Mobile-friendly · No user setup required

---

## Setup & Deployment

### API Keys Required
| Key | Used For |
|---|---|
| `OPENAI_API_KEY` | GPT-4o-mini (generation) + text-embedding-3-small (embeddings) |
| `PINECONE_API_KEY` | Vector database read/write |

**In Colab:** Use Colab Secrets (never hardcode keys)
**In Streamlit Cloud:** Add via the Secrets manager in the dashboard

### requirements.txt
```
langchain
langgraph
langchain-openai
langchain-pinecone
pinecone
streamlit
pandas
numpy
plotly
beautifulsoup4
requests
```

> **Note on version pinning:** Colab pre-installs packages like `tensorflow` and `numba` that conflict with pinned `numpy` versions. Let pip resolve compatible versions automatically — do not pin numpy explicitly.

---

*For questions or issues, open a GitHub issue on [nithinraja76/football_chatbot](https://github.com/nithinraja76/football_chatbot).*
