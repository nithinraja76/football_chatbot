"""
app.py — FIFA World Cup Analyst Chatbot
Streamlit Cloud deployment — fully self-contained
No Google Drive or local files needed
All data served from Pinecone cloud indexes
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pinecone import Pinecone

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup Chatbot",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Keys from Streamlit Secrets ───────────────────────
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY",""))
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.environ.get("PINECONE_API_KEY",""))

os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("❌ API keys missing. Add OPENAI_API_KEY and PINECONE_API_KEY to Streamlit secrets.")
    st.stop()

# ── Index names ───────────────────────────────────────────
INDEXES = {
    "wc_matches":   "worldcup-matches",
    "intl_matches": "international-matches",
    "team_stats":   "worldcup-team-stats",
    "wc_2026":      "worldcup-2026",
}

# ── Cached resources — init once per session ──────────────
@st.cache_resource
def init_resources():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    pc         = Pinecone(api_key=PINECONE_API_KEY)
    return embeddings, llm, pc

embeddings, llm, pc = init_resources()

def get_vs(label: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=INDEXES[label],
        embedding=embeddings
    )

# ── User preferences — stored in session state ────────────
def load_prefs() -> dict:
    return st.session_state.get("prefs", {
        "favorite_team":    "",
        "language":         "English",
        "detail_level":     "medium",
        "preferred_format": "paragraph",
    })

def save_pref(key: str, value: str):
    prefs      = load_prefs()
    prefs[key] = value
    st.session_state["prefs"] = prefs


# ══════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════

@tool
def match_retrieval_tool(query: str) -> str:
    """
    Searches FIFA World Cup match history from 1930 to 2022.
    Use for questions about specific match results, scores,
    goalscorers, penalty shootouts, or any historical World Cup fact.
    Do NOT use for team statistics or 2026 preview questions.
    Input: natural language question about a World Cup match.
    Returns: relevant match records with scorer details.
    """
    vs      = get_vs("wc_matches")
    results = vs.similarity_search_with_score(query, k=8)
    if not results:
        return "No relevant World Cup match records found."
    filtered = [(doc, score) for doc, score in results if score < 1.0]
    if not filtered:
        return "No closely matching records found. Try rephrasing with specific team names or years."
    return "\n\n".join(
        f"[relevance={round(1-score,3)}] {doc.page_content}"
        for doc, score in filtered
    )


@tool
def team_stats_tool(query: str) -> str:
    """
    Retrieves FIFA World Cup statistics for one or more teams.
    Use for questions about win rates, tournament appearances,
    goals scored, best stage reached, all-time top scorers,
    or penalty shootout records in World Cups.
    Input: question about team stats e.g. Brazil World Cup record.
    Returns: detailed team statistics.
    """
    vs      = get_vs("team_stats")
    results = vs.similarity_search_with_score(query, k=5)
    if not results:
        return "No team statistics found."
    filtered = [(doc, score) for doc, score in results if score < 1.2]
    if not filtered:
        return "Could not find stats. Use full country name e.g. Germany not GER."
    return "\n\n".join(
        f"[relevance={round(1-score,3)}] {doc.page_content}"
        for doc, score in filtered
    )


@tool
def head_to_head_tool(query: str) -> str:
    """
    Retrieves head-to-head history and recent form for two teams
    from the international matches database.
    Always call this BEFORE match_prediction_tool.
    Input format: Team A vs Team B e.g. Brazil vs Germany.
    Returns: H2H record and recent matches for each team.
    """
    if "vs" not in query.lower():
        return "Input must be 'Team A vs Team B'"

    parts  = query.split("vs", 1)
    team_a = parts[0].strip().title()
    team_b = parts[1].strip().title()

    # Search international matches index for H2H
    vs      = get_vs("intl_matches")
    h2h_q   = f"{team_a} vs {team_b} match result"
    results = vs.similarity_search_with_score(h2h_q, k=20)

    # Filter to only matches between these two teams
    h2h_docs = [
        doc for doc, score in results
        if (team_a.lower() in doc.metadata.get("home_team","").lower() and
            team_b.lower() in doc.metadata.get("away_team","").lower()) or
           (team_b.lower() in doc.metadata.get("home_team","").lower() and
            team_a.lower() in doc.metadata.get("away_team","").lower())
    ]

    if not h2h_docs:
        # Fallback — broader search
        r2 = vs.similarity_search(f"{team_a} {team_b}", k=10)
        h2h_docs = [
            doc for doc in r2
            if (team_a.lower() in doc.metadata.get("home_team","").lower() or
                team_a.lower() in doc.metadata.get("away_team","").lower()) and
               (team_b.lower() in doc.metadata.get("home_team","").lower() or
                team_b.lower() in doc.metadata.get("away_team","").lower())
        ]

    if not h2h_docs:
        a_wins = b_wins = draws = 0
        h2h_text = f"No international meetings found between {team_a} and {team_b}."
    else:
        a_wins = sum(
            doc.metadata.get("winner","").lower() == team_a.lower()
            for doc in h2h_docs
        )
        b_wins = sum(
            doc.metadata.get("winner","").lower() == team_b.lower()
            for doc in h2h_docs
        )
        draws  = len(h2h_docs) - a_wins - b_wins
        matches_text = "\n".join(
            f"  {doc.metadata.get('date','')} ({doc.metadata.get('tournament','')}): "
            f"{doc.metadata.get('home_team','')} "
            f"{int(doc.metadata.get('home_goals',0))}–"
            f"{int(doc.metadata.get('away_goals',0))} "
            f"{doc.metadata.get('away_team','')}"
            for doc in h2h_docs[:5]
        )
        h2h_text = (
            f"Head-to-Head ({len(h2h_docs)} meetings found):\n"
            f"  {team_a}: {a_wins}W | Draws: {draws} | {team_b}: {b_wins}W\n\n"
            f"Recent meetings:\n{matches_text}"
        )

    # Store for chart
    st.session_state["h2h_cache"] = {
        "team_a": team_a, "team_b": team_b,
        "a_wins": a_wins, "b_wins": b_wins,
        "draws":  draws,  "total": len(h2h_docs),
    }

    # Recent form from WC matches
    vs_wc = get_vs("wc_matches")
    form_a_docs = vs_wc.similarity_search(f"{team_a} recent matches", k=5,
                  filter={"home_team": team_a})
    form_b_docs = vs_wc.similarity_search(f"{team_b} recent matches", k=5,
                  filter={"home_team": team_b})

    form_a = "\n".join(f"  • {d.page_content[:80]}..." for d in form_a_docs[:3]) or f"  No recent WC data for {team_a}"
    form_b = "\n".join(f"  • {d.page_content[:80]}..." for d in form_b_docs[:3]) or f"  No recent WC data for {team_b}"

    return (
        f"{h2h_text}\n\n"
        f"{team_a} Recent WC Matches:\n{form_a}\n\n"
        f"{team_b} Recent WC Matches:\n{form_b}"
    )


PREDICTION_PROMPT = PromptTemplate(
    input_variables=["team_a","team_b","h2h_data",
                     "language","detail_level","preferred_format","favorite_team"],
    template="""You are a professional FIFA World Cup analyst.
Write a match preview in exactly 3 paragraphs:

Paragraph 1 — Head-to-head history and rivalry patterns
Paragraph 2 — Recent form and key strengths of each team
Paragraph 3 — Predicted winner, scoreline, and clear reasoning

Language: {language} | Detail: {detail_level} | Format: {preferred_format}
Favorite team: {favorite_team}

Teams: {team_a} vs {team_b}
Statistical data:
{h2h_data}

Match Preview:"""
)

@tool
def match_prediction_tool(query: str) -> str:
    """
    Generates a full match preview and predicted outcome for two teams.
    Always call head_to_head_tool first before calling this.
    Input format: Team A vs Team B e.g. Argentina vs France.
    Returns: 3-paragraph match preview with predicted winner and scoreline.
    """
    if "vs" not in query.lower():
        return "Input must be 'Team A vs Team B'"

    parts  = query.split("vs", 1)
    team_a = parts[0].strip().title()
    team_b = parts[1].strip().title()

    h2h_data = head_to_head_tool.invoke(f"{team_a} vs {team_b}")
    prefs    = load_prefs()
    chain    = PREDICTION_PROMPT | llm
    response = chain.invoke({
        "team_a":          team_a,
        "team_b":          team_b,
        "h2h_data":        h2h_data,
        "language":        prefs["language"],
        "detail_level":    prefs["detail_level"],
        "preferred_format":prefs["preferred_format"],
        "favorite_team":   prefs["favorite_team"],
    })

    st.session_state["show_chart"] = True
    return response.content


@tool
def wc2026_tool(query: str) -> str:
    """
    Retrieves information about the 2026 FIFA World Cup.
    Use for questions about qualified teams, host nations,
    host cities, or any 2026 tournament details.
    Input: question about the 2026 World Cup.
    Returns: relevant 2026 tournament information.
    """
    vs      = get_vs("wc_2026")
    results = vs.similarity_search_with_score(query, k=6)
    if not results:
        return "2026 World Cup hosted by USA, Canada, Mexico with 48 teams."
    filtered = [(doc, score) for doc, score in results if score < 1.2]
    if not filtered:
        return "Limited 2026 data. Tournament hosted by USA, Canada, Mexico with 48 teams."
    return "\n\n".join(
        f"[relevance={round(1-score,3)}] {doc.page_content}"
        for doc, score in filtered
    )


@tool
def set_preference_tool(query: str) -> str:
    """
    Saves a user preference that persists in this session.
    Input format: key=value
    Valid keys: favorite_team, language, detail_level, preferred_format
    Example: favorite_team=Brazil or language=Spanish
    """
    try:
        key, value = query.split("=", 1)
        key   = key.strip().lower()
        value = value.strip()
        valid = ["favorite_team","language","detail_level","preferred_format"]
        if key not in valid:
            return f"Invalid key. Valid: {', '.join(valid)}"
        save_pref(key, value)
        return f"✅ Saved: {key} = {value}"
    except ValueError:
        return "Format: key=value e.g. favorite_team=Brazil"


@tool
def get_preferences_tool(query: str) -> str:
    """Returns all current user preferences."""
    prefs = load_prefs()
    return "\n".join(f"  {k}: {v}" for k, v in prefs.items())


ALL_TOOLS = [
    match_retrieval_tool, team_stats_tool,
    head_to_head_tool,    match_prediction_tool,
    wc2026_tool,          set_preference_tool,
    get_preferences_tool,
]

# ── Agent ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert FIFA World Cup analyst chatbot.

Tool routing rules:
- History questions   → match_retrieval_tool
- Stats questions     → team_stats_tool
- Prediction requests → head_to_head_tool THEN match_prediction_tool
- 2026 questions      → wc2026_tool
- Preferences         → set_preference_tool or get_preferences_tool

Be concise, factual, and engaging. If data is missing say so clearly."""

@st.cache_resource
def build_agent(_llm, _tools):
    memory = MemorySaver()
    return create_react_agent(
        model=_llm,
        tools=_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

agent = build_agent(llm, ALL_TOOLS)

def chat(user_input: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    return result["messages"][-1].content

# ── H2H chart ─────────────────────────────────────────────
def render_h2h_chart():
    c = st.session_state.get("h2h_cache", {})
    if not c:
        return
    team_a = c.get("team_a","Team A")
    team_b = c.get("team_b","Team B")
    fig = go.Figure(go.Bar(
        x=[f"{team_a} Wins", "Draws", f"{team_b} Wins"],
        y=[c.get("a_wins",0), c.get("draws",0), c.get("b_wins",0)],
        marker_color=["#3b82f6","#94a3b8","#ef4444"],
        text=[c.get("a_wins",0), c.get("draws",0), c.get("b_wins",0)],
        textposition="auto",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        title=f"⚽ {team_a} vs {team_b} — H2H Record ({c.get('total',0)} meetings)",
        template="plotly_dark", height=350,
        showlegend=False, yaxis_title="Matches",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════

# Session state defaults
for key, default in [
    ("messages",   []),
    ("thread_id",  "session_1"),
    ("show_chart", False),
    ("h2h_cache",  {}),
    ("prefs", {
        "favorite_team":"","language":"English",
        "detail_level":"medium","preferred_format":"paragraph"
    }),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────
st.title("🏆 FIFA World Cup Analyst Chatbot")
st.caption("Powered by LangChain · LangGraph · Pinecone · OpenAI | Data: 1872–2025")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Preferences")
    prefs = load_prefs()

    fav    = st.text_input("⭐ Favorite Team",
                            value=prefs.get("favorite_team",""),
                            placeholder="e.g. Brazil")
    lang   = st.selectbox("🌐 Language",
                           ["English","Spanish","French","Portuguese","German"],
                           index=["English","Spanish","French","Portuguese","German"]
                           .index(prefs.get("language","English")))
    detail = st.selectbox("📊 Detail Level",
                           ["brief","medium","detailed"],
                           index=["brief","medium","detailed"]
                           .index(prefs.get("detail_level","medium")))
    fmt    = st.selectbox("📝 Format",
                           ["paragraph","bullet points"],
                           index=["paragraph","bullet points"]
                           .index(prefs.get("preferred_format","paragraph")))

    if st.button("💾 Save Preferences", use_container_width=True):
        save_pref("favorite_team",    fav)
        save_pref("language",         lang)
        save_pref("detail_level",     detail)
        save_pref("preferred_format", fmt)
        st.success("✅ Preferences saved!")

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        import time
        st.session_state.thread_id  = f"session_{int(time.time())}"
        st.session_state.messages   = []
        st.session_state.show_chart = False
        st.session_state.h2h_cache  = {}
        st.rerun()

    st.divider()
    st.markdown("""
**💡 Example questions:**
- Who won the 2014 World Cup final?
- Brazil's all-time World Cup record
- Who scored in the 2022 final?
- Predict Brazil vs Germany
- Which teams qualified for 2026?
- Who is the all-time top World Cup scorer?
- Argentina vs France head to head
""")

    st.divider()
    st.caption("Data sources: Kaggle Football Results 1872–2025 · Wikipedia 2026")

# ── Welcome message ───────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Welcome! I'm your FIFA World Cup analyst.\n\n"
            "I can help with:\n"
            "⚽ Match results and goalscorers (1930–2022)\n"
            "📊 Team statistics and records\n"
            "🔮 Match predictions with H2H charts\n"
            "🌍 2026 World Cup preview\n\n"
            "Try asking: *'Who won the 2018 World Cup?'* or "
            "*'predict Brazil vs Germany'*"
        )
    })

# ── Chat history ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show chart after prediction
if st.session_state.show_chart:
    render_h2h_chart()
    st.session_state.show_chart = False

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input("Ask about World Cup history or type: predict Brazil vs Germany"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = chat(prompt, st.session_state.thread_id)
                st.markdown(response)
                if st.session_state.show_chart:
                    render_h2h_chart()
                    st.session_state.show_chart = False
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
