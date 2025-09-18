# FetiiAI Chatbot
import os
import re
import pandas as pd
import streamlit as st
from typing import TypedDict, List, Union, Optional, Tuple, Literal, Dict
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

CENTRAL_TZ = ZoneInfo("America/Chicago")

# -------------------------------
# Streamlit + App Setup
# -------------------------------
st.set_page_config(page_title="Fetii AI Chatbot", page_icon="logo_image.png")

# -------------------------------
# Load OpenAI API key from Streamlit Secrets
# -------------------------------
api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------------
# Model --> OpenAI GPT-5
# -------------------------------
llm = ChatOpenAI(model="gpt-5", api_key=api_key, temperature=1)

# -------------------------------
# Load Data (Excel sheets) --> FetiiAI_Data_Austin.xlsx
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    trip_df = pd.read_excel("FetiiAI_Data_Austin.xlsx", sheet_name="Trip Data")
    rider_df = pd.read_excel("FetiiAI_Data_Austin.xlsx", sheet_name="Checked in User ID's")
    demo_df = pd.read_excel("FetiiAI_Data_Austin.xlsx", sheet_name="Customer Demographics")

    # Basic preprocessing used later
    trip_df["Trip Date and Time"] = pd.to_datetime(trip_df["Trip Date and Time"], errors="coerce")
    trip_df["year_month"] = trip_df["Trip Date and Time"].dt.to_period("M")
    trip_df["weekday"] = trip_df["Trip Date and Time"].dt.day_name().str.lower()
    return {"trip": trip_df, "rider": rider_df, "demo": demo_df}

DATA = load_data()
trip_df: pd.DataFrame = DATA["trip"]
rider_df: pd.DataFrame = DATA["rider"]
demo_df: pd.DataFrame = DATA["demo"]

# -------------------------------
# UI: Chat bubbles for conversation
# -------------------------------
def render_chat_message(message: str, sender: str = "assistant"):
    """Colored bubble wrapper; message itself renders as Markdown (no HTML injection)."""
    if sender == "user":
        bg_color = "#5A2D82"   # purple --> user text bubble
        text_color = "white"
        align = "flex-end"
    else:
        bg_color = "#E5E5EA"   # light gray --> response text bubble
        text_color = "black"
        align = "flex-start"

    safe_message = message.replace("</div>", "&lt;/div&gt;")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align};">
          <div style="
              background-color:{bg_color};
              color:{text_color};
              padding:10px 12px;
              border-radius:12px;
              margin:6px 0;
              max-width: 90%;
              white-space: pre-wrap;
              overflow-wrap: anywhere;
          ">{safe_message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Date-range resolver
# -------------------------------
def resolve_date_range(dr: Optional[Dict[str, Optional[str]]]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Supports 'last month', 'this month', 'last 7 days', or YYYY-MM-DD. Default: last full month."""
    def last_full_month():
        today = pd.Timestamp.now().normalize()
        first_this = today.replace(day=1)
        start = (first_this - pd.DateOffset(months=1)).normalize()
        end = first_this - pd.Timedelta(days=1)
        return start, end

    if not dr or (dr.get("start") is None and dr.get("end") is None):
        return last_full_month()

    kw = (dr.get("start") or "").strip().lower()
    if kw in ["last month", "previous month"]:
        return last_full_month()
    if kw in ["this month", "current month"]:
        today = pd.Timestamp.now().normalize()
        return today.replace(day=1), today
    if kw in ["last 7 days", "past 7 days"]:
        end = pd.Timestamp.now().normalize()
        return end - pd.Timedelta(days=7), end

    def parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
        if not s:
            return None
        try:
            return pd.to_datetime(s, errors="raise").normalize()
        except Exception:
            return None

    start = parse_date(dr.get("start"))
    end = parse_date(dr.get("end"))
    if start is None and end is None:
        return last_full_month()
    if start is not None and end is None:
        return start, start
    if start is None and end is not None:
        return end, end
    return start, end

# -------------------------------
# Structured Intent + Slots
# -------------------------------
from typing import Optional as Opt, Literal as Lit

class DateRange(BaseModel):
    start: Opt[str] = Field(None, description="e.g., 'last month' or '2025-08-01'")
    end: Opt[str] = Field(None, description="Optional end date if explicit")

class IntentParse(BaseModel):
    intent: Lit[
        "COUNT_TRIPS_TO_LOCATION",
        "TOP_DROPOFFS_BY_AGE_AND_DAY",
        "GROUPS_TREND",
        "ABOUT_FETII",
        "HELP_FAQ",
        "UNKNOWN"
    ] = "UNKNOWN"
    location: Opt[str] = None
    age_low: Opt[int] = None
    age_high: Opt[int] = None
    day_of_week: Opt[str] = None
    group_size_min: Opt[int] = None
    group_size_max: Opt[int] = None
    date_range: Opt[DateRange] = None
    topic: Opt[str] = None

def parse_intent(user_text: str) -> IntentParse:
    sys = SystemMessage(content=(
        "Extract a structured intent and slots from a question about rideshare data and the Fetii app.\n"
        "Choose ONE intent from:\n"
        "- COUNT_TRIPS_TO_LOCATION (e.g., 'How many groups went to X last month?')\n"
        "- TOP_DROPOFFS_BY_AGE_AND_DAY (e.g., 'Top drop-offs for ages 18–24 on Saturdays?')\n"
        "- GROUPS_TREND (when do groups ride) — if they say 'large groups', set group_size_min=6; "
        "  if they say 'small groups', set group_size_max=5; if they say 'less than 6', set group_size_max=5; "
        "  if they say 'at least N' or 'N+ / ≥N', set group_size_min=N.\n"
        "- ABOUT_FETII (for general company/app questions like 'What is Fetii?', booking, pricing, vehicles, support)\n"
        "- HELP_FAQ (for 'what can you do?', 'help')\n"
        "- UNKNOWN (last resort)\n"
        "If intent=ABOUT_FETII, also set `topic` to one of: "
        "['what_is','how_to_book','why_use','pricing','vehicles','support'] based on the user's wording.\n"
        "Fill any slots you can find, including location if stated."
    ))
    hm = HumanMessage(content=user_text)
    try:
        return llm.with_structured_output(IntentParse).invoke([sys, hm])
    except Exception:
        return IntentParse(intent="UNKNOWN")

# -------------------------------
# Simple text heuristics (size + location) for robustness & follow-ups
# -------------------------------
SIZE_WORDS_SMALL = re.compile(r"\b(small group|small groups)\b", re.I)
SIZE_UNDER_6 = re.compile(r"\b(less than|under)\s*6\b|<\s*6|\b<=\s*5\b", re.I)
SIZE_RANGE = re.compile(r"\b(\d+)\s*(?:to|-|–|—)\s*(\d+)\b")
SIZE_AT_LEAST = re.compile(r"\b(at least|>=|≥)\s*(\d+)\b", re.I)
SIZE_N_PLUS = re.compile(r"\b(\d+)\s*\+\b", re.I)
SIZE_LESS_EQ = re.compile(r"\b<=\s*(\d+)\b", re.I)

def extract_size_bounds(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (min,max) inferred from plain English and symbols."""
    t = text.lower()

    m = SIZE_RANGE.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))

    if SIZE_WORDS_SMALL.search(t) or SIZE_UNDER_6.search(t):
        return (None, 5)

    m = SIZE_AT_LEAST.search(t)
    if m:
        return (int(m.group(2)), None)

    m = SIZE_N_PLUS.search(t)
    if m:
        return (int(m.group(1)), None)

    m = SIZE_LESS_EQ.search(t)
    if m:
        return (None, int(m.group(1)))

    if "large group" in t or "large groups" in t:
        return (6, None)

    return (None, None)

def extract_location(text: str) -> Optional[str]:
    t = text.lower()
    if "downtown" in t:
        return "downtown"
    if "moody center" in t:
        return "Moody Center"
    return None

# -------------------------------
# Central Time helpers
# -------------------------------
def _to_central(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return s
    try:
        if s.dt.tz is not None:
            return s.dt.tz_convert(CENTRAL_TZ)
        else:
            return s.dt.tz_localize(CENTRAL_TZ)
    except Exception:
        return s.dt.tz_localize(CENTRAL_TZ)

def _fmt_hour_ampm(h: int) -> str:
    return pd.Timestamp(2000, 1, 1, int(h), tz=CENTRAL_TZ).strftime("%I %p").lstrip("0")

def _trip_word(n: int) -> str:
    return "trip" if n == 1 else "trips"

# -------------------------------
# Mini FAQ for general Fetii questions
# -------------------------------
FETII_FAQ = {
    "what is fetii": (
        "Fetii is a group rideshare platform that moves people together efficiently—"
        "popular for nights out, events, and corporate transport."
    ),
    "how to book": (
        "Open the Fetii iOS/Android app → enter pickup & drop-off, group size, and time → "
        "choose a vehicle → confirm. You’ll get driver details and live status."
    ),
    "vehicles": "Choose vehicles sized for your group; availability varies by time and demand.",
    "pricing": "Pricing depends on time, demand, distance, and vehicle size.",
    "support": "Use in-app Help to reach Fetii Support. (In this demo, the sidebar button is a placeholder.)",
    "why use": "Move your entire group together, fewer split rides, simpler coordination, often cheaper per person, and more fun."
}

def faq_lookup(user_text: str) -> Optional[str]:
    t = user_text.lower()
    if "what is fetii" in t:
        return FETII_FAQ["what is fetii"]
    if any(k in t for k in ["how do i book", "how to book", "book in app", "booking", "how to book a ride"]):
        return FETII_FAQ["how to book"]
    if "vehicle" in t:
        return FETII_FAQ["vehicles"]
    if "price" in t or "cost" in t or "pricing" in t:
        return FETII_FAQ["pricing"]
    if "support" in t or "help" in t or "contact" in t:
        return FETII_FAQ["support"]
    if "why use" in t or "why fetii" in t or "benefit" in t or "value" in t:
        return FETII_FAQ["why use"]
    return None

# -------------------------------
# Handlers
# -------------------------------
def handle_count_trips_to_location(location: Optional[str], dr_obj: Optional[DateRange]) -> str:
    start, end = resolve_date_range(dr_obj.model_dump() if dr_obj else None)
    loc = (location or "").strip()
    mask = (
        trip_df["Trip Date and Time"].between(start, end)
        & trip_df["Drop Off Address"].fillna("").str.contains(loc, case=False, na=False)
    )
    count = int(mask.sum())
    if count == 0:
        hits = trip_df[trip_df["Drop Off Address"].fillna("").str.contains(loc, case=False, na=False)]
        if hits.empty:
            return f"No trips found for '{loc}' in the dataset."
        mr = hits["year_month"].max()
        alt = hits[hits["year_month"] == mr]
        return f"In {mr.strftime('%B %Y')}, {len(alt)} groups went to {loc}."
    return f"Between {start.date()} and {end.date()}, **{count}** groups went to **{loc or '(unspecified)'}**."

def handle_top_dropoffs_by_age_and_day(age_low: Optional[int],
                                       age_high: Optional[int],
                                       day_of_week: Optional[str],
                                       dr_obj: Optional[DateRange]) -> str:
    start, end = resolve_date_range(dr_obj.model_dump() if dr_obj else None)
    day = (day_of_week or "").strip().lower()

    merged = trip_df.merge(rider_df, on="Trip ID").merge(demo_df, on="User ID")

    mask = merged["Trip Date and Time"].between(start, end)
    if day:
        mask &= merged["weekday"].eq(day)
    if age_low is not None:
        mask &= merged["Age"].fillna(0) >= age_low
    if age_high is not None:
        mask &= merged["Age"].fillna(0) <= age_high

    counts = merged.loc[mask, "Drop Off Address"].value_counts().head(5)

    if not counts.empty:
        lines = "\n".join(f"- {k}: {v}" for k, v in counts.items())
        return (f"Top drop-offs for ages **{age_low}–{age_high}** on **{day_of_week or '(any day)'}** "
                f"({start.date()}–{end.date()}):\n{lines}")

    fallback_mask = pd.Series(True, index=merged.index)
    if day:
        fallback_mask &= merged["weekday"].eq(day)
    if age_low is not None:
        fallback_mask &= merged["Age"].fillna(0) >= age_low
    if age_high is not None:
        fallback_mask &= merged["Age"].fillna(0) <= age_high

    candidates = merged.loc[fallback_mask].copy()
    if candidates.empty:
        return (f"No drop-offs found for ages {age_low}–{age_high} "
                f"on {day_of_week or '(any day)'} in the dataset.")

    candidates["year_month"] = candidates["Trip Date and Time"].dt.to_period("M")
    mr = candidates["year_month"].max()
    monthly = candidates[candidates["year_month"] == mr]
    counts2 = monthly["Drop Off Address"].value_counts().head(5)
    if counts2.empty:
        return (f"No drop-offs found for ages {age_low}–{age_high} "
                f"on {day_of_week or '(any day)'} in the dataset.")

    lines = "\n".join(f"- {k}: {v}" for k, v in counts2.items())
    return (f"In {mr.strftime('%B %Y')}, top drop-offs for ages "
            f"{age_low}–{age_high} on {day_of_week or '(any day)'} were:\n{lines}")

def _label_for_size_range(size_min: Optional[int], size_max: Optional[int]) -> str:
    if size_min is not None and size_max is not None:
        return f"{size_min}–{size_max} riders"
    if size_min is not None:
        return f"{size_min}+ riders"
    if size_max is not None:
        return f"≤{size_max} riders"
    return "all group sizes"

def handle_groups_trend(location: Optional[str],
                        size_min: Optional[int],
                        size_max: Optional[int],
                        dr_obj: Optional[DateRange]) -> str:
    """Generic: supports small (e.g., ≤5) or large (e.g., ≥6). Returns hours in Central Time."""
    start, end = resolve_date_range(dr_obj.model_dump() if dr_obj else None)
    loc = (location or "").strip()

    mask = trip_df["Trip Date and Time"].between(start, end)
    if size_min is not None:
        mask &= trip_df["Total Passengers"].fillna(0).astype(int).ge(size_min)
    if size_max is not None:
        mask &= trip_df["Total Passengers"].fillna(0).astype(int).le(size_max)
    if loc:
        mask &= trip_df["Drop Off Address"].fillna("").str.contains(loc, case=False, na=False)

    ts_ct = _to_central(trip_df.loc[mask, "Trip Date and Time"])
    counts = ts_ct.dt.hour.value_counts().sort_index()

    label = _label_for_size_range(size_min, size_max)
    target_loc = loc or "(any location)"
    if not counts.empty:
        lines = "\n".join(f"- {_fmt_hour_ampm(int(h))} → {int(c)} {_trip_word(int(c))}" for h, c in counts.items())
        return (f"Groups ({label}) going to **{target_loc}** by hour "
                f"({start.date()}–{end.date()}), Central Time:\n{lines}\n\n_All times Central Time (CT)._")

    # fallback: most recent month with any matches for the size/location filters
    fallback_mask = pd.Series(True, index=trip_df.index)
    if size_min is not None:
        fallback_mask &= trip_df["Total Passengers"].fillna(0).astype(int).ge(size_min)
    if size_max is not None:
        fallback_mask &= trip_df["Total Passengers"].fillna(0).astype(int).le(size_max)
    if loc:
        fallback_mask &= trip_df["Drop Off Address"].fillna("").str.contains(loc, case=False, na=False)

    candidates = trip_df.loc[fallback_mask].copy()
    if candidates.empty:
        return f"No trips for {label} to {target_loc} in the dataset."

    candidates["year_month"] = candidates["Trip Date and Time"].dt.to_period("M")
    mr = candidates["year_month"].max()
    monthly = candidates[candidates["year_month"] == mr]
    ts_ct2 = _to_central(monthly["Trip Date and Time"])
    counts2 = ts_ct2.dt.hour.value_counts().sort_index()
    if counts2.empty:
        return f"No trips for {label} to {target_loc} in the dataset."

    lines = "\n".join(f"- {_fmt_hour_ampm(int(h))} → {int(c)} {_trip_word(int(c))}" for h, c in counts2.items())
    return (f"In {mr.strftime('%B %Y')}, groups ({label}) going to {target_loc} rode at these hours (Central Time):\n"
            f"{lines}\n\n_All times Central Time (CT)._")

def handle_about_fetii(topic: Optional[str], user_text: str) -> str:
    """Return a concise answer scoped to exactly the user's topic."""
    if topic == "what_is":
        return FETII_FAQ["what is fetii"]
    if topic == "how_to_book":
        return FETII_FAQ["how to book"]
    if topic == "why_use":
        return FETII_FAQ["why use"]
    if topic == "pricing":
        return FETII_FAQ["pricing"]
    if topic == "vehicles":
        return FETII_FAQ["vehicles"]
    if topic == "support":
        return FETII_FAQ["support"]

    # Fallback: keyword lookup for robustness
    faq = faq_lookup(user_text)
    if faq:
        return faq

    # Last resort: concise menu, not multi-section
    return "Ask me about: What is Fetii? How to book a ride? Why use Fetii? Pricing, vehicles, or support."

def handle_help_faq() -> str:
    return (
        "I can answer questions about Fetii and this Austin dataset. Try:\n"
        "• “How many groups went to Moody Center last month?”\n"
        "• “Top drop-offs for ages 18–24 on Saturdays last month?”\n"
        "• “When do large (6+) groups go downtown?”\n\n"
        "You can also say “this month”, “last 7 days”, or give dates like 2025-08-01."
    )

# -------------------------------
# Router + light memory
# -------------------------------
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

if "last_location" not in st.session_state:
    st.session_state.last_location = None
if "last_size_min" not in st.session_state:
    st.session_state.last_size_min = None
if "last_size_max" not in st.session_state:
    st.session_state.last_size_max = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""   # draft set by sidebar suggestions

def process(state: AgentState) -> AgentState:
    user_text = state["messages"][-1].content
    parsed = parse_intent(user_text)

    # Heuristic extraction from raw text
    ext_loc = extract_location(user_text)
    ext_min, ext_max = extract_size_bounds(user_text)

    # Merge precedence: explicit parse > extracted > memory
    loc = parsed.location or ext_loc or st.session_state.last_location
    size_min = parsed.group_size_min if parsed.group_size_min is not None else ext_min
    size_max = parsed.group_size_max if parsed.group_size_max is not None else ext_max
    if size_min is None and size_max is None:
        size_min = st.session_state.last_size_min
        size_max = st.session_state.last_size_max

    # Route by intent
    if parsed.intent == "COUNT_TRIPS_TO_LOCATION":
        response = handle_count_trips_to_location(loc, parsed.date_range)
    elif parsed.intent == "TOP_DROPOFFS_BY_AGE_AND_DAY":
        response = handle_top_dropoffs_by_age_and_day(
            parsed.age_low, parsed.age_high, parsed.day_of_week, parsed.date_range
        )
    elif parsed.intent == "GROUPS_TREND":
        response = handle_groups_trend(loc, size_min, size_max, parsed.date_range)
    elif parsed.intent == "ABOUT_FETII":
        response = handle_about_fetii(parsed.topic, user_text)
    elif parsed.intent == "HELP_FAQ":
        response = handle_help_faq()
    else:
        low = user_text.lower()
        faq = faq_lookup(user_text)
        if faq:
            response = faq
        elif ("small group" in low or "small groups" in low) or SIZE_UNDER_6.search(low):
            response = handle_groups_trend(loc or "downtown", None, 5, None)
        elif "large group" in low or "large groups" in low:
            response = handle_groups_trend(loc or "downtown", 6, None, None)
        elif any(k in low for k in ["what is fetii", "how do i book", "book in app", "pricing", "vehicles", "support", "why use", "how to book a ride"]):
            response = handle_about_fetii(None, user_text)
        else:
            response = llm.invoke(state["messages"]).content

    # Update memory
    st.session_state.last_location = loc
    st.session_state.last_size_min = size_min
    st.session_state.last_size_max = size_max

    state["messages"].append(AIMessage(content=response))
    return state

# -------------------------------
# LangGraph wiring
# -------------------------------
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# -------------------------------
# Sidebar (actions + sample questions)
# -------------------------------
with st.sidebar:
    st.header("Actions")
    if st.button("Start new chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.last_location = None
        st.session_state.last_size_min = None
        st.session_state.last_size_max = None
        st.session_state.pending_input = ""
        st.rerun()

    if st.button("Contact Fetii/Support", use_container_width=True):
        st.info("Support coming soon. (Placeholder)")

    st.markdown("### Try a question")
    samples = [
        "How many groups went to Moody Center last month?",
        "What are the top drop-off spots for 18–24 year-olds on Saturday nights?",
        "When do large groups (6+ riders) typically ride downtown?",
        "When do small groups typically ride downtown?",
        "What is Fetii?",
        "How to book a ride?",
        "Why use Fetii?"
    ]
    for i, q in enumerate(samples, start=1):
        if st.button(q, key=f"sb_q{i}", use_container_width=True):
            st.session_state.pending_input = q  # set draft (will show above bottom input)
            st.rerun()

# -------------------------------
# Main UI (center + chat history)
# -------------------------------
center_col = st.columns([1, 2, 1])[1]
with center_col:
    st.title("Fetii AI Chatbot")
    if os.path.exists("logo_image.png"):
        st.image("logo_image.png", use_column_width=True)

# Init system message
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=(
            "You are a helpful virtual assistant for Fetii Inc., a shared mobility company focused on "
            "on-demand group rides in Austin, Texas. Answer questions using the provided dataset when relevant. "
            "If the user asks about the company or how the app works, explain it clearly and succinctly."
        ))
    ]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        render_chat_message(msg.content, sender="user")
    elif isinstance(msg, AIMessage):
        render_chat_message(msg.content, sender="assistant")

if st.session_state.pending_input:
    with st.container():
        st.info(f"Draft: {st.session_state.pending_input}")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Send draft", use_container_width=True):
                st.session_state.messages.append(HumanMessage(content=st.session_state.pending_input))
                st.session_state.pending_input = ""
                result = agent.invoke({"messages": st.session_state.messages})
                st.session_state.messages = result["messages"]
                st.rerun()
        with c2:
            if st.button("Clear draft", use_container_width=True):
                st.session_state.pending_input = ""
                st.rerun()

typed = st.chat_input("Ask me something about rideshare data or the Fetii app...")
if typed:
    st.session_state.messages.append(HumanMessage(content=typed))
    result = agent.invoke({"messages": st.session_state.messages})
    st.session_state.messages = result["messages"]
    st.rerun()
