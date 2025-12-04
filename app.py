import streamlit as st
import json
import os
from pydantic import BaseModel
from typing import List
from gtts import gTTS
import io

# ============================================================
#  UI CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="GuppShupp – Lifelong Friend",
    page_icon="💬"
)

# Stealth + friendly UI theme
st.markdown("""
<style>
    .stApp { background-color: #050814; color: #FAFAFA; }
    .stButton>button { border-radius: 999px; font-weight: 600;
                       border: 1px solid #ff4b7a; background: #111827; }
    .stButton>button:hover { border-color: #ff6b94; }
    .memory-card { background: #111827; padding: 20px; border-radius: 16px;
                   border-left: 5px solid #ff4b7a;
                   box-shadow: 0 8px 18px rgba(0,0,0,0.45); }
    .stTextInput>div>div>input,
    .stChatInputContainer textarea {
        background-color: #050814;
        color: #FAFAFA;
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 30-MESSAGE CHAT HISTORY (ASSIGNMENT DATA)
# ============================================================
CHAT_HISTORY_30 = [
    "I'm just setting up my dev environment.", "I hate cluttered UIs, keep it minimal.",
    "My dog Barnaby kept me up all night.", "I need a Python script for ETL.",
    "I prefer snake_case variables.", "I'm feeling anxious about the Q4 deadline.",
    "No long explanations, just code.", "I'm in Seattle (PST).",
    "Watched Dune again, love sci-fi.", "Anxiety is better today after a run.",
    "Refactoring this React component is a pain.", "I'm a Senior Dev, skip the basics.",
    "Cutting down on caffeine.", "How do I handle async errors in Node?",
    "Don't talk to me like a robot.", "Project 'TitanAPI' is launching soon.",
    "Proud of the progress we made.", "Remind me to call mom on Sunday.",
    "I hate dark mode in documentation.", "Impostor syndrome is hitting hard.",
    "Let's discuss the DB schema.", "I use PostgreSQL exclusively.",
    "Functional programming is the way.", "Overwhelmed by Jira tickets.",
    "Must work on Linux.", "Excited for the hackathon.",
    "No emojis in comments.", "Goal: Become a CTO.",
    "Feeling calm and focused.", "Thanks, that was short and sweet."
]

# ============================================================
# MEMORY SCHEMA
# ============================================================
class UserProfile(BaseModel):
    user_preferences: List[str]
    emotional_patterns: str
    facts: List[str]

# ============================================================
# COGNITIVE ENGINE (RULE-BASED, NO LLM / NO API)
# ============================================================
class CognitiveEngine:
    """
    Deterministic memory extraction from the fixed 30-message history.
    This simulates a 'memory module' without relying on external APIs.
    """

    def __init__(self):
        self.memory_file = "long_term_memory.json"

    def extract_profile(self, history: List[str]) -> UserProfile:
        prefs = set()
        facts = set()
        emotion_flags = []

        for msg in history:
            lower = msg.lower()

            # --- Preferences ---
            if "hate cluttered uis" in lower or "keep it minimal" in lower:
                prefs.add("Prefers minimal, uncluttered UIs")
            if "prefer snake_case" in lower:
                prefs.add("Prefers snake_case for variables")
            if "no long explanations" in lower:
                prefs.add("Wants concise answers (no long explanations)")
            if "skip the basics" in lower:
                prefs.add("Senior-level explanations only (skip basics)")
            if "no emojis in comments" in lower:
                prefs.add("No emojis in code comments")
            if "hate dark mode" in lower:
                prefs.add("Dislikes dark mode in documentation")

            if "postgresql" in lower:
                prefs.add("Prefers PostgreSQL as primary database")
            if "must work on linux" in lower:
                prefs.add("Wants solutions compatible with Linux")
            if "functional programming" in lower:
                prefs.add("Favors functional programming style")

            # --- Emotional patterns ---
            if "anxious" in lower:
                emotion_flags.append("anxiety around deadlines and work")
            if "impostor syndrome" in lower:
                emotion_flags.append("impostor syndrome")
            if "overwhelmed" in lower:
                emotion_flags.append("overwhelmed by task load (Jira tickets)")
            if "excited" in lower:
                emotion_flags.append("excitement about projects and hackathons")
            if "calm and focused" in lower:
                emotion_flags.append("ability to reach a calm, focused state")

            # --- Facts worth remembering ---
            if "my dog" in lower and "barnaby" in lower:
                facts.add("Has a dog named Barnaby")
            if "i'm in seattle" in lower or "seattle (pst)" in lower:
                facts.add("Lives in Seattle (PST)")
            if "senior dev" in lower:
                facts.add("Is a Senior Developer")
            if "need a python script for etl" in lower or "etl" in lower:
                facts.add("Works with Python ETL scripts")
            if "project 'titanapi'" in lower:
                facts.add("Working on a project called 'TitanAPI'")
            if "remind me to call mom on sunday" in lower:
                facts.add("Values family (reminder to call mom on Sundays)")
            if "goal: become a cto" in lower:
                facts.add("Long-term goal: Become a CTO")
            if "dune" in lower and "sci-fi" in lower:
                facts.add("Enjoys sci-fi, especially the movie Dune")
            if "cutting down on caffeine" in lower:
                facts.add("Trying to cut down on caffeine")

        if not emotion_flags:
            emotional_patterns = "Emotionally stable; no strong recurring negative patterns detected."
        else:
            emotional_patterns = (
                "Frequently experiences "
                + ", ".join(sorted(set(emotion_flags)))
                + ", but remains motivated and proud of progress."
            )

        profile = UserProfile(
            user_preferences=sorted(prefs),
            emotional_patterns=emotional_patterns,
            facts=sorted(facts),
        )
        return profile

    def save(self, profile: UserProfile):
        with open(self.memory_file, "w") as f:
            f.write(profile.model_dump_json(indent=2))

    def load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                return UserProfile(**data)
        return None

# ============================================================
# PERSONA ENGINE (TEMPLATE-BASED, NO LLM)
# ============================================================
def persona_rules(persona: str) -> str:
    return {
        "Calm Mentor": "Warm, calm, encouraging. Normalize struggles, give small steps.",
        "Witty Friend": "Casual, playful, a bit cheeky but caring.",
        "Therapist-style Guide": "Reflect emotions, ask gentle questions, supportive.",
        "No-Nonsense CTO": "Direct, blunt about priorities, action-focused.",
        "Neutral": "Plain, factual, minimal style."
    }[persona]


def generate_reply(user_msg: str, profile: UserProfile, persona: str = "Neutral") -> str:
    """
    Simple handcrafted replies showing how GuppShupp’s persona & memory
    change tone. No LLM calls – fully deterministic.
    """

    tail = ""
    if any("PostgreSQL" in p for p in profile.user_preferences):
        tail += " Since you like PostgreSQL, keep your data design tight while you work through this."
    if any("Linux" in p for p in profile.user_preferences):
        tail += " Also, remember everything should behave well on Linux."
    if "Has a dog named Barnaby" in profile.facts:
        tail += " And hey, maybe a quick break with Barnaby wouldn’t hurt."

    msg = user_msg.strip()

    if persona == "Neutral":
        return (
            f"You said: \"{msg}\"\n\n"
            "Here’s a simple way forward:\n"
            "1. Pick the smallest concrete sub-task.\n"
            "2. Focus only on that until it’s done.\n"
            "3. Then move to the next piece.\n"
            f"{tail}"
        )

    if persona == "Calm Mentor":
        return (
            f"GuppShupp hears you: \"{msg}\".\n\n"
            "It makes sense to feel this way, especially with everything on your plate.\n"
            "Let’s keep it gentle:\n"
            "1. Name one tiny part of this that you can control right now.\n"
            "2. Give it 20–25 focused minutes.\n"
            "3. At the end, write down one win, no matter how small.\n"
            f"{tail}\n\n"
            "From your patterns, you juggle anxiety and impostor feelings, but you also reach calm and feel proud of progress. Lean on that."
        )

    if persona == "Witty Friend":
        return (
            f"Okay, so you’re dealing with: \"{msg}\".\n\n"
            "Classic overworked, under-caffeinated dev energy.\n"
            "Here’s the GuppShupp hack:\n"
            "- Pick ONE tiny thing you can fix or ship in the next 30 minutes.\n"
            "- Ignore everything else like it’s dark-mode docs you hate.\n"
            "- Ship it. Tiny wins beat big stress.\n"
            f"{tail}"
        )

    if persona == "Therapist-style Guide":
        return (
            f"It sounds like you’re saying: \"{msg}\".\n\n"
            "That tells me there’s a mix of pressure, responsibility, and your own high standards.\n"
            "A few gentle questions:\n"
            "- What part of this feels heaviest right now?\n"
            "- What have you already done that you’re quietly proud of?\n"
            "- If you broke this into two smaller steps, what would they be?\n\n"
            "Your history shows you’ve handled anxiety before and found your calm again.\n"
            f"{tail}"
        )

    if persona == "No-Nonsense CTO":
        return (
            f"Input: \"{msg}\".\n\n"
            "Here’s the ruthless version:\n"
            "1. Decide what absolutely must be true by end of day.\n"
            "2. Kill everything that doesn’t serve that (tickets, polish, ego).\n"
            "3. Ship the smallest usable slice.\n"
            "4. Log follow-ups instead of doing them now.\n"
            f"{tail}\n\n"
            "You want to be a CTO. This is the muscle: prioritizing under pressure, not doing everything."
        )

    return f"[{persona}] {msg}"

# ============================================================
# TEXT-TO-SPEECH (OPTIONAL)
# ============================================================
def tts(text: str):
    try:
        obj = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        obj.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header with logo + brand
    header_col1, header_col2 = st.columns([1, 4])

    with header_col1:
        # Put your logo file as "guppshupp_logo.png" in the same folder as app.py
        try:
            st.image("guppshupp_logo.png", width=72)
        except Exception:
            st.markdown("💬")

    with header_col2:
        st.markdown(
            "<h1 style='margin-bottom:0;'>GuppShupp – Lifelong Friend</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#d1d5db;'>A chat companion that remembers you, "
            "adapts its personality, and keeps things short, real, and helpful.</p>",
            unsafe_allow_html=True,
        )

    brain = CognitiveEngine()

    # ---------------------------------------------------------
    # 1. MEMORY EXTRACTION
    # ---------------------------------------------------------
    st.subheader("1. How GuppShupp Remembers You")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📄 See your 30-message story"):
            st.code(CHAT_HISTORY_30)

        if st.button("🧠 Build Memory Profile", use_container_width=True):
            with st.spinner("GuppShupp is quietly connecting the dots..."):
                profile = brain.extract_profile(CHAT_HISTORY_30)
                brain.save(profile)
                st.session_state["profile"] = profile
                st.success("✅ Memory profile created (preferences, emotions, and key facts).")

    with col2:
        if "profile" not in st.session_state:
            loaded = brain.load()
            if loaded:
                st.session_state["profile"] = loaded

        if "profile" in st.session_state:
            p = st.session_state["profile"]
            st.markdown(f"""
            <div class="memory-card">
                <h4>👤 What GuppShupp Learns About You</h4>
                <p><b>Preferences:</b><br>{", ".join(p.user_preferences) or "Not enough data yet."}</p>
                <p><b>Emotional Patterns:</b><br>{p.emotional_patterns}</p>
                <p><b>Facts Worth Remembering:</b><br>{", ".join(p.facts) or "None extracted yet."}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click **Build Memory Profile** to let GuppShupp learn from the 30 messages.")

    st.divider()

    # ---------------------------------------------------------
    # 2. PERSONA ENGINE — BEFORE / AFTER
    # ---------------------------------------------------------
    st.subheader("2. GuppShupp’s Personality Modes")

    persona = st.selectbox(
        "How should GuppShupp talk right now?",
        ["Calm Mentor", "Witty Friend", "Therapist-style Guide", "No-Nonsense CTO"]
    )

    user_msg = st.chat_input("Tell GuppShupp what’s on your mind...")

    if user_msg and "profile" in st.session_state:
        profile = st.session_state["profile"]

        neutral = generate_reply(user_msg, profile, persona="Neutral")
        styled = generate_reply(user_msg, profile, persona=persona)

        left, right = st.columns(2)

        with left:
            with st.chat_message("assistant"):
                st.markdown("### 🔹 Neutral GuppShupp")
                st.write(neutral)

        with right:
            with st.chat_message("assistant"):
                st.markdown(f"### 🔸 {persona} GuppShupp")
                st.write(styled)

                audio = tts(styled)
                if audio:
                    st.audio(audio, format="audio/mp3")


if __name__ == "__main__":
    main()
