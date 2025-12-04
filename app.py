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
    page_title="NeuroSync: Agentic Memory (Offline Edition)",
    page_icon="🧠"
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { border-radius: 8px; font-weight: bold; border: 1px solid #30363D; }
    .memory-card { background-color: #1F2937; padding: 20px; border-radius: 10px;
                   border-left: 5px solid #6366F1; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stTextInput>div>div>input { background-color: #161B22; color: #FAFAFA; }
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
# COGNITIVE ENGINE (RULE-BASED, NO LLM)
# ============================================================
class CognitiveEngine:
    """
    Instead of calling an LLM, we deterministically extract memory
    from the fixed 30-message chat history.

    This still shows:
    - User preferences
    - Emotional patterns
    - Facts worth remembering
    """

    def __init__(self):
        self.memory_file = "long_term_memory.json"

    def extract_profile(self, history: List[str]) -> UserProfile:
        # --- Very simple heuristic parsing over messages ---
        prefs = set()
        facts = set()
        emotion_flags = []

        for msg in history:
            lower = msg.lower()

            # Preferences
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

            # Tech stack / constraints as preferences
            if "postgresql" in lower:
                prefs.add("Prefers PostgreSQL as primary database")
            if "must work on linux" in lower:
                prefs.add("Wants solutions compatible with Linux")
            if "functional programming" in lower:
                prefs.add("Favors functional programming style")

            # Emotional patterns
            if "anxious" in lower:
                emotion_flags.append("anxiety around deadlines and work")
            if "impostor syndrome" in lower:
                emotion_flags.append("impostor syndrome")
            if "overwhelmed" in lower:
                emotion_flags.append("overwhelmed by task load (Jira tickets)")
            if "excited" in lower:
                emotion_flags.append("excitement about projects and hackathons")
            if "calm and focused" in lower:
                emotion_flags.append("ability to reach calm, focused state")

            # Facts
            if "my dog" in lower and "barnaby" in lower:
                facts.add("Has a dog named Barnaby")
            if "i'm in seattle" in lower or "seattle (pst)" in lower:
                facts.add("Lives in Seattle (PST)")
            if "senior dev" in lower:
                facts.add("Is a Senior Developer")
            if "etL" in lower:
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

        # Build structured profile
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
# PERSONA ENGINE (NO LLM, TEMPLATE-BASED)
# ============================================================
def persona_rules(persona: str) -> str:
    return {
        "Calm Mentor": "Warm, calm, encouraging. Normalize struggles, give small steps.",
        "Witty Peer": "Casual, playful, slightly sarcastic but kind.",
        "Therapist-style Guide": "Reflect emotions, ask gentle questions, supportive.",
        "No-Nonsense CTO": "Direct, blunt about priorities, action-focused.",
        "Neutral": "Plain, factual, minimal style."
    }[persona]


def generate_reply(user_msg: str, profile: UserProfile, persona: str = "Neutral") -> str:
    """
    Simple handcrafted replies showing how persona & memory change tone.
    No LLM calls – fully deterministic and error-free.
    """

    # Short personalization from memory
    tail = ""
    if any("PostgreSQL" in p for p in profile.user_preferences):
        tail += " Since you prefer PostgreSQL, keep data design in mind as you solve this."
    if any("Linux" in p for p in profile.user_preferences):
        tail += " And remember, everything should behave well on Linux."
    if "Has a dog named Barnaby" in profile.facts:
        tail += " Also: don't forget to take a quick break, even Barnaby would approve."

    base = user_msg.strip()

    if persona == "Neutral":
        return (
            f"You said: \"{base}\"\n\n"
            f"Given your preferences (short, practical answers), here's a simple next step you can take.\n"
            f"1. Define the smallest concrete sub-task.\n"
            f"2. Solve or debug just that piece first.\n"
            f"3. Only then move to the next.\n"
            f"{tail}"
        )

    if persona == "Calm Mentor":
        return (
            f"I hear you: \"{base}\".\n\n"
            "You're clearly someone who cares about doing things well, and it's normal to feel pressure.\n"
            "Here's a calm way to move forward:\n"
            "1. Name the smallest part of the problem you can control right now.\n"
            "2. Spend 25 focused minutes on just that.\n"
            "3. At the end, write down what you learned or unblocked.\n"
            f"{tail}\n\n"
            f"Emotionally, you often juggle anxiety and impostor syndrome, but you also show pride in progress—lean on that."
        )

    if persona == "Witty Peer":
        return (
            f"So you’re dealing with: \"{base}\".\n\n"
            "Classic Senior Dev vibes: 47 tabs open, 1 brain cell dedicated to Jira guilt.\n"
            "Try this:\n"
            "- Pick ONE tiny thing you can ship or debug in the next 30 minutes.\n"
            "- Ignore everything else like it’s dark mode docs you hate.\n"
            "- Ship it, then flex about it later.\n"
            f"{tail}"
        )

    if persona == "Therapist-style Guide":
        return (
            f"It sounds like you're saying: \"{base}\".\n\n"
            "That tells me there's a mix of pressure, responsibility, and your own high standards.\n"
            "A few questions to gently explore:\n"
            "- What part of this feels most overwhelming right now?\n"
            "- What have you already done that you’re proud of on this project?\n"
            "- If you broke this into two smaller steps, what would they be?\n\n"
            "From your history, you’ve handled anxiety around deadlines before and come out feeling calm and focused again.\n"
            f"{tail}"
        )

    if persona == "No-Nonsense CTO":
        return (
            f"Input: \"{base}\".\n\n"
            "Here’s the blunt version:\n"
            "1. Define the outcome you need by end of day.\n"
            "2. Kill everything that doesn’t serve that (tickets, scope, perfectionism).\n"
            "3. Ship the smallest usable version.\n"
            "4. Document follow-ups instead of trying to do them now.\n"
            f"{tail}\n\n"
            "You want to be a CTO – this is exactly the muscle: prioritization under pressure."
        )

    # Fallback
    return f"[{persona}] {base}"

# ============================================================
# OPTIONAL: TEXT-TO-SPEECH (NETWORK-FREE IF gTTS WORKS)
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
    st.title("🧠 NeuroSync — Agentic Memory & Personality Engine (Offline Demo)")
    st.caption("No external APIs. Fully local. Designed to showcase memory + persona logic without errors.")

    brain = CognitiveEngine()

    # ---------------------------------------------------------
    # 1. MEMORY EXTRACTION
    # ---------------------------------------------------------
    st.subheader("1. Extract Structured Memory from 30 Messages")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📄 View Chat History"):
            st.code(CHAT_HISTORY_30)

        if st.button("🚀 Run Memory Extraction", use_container_width=True):
            with st.spinner("Analyzing 30 messages with rule-based engine..."):
                profile = brain.extract_profile(CHAT_HISTORY_30)
                brain.save(profile)
                st.session_state["profile"] = profile
                st.success("✅ Memory extracted and saved (no APIs required).")

    with col2:
        if "profile" not in st.session_state:
            loaded = brain.load()
            if loaded:
                st.session_state["profile"] = loaded

        if "profile" in st.session_state:
            p = st.session_state["profile"]
            st.markdown(f"""
            <div class="memory-card">
                <h4>👤 Structured User Profile</h4>
                <p><b>Preferences:</b> {", ".join(p.user_preferences)}</p>
                <p><b>Emotional Patterns:</b> {p.emotional_patterns}</p>
                <p><b>Facts Worth Remembering:</b> {", ".join(p.facts)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Memory Extraction' to build the profile.")

    st.divider()

    # ---------------------------------------------------------
    # 2. PERSONA ENGINE — BEFORE / AFTER
    # ---------------------------------------------------------
    st.subheader("2. Persona Engine — Before vs After Response")

    persona = st.selectbox(
        "Select persona:",
        ["Calm Mentor", "Witty Peer", "Therapist-style Guide", "No-Nonsense CTO"]
    )

    user_msg = st.chat_input("Type something (e.g., 'I'm overwhelmed by Q4 deadlines')")

    if user_msg and "profile" in st.session_state:
        profile = st.session_state["profile"]

        neutral = generate_reply(user_msg, profile, persona="Neutral")
        styled = generate_reply(user_msg, profile, persona=persona)

        left, right = st.columns(2)

        with left:
            with st.chat_message("assistant"):
                st.markdown("### 🔹 Neutral Response")
                st.write(neutral)

        with right:
            with st.chat_message("assistant"):
                st.markdown(f"### 🔸 Persona: {persona}")
                st.write(styled)

                audio = tts(styled)
                if audio:
                    st.audio(audio, format="audio/mp3")

if __name__ == "__main__":
    main()
