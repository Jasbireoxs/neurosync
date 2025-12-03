import streamlit as st
import json
import os
from pydantic import BaseModel
from typing import List
from gtts import gTTS
import io
from openai import OpenAI

# ============================================================
#  UI CONFIGURATION
# ============================================================
st.set_page_config(layout="wide", page_title="NeuroSync: Agentic Memory", page_icon="🧠")

# Stealth UI theme
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { border-radius: 8px; font-weight: bold; border: 1px solid #30363D; }
    .memory-card { background-color: #1F2937; padding: 20px; border-radius: 10px; border-left: 5px solid #6366F1; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stTextInput>div>div>input { background-color: #161B22; color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 30-MESSAGE CHAT HISTORY (INPUT DATA)
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
# MEMORY SCHEMA (Pydantic)
# ============================================================
class UserProfile(BaseModel):
    user_preferences: List[str]
    emotional_patterns: str
    facts: List[str]

# ============================================================
# HUGGINGFACE ROUTER CLIENT (OpenAI-Compatible)
# ============================================================

@st.cache_resource
def get_hf_client():
    """
    Creates a client that talks to HuggingFace Router using OpenAI-compatible API.
    Token is never exposed to frontend.
    You must set HF_TOKEN in Streamlit secrets.
    """
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=st.secrets["HF_TOKEN"],
    )

def hf_generate(prompt: str, max_new_tokens: int = 300, temperature: float = 0.4) -> str:
    """
    Unified LLM call via HF Router.
    """
    client = get_hf_client()

    try:
        completion = client.chat.completions.create(
            model="HuggingFaceTB/SmolLM3-3B:hf-inference",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ HuggingFace Router Error:\n{e}")
        raise

# ============================================================
# COGNITIVE ENGINE — MEMORY EXTRACTION
# ============================================================
class CognitiveEngine:
    def __init__(self):
        # This file will be created at runtime – no need to commit it to Git
        self.memory_file = "long_term_memory.json"

    def extract_profile(self, history):
        history_text = "\n".join(f"- {m}" for m in history)

        prompt = f"""
You are an AI Memory Architect.

From the chat history below, extract long-term memory and return it in JSON.

CHAT LOG:
{history_text}

Your entire response MUST be ONLY a single valid JSON object.
Do NOT include any explanations, thoughts, analysis, comments, or <think> tags.
Do NOT wrap it in markdown.
Just output JSON exactly like:

{{
  "user_preferences": ["...", "..."],
  "emotional_patterns": "string",
  "facts": ["...", "..."]
}}
"""

        raw = hf_generate(prompt, temperature=0.2)

        # ---- Robust JSON extraction ----
        try:
            start = raw.find("{")
            end = raw.rfind("}")

            if start == -1 or end == -1:
                st.error(f"❌ Model did not return JSON.\nRaw output:\n{raw}")
                return None

            json_str = raw[start:end + 1]
            data = json.loads(json_str)
            profile = UserProfile(**data)
            return profile

        except Exception as e:
            st.error(f"❌ JSON parsing/validation failed.\nRaw output:\n{raw}\n\nError:\n{e}")
            return None

    def save(self, profile: UserProfile):
        with open(self.memory_file, "w") as f:
            f.write(profile.model_dump_json(indent=2))

    def load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return UserProfile(**json.load(f))
        return None

# ============================================================
# PERSONA ENGINE
# ============================================================
def persona_instructions(persona: str) -> str:
    return {
        "Calm Mentor": "Speak slowly, gently, encouragingly. Validate feelings and provide simple next steps.",
        "Witty Peer": "Be playful, quick, lightly humorous without being rude.",
        "Therapist-style Guide": "Reflect emotions, ask guiding questions, avoid diagnosing, supportive tone.",
        "No-Nonsense CTO": "Be direct, concise, prioritization-focused, action-driven.",
        "Neutral": "Respond plainly, factually, without any stylistic flavor."
    }[persona]

def generate_reply(user_msg: str, profile: UserProfile, persona: str = "Neutral") -> str:
    memory_block = f"""
User Preferences: {profile.user_preferences}
Emotional Patterns: {profile.emotional_patterns}
Facts: {profile.facts}
"""

    persona_rules = persona_instructions(persona)

    prompt = f"""
You are an AI assistant replying to a recurring user.

MEMORY:
{memory_block}

PERSONA = {persona}
PERSONA_TRAITS = {persona_rules}

RULES:
- Respect all preferences strictly.
- Adapt tone to emotional patterns.
- Use facts naturally, not creepily.
- No emojis unless user preferences explicitly allow them.
- Respond only to the user's latest message.

User: {user_msg}
Assistant:
"""

    return hf_generate(prompt, temperature=0.5)

# ============================================================
# TEXT TO SPEECH (Optional)
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
    st.title("🧠 NeuroSync — Agentic Memory & Personality Engine (HuggingFace Edition)")
    st.caption("Fully free. Powered by HuggingFace Router. HF token is stored only in backend secrets.")

    brain = CognitiveEngine()

    # ========================================================
    # MEMORY EXTRACTION
    # ========================================================
    st.subheader("1. Extract Structured Memory from 30 Messages")

    col1, col2 = st.columns(2)

    # Left: raw history + button
    with col1:
        with st.expander("📄 View Chat History"):
            st.code(CHAT_HISTORY_30)

        if st.button("🚀 Extract Memory", use_container_width=True):
            with st.spinner("Extracting user profile using HuggingFace Router..."):
                profile = brain.extract_profile(CHAT_HISTORY_30)
                if profile:
                    brain.save(profile)
                    st.session_state["profile"] = profile
                    st.success("✅ Memory extracted and saved!")

    # Right: memory card
    with col2:
        if "profile" not in st.session_state:
            loaded = brain.load()
            if loaded:
                st.session_state["profile"] = loaded

        if "profile" in st.session_state:
            p = st.session_state["profile"]
            st.markdown(f"""
            <div class="memory-card">
            <h4>👤 Extracted User Profile</h4>
            <p><b>Preferences:</b> {", ".join(p.user_preferences)}</p>
            <p><b>Emotional Patterns:</b> {p.emotional_patterns}</p>
            <p><b>Facts:</b> {", ".join(p.facts)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No memory extracted yet. Click 'Extract Memory' to run the pipeline.")

    st.divider()

    # ========================================================
    # INTERACTION / PERSONA ENGINE
    # ========================================================
    st.subheader("2. Persona Engine — Before / After Response")

    persona = st.selectbox(
        "Choose persona:",
        ["Calm Mentor", "Witty Peer", "Therapist-style Guide", "No-Nonsense CTO"]
    )

    user_input = st.chat_input("Say something (e.g., 'I'm stressed about Q4 deadlines')")

    if user_input and "profile" in st.session_state:
        profile = st.session_state["profile"]

        # Neutral vs Persona responses
        neutral = generate_reply(user_input, profile, persona="Neutral")
        styled = generate_reply(user_input, profile, persona=persona)

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
