import streamlit as st
import json
import os
from pydantic import BaseModel
from typing import List
from gtts import gTTS
import io
from openai import OpenAI

# ============================================================
#  UI CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="NeuroSync: Agentic Memory (HuggingFace Edition)",
    page_icon="🧠"
)

# Stealth UI theme
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
# CHAT HISTORY (Assignment Input Data)
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
# HUGGINGFACE ROUTER CLIENT
# ============================================================

@st.cache_resource
def get_hf_client():
    """Create OpenAI-compatible HF Router client."""
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=st.secrets["HF_TOKEN"],  # Stored ONLY in backend
    )

def hf_generate(prompt: str, max_tokens=400, temperature=0.2) -> str:
    """Unified LLM call using Llama 3.2 3B Instruct."""
    client = get_hf_client()
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct:hf-inference",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ HuggingFace Router Error:\n{e}")
        raise

# ============================================================
# MEMORY EXTRACTION ENGINE
# ============================================================

class CognitiveEngine:
    def __init__(self):
        self.memory_file = "long_term_memory.json"  # auto-created

    def extract_profile(self, history):
        history_text = "\n".join(f"- {m}" for m in history)

        prompt = f"""
You are a Memory Architect AI.

Your task:
Return ONLY a valid JSON object. 
NO prose, NO explanation, NO <think>, NO notes.

CHAT LOG:
{history_text}

FORMAT (mandatory):
{{
  "user_preferences": ["...", "..."],
  "emotional_patterns": "string",
  "facts": ["...", "..."]
}}
"""

        raw = hf_generate(prompt, temperature=0.1)

        # --- Strip hidden <think> reasoning blocks ---
        while "<think>" in raw and "</think>" in raw:
            s = raw.find("<think>")
            e = raw.find("</think>") + len("</think>")
            raw = raw[:s] + raw[e:]

        raw = raw.strip()

        # --- Extract JSON safely ---
        try:
            start = raw.find("{")
            end = raw.rfind("}")

            if start == -1 or end == -1:
                st.error(f"❌ Model did not output JSON.\n\nCleaned output:\n{raw}")
                return None

            json_str = raw[start:end + 1]

            data = json.loads(json_str)
            return UserProfile(**data)

        except Exception as e:
            st.error(f"❌ JSON parsing failed.\n\nOutput:\n{raw}\n\nError:\n{e}")
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

def persona_rules(persona: str) -> str:
    return {
        "Calm Mentor": "Gentle, warm, validating, short steps.",
        "Witty Peer": "Playful, casual, lightly humorous.",
        "Therapist-style Guide": "Reflect feelings, ask gentle questions, supportive.",
        "No-Nonsense CTO": "Direct, concise, prioritization-focused, action-driven.",
        "Neutral": "Plain, factual, minimal style."
    }[persona]

def generate_reply(user_msg: str, profile: UserProfile, persona="Neutral") -> str:
    memory_block = f"""
User Preferences: {profile.user_preferences}
Emotional Patterns: {profile.emotional_patterns}
Facts: {profile.facts}
"""

    persona_desc = persona_rules(persona)

    prompt = f"""
You are an assistant replying to a recurring user.

MEMORY:
{memory_block}

PERSONA = {persona}
PERSONA_TRAITS = {persona_desc}

RULES:
- Respect user's preferences STRICTLY.
- Adapt tone to emotional patterns.
- Use facts naturally, not creepily.
- No emojis unless user requests them.

User: {user_msg}
Assistant:
"""

    return hf_generate(prompt, temperature=0.5)

# ============================================================
# OPTIONAL TTS
# ============================================================

def tts(text):
    try:
        speech = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        speech.write_to_fp(fp)
        fp.seek(0)
        return fp
    except:
        return None

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("🧠 NeuroSync — Agentic Memory & Personality Engine (HF Edition)")
    st.caption("100% free. Powered by Llama 3.2–3B via HuggingFace Router.")

    brain = CognitiveEngine()

    # ---------------------------------------------------------
    # MEMORY EXTRACTION
    # ---------------------------------------------------------
    st.subheader("1. Extract User Memory (30 Messages)")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📄 View Chat History"):
            st.code(CHAT_HISTORY_30)

        if st.button("🚀 Extract Memory", use_container_width=True):
            with st.spinner("Extracting memory using Llama 3.2–3B…"):
                profile = brain.extract_profile(CHAT_HISTORY_30)

                if profile:
                    brain.save(profile)
                    st.session_state["profile"] = profile
                    st.success("Memory extracted & saved!")

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
            st.info("Run the extraction to build memory.")

    st.divider()

    # ---------------------------------------------------------
    # PERSONA ENGINE — INTERACTION
    # ---------------------------------------------------------
    st.subheader("2. Persona Engine: Before / After Comparison")

    persona = st.selectbox(
        "Choose persona:",
        ["Calm Mentor", "Witty Peer", "Therapist-style Guide", "No-Nonsense CTO"]
    )

    user_msg = st.chat_input("Ask something… (e.g., 'I'm stressed about deadlines')")

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
