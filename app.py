import streamlit as st
import json
import os
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from gtts import gTTS
import io
import requests

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NeuroSync: Agentic Memory", page_icon="🧠")

# Custom CSS for "Stealth Startup" aesthetic
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { border-radius: 8px; font-weight: bold; border: 1px solid #30363D; }
    .memory-card { background-color: #1F2937; padding: 20px; border-radius: 10px; border-left: 5px solid #6366F1; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stTextInput>div>div>input { background-color: #161B22; color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

# --- 1. MOCK DATA (30 User Messages) ---
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

# --- 2. MEMORY SCHEMA ---
class UserProfile(BaseModel):
    user_preferences: List[str]
    emotional_patterns: str
    facts: List[str]

# --- 3. HUGGINGFACE INFERENCE ---
def hf_generate(prompt: str, max_new_tokens=400, temperature=0.4):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        }
    }

    resp = requests.post(API_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text}")

    data = resp.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    raise RuntimeError(f"Unexpected HF response format: {data}")

# --- 4. COGNITIVE ENGINE ---
class CognitiveEngine:
    def __init__(self):
        self.memory_file = "long_term_memory.json"

    def extract_profile(self, history):
        history_text = "\n".join(f"- {m}" for m in history)

        prompt = f"""
You are a Memory Architect AI. From the chat history below, extract structured long-term memory.

CHAT LOG:
{history_text}

Return ONLY valid JSON:

{{
  "user_preferences": ["...", "..."],
  "emotional_patterns": "string",
  "facts": ["...", "..."]
}}
"""

        raw = hf_generate(prompt, temperature=0.2)

        # Extract JSON
        try:
            json_str = raw[raw.index("{"): raw.rindex("}") + 1]
            data = json.loads(json_str)
            return UserProfile(**data)
        except Exception as e:
            st.error(f"JSON parsing failed.\nRaw output:\n{raw}\nError:\n{e}")
            return None

    def save(self, profile):
        with open(self.memory_file, "w") as f:
            f.write(profile.model_dump_json(indent=2))

    def load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return UserProfile(**json.load(f))
        return None

# --- 5. PERSONALITY ENGINE ---
def persona_instructions(persona):
    return {
        "Calm Mentor": "speak calmly, validate emotions, provide gentle guidance.",
        "Witty Peer": "be playful, witty, casual but helpful.",
        "Therapist-style Guide": "reflect feelings, ask guiding questions, avoid diagnosing.",
        "No-Nonsense CTO": "be direct, concise, cut scope, push action.",
        "Neutral": "be plain, factual and simple."
    }[persona]

def generate_reply(user_msg, profile, persona="Neutral"):
    memory_block = f"""
User Preferences: {profile.user_preferences}
Emotional Patterns: {profile.emotional_patterns}
Facts: {profile.facts}
"""

    instructions = persona_instructions(persona)

    prompt = f"""
You are an AI assistant replying to a recurring user.

MEMORY:
{memory_block}

PERSONA = {persona}
PERSONA TRAITS = {instructions}

RULES:
- Respect preferences strictly.
- Adapt to emotional patterns.
- Use facts naturally.
- No emojis unless user prefers them.
- Respond only to the last message.

User: {user_msg}
Assistant:
"""

    return hf_generate(prompt, temperature=0.5)

def tts(text):
    try:
        tts_obj = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        tts_obj.write_to_fp(fp)
        return fp
    except:
        return None


# --- 6. MAIN UI ---
def main():
    st.title("🧠 NeuroSync: Agentic Memory (HuggingFace · Free · Backend Secrets)")

    st.info("This version uses **HuggingFace Inference API** with **your token stored in Streamlit Secrets**. No frontend key required.")

    brain = CognitiveEngine()

    # --- Memory extraction UI ---
    st.subheader("1. Extract Memory From 30 Messages")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📄 View Chat History"):
            st.code(CHAT_HISTORY_30)

        if st.button("🚀 Extract Memory"):
            with st.spinner("Extracting memory using HuggingFace model..."):
                profile = brain.extract_profile(CHAT_HISTORY_30)
                if profile:
                    brain.save(profile)
                    st.session_state["profile"] = profile
                    st.success("Memory extracted and stored!")

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
            st.info("No memory extracted yet.")

    st.divider()

    # --- Interaction Layer ---
    st.subheader("2. Persona Engine · Before/After")

    persona = st.selectbox(
        "Choose persona:",
        ["Calm Mentor", "Witty Peer", "Therapist-style Guide", "No-Nonsense CTO"]
    )

    user_input = st.chat_input("Say something (e.g., 'I'm stressed about deadlines')")

    if user_input and "profile" in st.session_state:
        p = st.session_state["profile"]

        neutral = generate_reply(user_input, p, persona="Neutral")
        styled = generate_reply(user_input, p, persona=persona)

        col1, col2 = st.columns(2)

        with col1:
            with st.chat_message("assistant"):
                st.markdown("**Neutral Response:**")
                st.write(neutral)

        with col2:
            with st.chat_message("assistant"):
                st.markdown(f"**Persona: {persona}**")
                st.write(styled)

                # Optional voice
                audio = tts(styled)
                if audio:
                    st.audio(audio, format="audio/mp3")

if __name__ == "__main__":
    main()
