import streamlit as st
import json
import os
from pydantic import BaseModel
from typing import List
from gtts import gTTS
import io
from huggingface_hub import InferenceClient

# ============================================================
#  UI CONFIG  (GuppShupp - Lifelong Friend)
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="GuppShupp - Lifelong Friend",
    page_icon="💬"
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
# ASSIGNMENT DATA (30 MESSAGES)
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
# MEMORY MODULE
# ============================================================
class UserProfile(BaseModel):
    user_preferences: List[str]
    emotional_patterns: str
    facts: List[str]


class CognitiveEngine:
    def __init__(self):
        self.memory_file = "long_term_memory.json"

    def extract_profile(self, history: List[str]) -> UserProfile:
        prefs, facts, emos = set(), set(), []

        for msg in history:
            l = msg.lower()

            if "minimal" in l: prefs.add("Prefers minimal UIs")
            if "snake_case" in l: prefs.add("Prefers snake_case")
            if "no long explanations" in l: prefs.add("Wants concise answers")
            if "skip the basics" in l: prefs.add("Senior-level (skip basics)")
            if "postgresql" in l: prefs.add("Prefers PostgreSQL")
            if "linux" in l: prefs.add("Linux user")

            if "anxious" in l: emos.append("anxiety")
            if "impostor syndrome" in l: emos.append("impostor syndrome")
            if "overwhelmed" in l: emos.append("overwhelmed")

            if "barnaby" in l: facts.add("Has dog named Barnaby")
            if "seattle" in l: facts.add("Lives in Seattle (PST)")
            if "titanapi" in l: facts.add("Project: TitanAPI")
            if "cto" in l: facts.add("Goal: Become CTO")

        emo_summary = "Stable" if not emos else f"Recurring: {', '.join(set(emos))}"

        return UserProfile(
            user_preferences=sorted(prefs),
            emotional_patterns=emo_summary,
            facts=sorted(facts)
        )

    def save(self, profile: UserProfile):
        with open(self.memory_file, "w") as f:
            f.write(profile.model_dump_json(indent=2))

    def load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return UserProfile(**json.load(f))
        return None


# ============================================================
# CLOUD AI (HuggingFace InferenceClient)
# ============================================================
@st.cache_resource
def get_client():
    try:
        return InferenceClient(token=st.secrets["HF_TOKEN"])
    except:
        return None


def generate_reply_cloud(user_msg, profile: UserProfile, persona: str):
    client = get_client()
    if not client:
        return "⚠️ Missing or invalid HF_TOKEN."

    MODELS = [
        {"id": "Qwen/Qwen2.5-7B-Instruct", "mode": "chat"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.2", "mode": "text"},
        {"id": "google/flan-t5-large", "mode": "text"},
        {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mode": "text"},
    ]

    system_prompt = f"""
You are GuppShupp - a lifelong AI friend.

Persona: {persona}
User Preferences: {', '.join(profile.user_preferences)}
Emotional Patterns: {profile.emotional_patterns}
Facts: {', '.join(profile.facts)}

Keep responses short, warm, and natural.
Adapt tone to persona.
"""

    last_err = ""

    for m in MODELS:
        try:
            if m["mode"] == "chat":
                resp = client.chat_completion(
                    model=m["id"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=200
                )
                return resp.choices[0].message.content

            else:
                prompt = f"{system_prompt}\nUser: {user_msg}\nAssistant:"
                resp = client.text_generation(
                    model=m["id"],
                    prompt=prompt,
                    max_new_tokens=200
                )
                return resp

        except Exception as e:
            last_err = str(e)
            continue

    return f"⚠️ All models failed. Last error: {last_err}"


# ============================================================
# MAIN APP (Clean UI + Logo + Memory + Before/After)
# ============================================================
def main():

    # HEADER
    c1, c2 = st.columns([1, 4])
    with c1:
        try:
            st.image("guppshupp_logo.png", width=72)
        except:
            st.write("💬")

    with c2:
        st.markdown("<h1>GuppShupp - Lifelong Friend</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#d1d5db;'>Your personal AI companion that remembers you.</p>",
                    unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.title("⚙️ Controls")

        client = get_client()
        if client:
            st.success("Connected to HuggingFace")
        else:
            st.error("HF_TOKEN missing")

        persona = st.selectbox("Persona", ["Calm Mentor", "Witty Friend", "No-Nonsense CTO"])
        enable_voice = st.toggle("🎙️ Voice Output")

    brain = CognitiveEngine()

    # MEMORY EXTRACTION SECTION
    st.subheader("1. Memory Extraction")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("View 30 Messages"):
            st.code(CHAT_HISTORY_30)

        if st.button("Extract Memory", use_container_width=True):
            with st.spinner("Building user profile..."):
                profile = brain.extract_profile(CHAT_HISTORY_30)
                brain.save(profile)
                st.session_state["profile"] = profile
                st.success("Memory Extracted!")

    with col2:
        prof = st.session_state.get("profile", brain.load())
        if prof:
            st.session_state["profile"] = prof
            st.markdown(f"""
            <div class="memory-card">
                <h4>👤 User Memory</h4>
                <b>Preferences:</b> {", ".join(prof.user_preferences)}<br>
                <b>Emotions:</b> {prof.emotional_patterns}<br>
                <b>Facts:</b> {", ".join(prof.facts)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Extract memory to populate this section.")

    st.divider()

    # BEFORE / AFTER PERSONALITY SECTION
    st.subheader(f"2. Persona Comparison ({persona})")

    if "profile" not in st.session_state:
        st.warning("Extract memory first.")
        return

    profile = st.session_state["profile"]

    user_msg = st.chat_input("Talk to GuppShupp...")
    if user_msg:

        with st.chat_message("user"):
            st.write(user_msg)

        left, right = st.columns(2)

        # Neutral Response
        with left:
            with st.chat_message("assistant"):
                st.markdown("### 🔹 Neutral Response")
                with st.spinner("Thinking..."):
                    neutral = generate_reply_cloud(user_msg, profile, "Neutral")
                    st.write(neutral)

        # Persona Response
        with right:
            with st.chat_message("assistant"):
                st.markdown(f"### 🔸 {persona} Response")
                with st.spinner("Thinking..."):
                    styled = generate_reply_cloud(user_msg, profile, persona)
                    st.write(styled)

                    if enable_voice:
                        try:
                            tts = gTTS(styled)
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            fp.seek(0)
                            st.audio(fp, format="audio/mp3")
                        except:
                            pass

if __name__ == "__main__":
    main()
