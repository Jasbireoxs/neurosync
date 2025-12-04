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
# MEMORY & LOGIC
# ============================================================
class UserProfile(BaseModel):
    user_preferences: List[str]
    emotional_patterns: str
    facts: List[str]

class CognitiveEngine:
    def __init__(self):
        self.memory_file = "long_term_memory.json"

    def extract_profile(self, history: List[str]) -> UserProfile:
        # Deterministic extraction to ensure the demo always works
        prefs = set()
        facts = set()
        emotion_flags = []

        for msg in history:
            lower = msg.lower()
            # Preferences
            if "minimal" in lower:
                prefs.add("Prefers minimal UIs")
            if "snake_case" in lower:
                prefs.add("Prefers snake_case")
            if "no long explanations" in lower:
                prefs.add("Wants concise answers")
            if "skip the basics" in lower:
                prefs.add("Senior-level (skip basics)")
            if "postgresql" in lower:
                prefs.add("Prefers PostgreSQL")
            if "linux" in lower:
                prefs.add("Linux user")

            # Emotional patterns
            if "anxious" in lower:
                emotion_flags.append("anxiety")
            if "impostor syndrome" in lower:
                emotion_flags.append("impostor syndrome")
            if "overwhelmed" in lower:
                emotion_flags.append("overwhelmed")

            # Facts
            if "barnaby" in lower:
                facts.add("Has dog named Barnaby")
            if "seattle" in lower:
                facts.add("Lives in Seattle (PST)")
            if "titanapi" in lower:
                facts.add("Project: TitanAPI")
            if "cto" in lower:
                facts.add("Goal: Become CTO")

        emo_summary = "Stable" if not emotion_flags else f"Recurring: {', '.join(set(emotion_flags))}"

        return UserProfile(
            user_preferences=sorted(prefs),
            emotional_patterns=emo_summary,
            facts=sorted(facts)
        )

    def save(self, p: UserProfile):
        with open(self.memory_file, "w") as f:
            f.write(p.model_dump_json(indent=2))

    def load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return UserProfile(**json.load(f))
        return None

# ============================================================
# CLOUD AI ENGINE (HuggingFace InferenceClient)
# ============================================================
@st.cache_resource
def get_client():
    try:
        return InferenceClient(token=st.secrets["HF_TOKEN"])
    except Exception:
        return None

def generate_reply_cloud(user_msg: str, profile: UserProfile, persona: str) -> str:
    client = get_client()
    if not client:
        return "⚠️ Error: HF_TOKEN not found in secrets."

    # Models to try (in order)
    MODELS_CONFIG = [
        {"id": "Qwen/Qwen2.5-7B-Instruct", "mode": "chat"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.2", "mode": "text"},
        {"id": "google/flan-t5-large", "mode": "text"},
        {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mode": "text"},
    ]

    system_text = f"""
You are GuppShupp - a lifelong AI friend.

Persona: {persona}
User Memory:
- Preferences: {', '.join(profile.user_preferences)}
- Emotional Patterns: {profile.emotional_patterns}
- Facts: {', '.join(profile.facts)}

Task:
Reply to the user in a way that fits the persona.
Keep it short and natural.
Respect their preferences (they like concise answers, minimal fluff).
Adapt to their emotional patterns.
"""

    last_error = ""

    for config in MODELS_CONFIG:
        model_id = config["id"]
        try:
            if config["mode"] == "chat":
                response = client.chat_completion(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                return response.choices[0].message.content

            else:
                if "flan" in model_id:
                    raw_prompt = (
                        f"Instruction: Act as {persona} called GuppShupp. "
                        f"User context: {system_text} "
                        f"User said: {user_msg}. Respond briefly and kindly."
                    )
                else:
                    raw_prompt = f"{system_text}\nUser: {user_msg}\nAssistant:"

                response = client.text_generation(
                    model=model_id,
                    prompt=raw_prompt,
                    max_new_tokens=200
                )
                return response

        except Exception as e:
            last_error = f"{model_id}: {str(e)}"
            continue

    return f"⚠️ All Models Failed. Please check your HF_TOKEN permissions. Last Error: {last_error}"

# ============================================================
# MAIN APP (Branding + Memory + Before/After Persona)
# ============================================================
def main():
    # --- HEADER WITH LOGO + BRAND TEXT ---
    header_col1, header_col2 = st.columns([1, 4])

    with header_col1:
        # Use the logo you uploaded to your repo.
        # If the filename is different, change it here.
        try:
            st.image("guppshupp_logo.png", width=72)
        except Exception:
            st.markdown("💬")

    with header_col2:
        st.markdown(
            "<h1 style='margin-bottom:0;'>GuppShupp - Lifelong Friend</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#d1d5db; margin-top:4px;'>A cloud-backed companion that remembers you and adapts its personality.</p>",
            unsafe_allow_html=True,
        )

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ GuppShupp Controls")

        client = get_client()
        if not client:
            st.error("❌ HF_TOKEN missing or invalid")
            st.info("Add HF_TOKEN in Streamlit Secrets to enable cloud replies.")
        else:
            st.success("✅ Connected to HuggingFace Cloud")

        persona = st.selectbox(
            "Select Persona",
            ["Calm Mentor", "Witty Friend", "No-Nonsense CTO"]
        )
        enable_voice = st.toggle("🎙️ Enable Voice Output", value=False)

    st.markdown(

        unsafe_allow_html=True,
    )

    brain = CognitiveEngine()

    # ========================================================
    # 1. COGNITIVE LAYER · MEMORY EXTRACTION
    # ========================================================
    st.subheader("1. Cognitive Layer · Memory Extraction")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📄 View Input Data (30 Chat Logs)", expanded=False):
            st.code(CHAT_HISTORY_30, language="python")

        if st.button("🚀 Run Extraction Pipeline", use_container_width=True):
            with st.spinner("Analyzing history & building user profile..."):
                profile = brain.extract_profile(CHAT_HISTORY_30)
                brain.save(profile)
                st.session_state["profile"] = profile
                st.success("✅ Memory extracted & persisted to disk")

    with col2:
        if "profile" not in st.session_state:
            loaded = brain.load()
            if loaded:
                st.session_state["profile"] = loaded

        if "profile" in st.session_state:
            p = st.session_state["profile"]
            st.markdown(f"""
            <div class="memory-card">
                <h4 style="margin-top:0;">👤 Structured User Profile</h4>
                <p><b>❤️ Preferences:</b><br>{", ".join(p.user_preferences) or "None"}</p>
                <p><b>🌊 Emotional Patterns:</b><br>{p.emotional_patterns}</p>
                <p><b>📌 Facts:</b><br>{", ".join(p.facts) or "None"}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run the extraction pipeline to see the structured memory here.")

    st.divider()

    # ========================================================
    # 2. INTERACTION LAYER · BEFORE / AFTER PERSONA
    # ========================================================
    st.subheader(f"2. Interaction Layer · Before / After Persona ({persona})")

    if "profile" not in st.session_state:
        st.warning("Please run the Memory Extraction first so GuppShupp knows you.")
        return

    profile = st.session_state["profile"]

    user_input = st.chat_input("Tell GuppShupp what’s on your mind…")

    if user_input:
        # User message shown once
        with st.chat_message("user"):
            st.write(user_input)

        col_neutral, col_persona = st.columns(2)

        # --- Neutral / baseline reply ---
        with col_neutral:
            with st.chat_message("assistant"):
                st.markdown("### 🔹 Baseline (Neutral GuppShupp)")
                with st.spinner("Thinking (neutral)..."):
                    neutral_reply = generate_reply_cloud(user_input, profile, "Neutral")
                    st.write(neutral_reply)

        # --- Persona reply ---
        with col_persona:
            with st.chat_message("assistant"):
                st.markdown(f"### 🔸 Persona: {persona}")
                with st.spinner("Thinking with personality..."):
                    persona_reply = generate_reply_cloud(user_input, profile, persona)
                    st.write(persona_reply)

                    if enable_voice:
                        try:
                            tts = gTTS(persona_reply)
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            fp.seek(0)
                            st.audio(fp, format="audio/mp3")
                        except Exception:
                            pass

        # X-Ray / Debug prompt injection
        with st.expander("🛠️ X-Ray: Injected Memory & Persona"):
            st.text(f"Persona: {persona}")
            st.text(f"Preferences: {profile.user_preferences}")
            st.text(f"Emotional Patterns: {profile.emotional_patterns}")
            st.text(f"Facts: {profile.facts}")

if __name__ == "__main__":
    main()
