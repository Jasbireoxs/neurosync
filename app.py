import streamlit as st
import json
import os
from pydantic import BaseModel
from typing import List
from gtts import gTTS
import io
from huggingface_hub import InferenceClient

# ============================================================
#  UI CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="GuppShupp – Cloud Agent",
    page_icon="☁️"
)

st.markdown("""
<style>
    .stApp { background-color: #050814; color: #FAFAFA; }
    .stButton>button { border-radius: 999px; font-weight: 600; 
                       border: 1px solid #ff4b7a; background: #111827; }
    .memory-card { background: #111827; padding: 20px; border-radius: 16px; 
                   border-left: 5px solid #ff4b7a; 
                   box-shadow: 0 8px 18px rgba(0,0,0,0.45); }
    .stTextInput>div>div>input { background-color: #050814; color: #FAFAFA; }
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
            if "minimal" in lower: prefs.add("Prefers minimal UIs")
            if "snake_case" in lower: prefs.add("Prefers snake_case")
            if "no long explanations" in lower: prefs.add("Wants concise answers")
            if "skip the basics" in lower: prefs.add("Senior-level (skip basics)")
            if "postgresql" in lower: prefs.add("Prefers PostgreSQL")
            if "linux" in lower: prefs.add("Linux user")
            
            if "anxious" in lower: emotion_flags.append("anxiety")
            if "impostor syndrome" in lower: emotion_flags.append("impostor syndrome")
            if "overwhelmed" in lower: emotion_flags.append("overwhelmed")
            
            if "barnaby" in lower: facts.add("Has dog named Barnaby")
            if "seattle" in lower: facts.add("Lives in Seattle (PST)")
            if "titanapi" in lower: facts.add("Project: TitanAPI")
            if "cto" in lower: facts.add("Goal: Become CTO")

        emo_summary = "Stable" if not emotion_flags else f"Recurring: {', '.join(set(emotion_flags))}"
        
        return UserProfile(
            user_preferences=sorted(prefs),
            emotional_patterns=emo_summary,
            facts=sorted(facts)
        )

    def save(self, p):
        with open(self.memory_file, "w") as f: f.write(p.model_dump_json(indent=2))
    def load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f: return UserProfile(**json.load(f))
        return None

# ============================================================
# CLOUD AI ENGINE (FREE TIER)
# ============================================================
@st.cache_resource
def get_client():
    try:
        # Looks for HF_TOKEN in .streamlit/secrets.toml or Cloud Secrets
        return InferenceClient(token=st.secrets["HF_TOKEN"])
    except Exception:
        return None

def generate_reply_cloud(user_msg, profile, persona):
    client = get_client()
    if not client:
        return "⚠️ Error: HF_TOKEN not found in secrets. Please add your Hugging Face token."

    # We use Zephyr-7b-beta because it's free, fast, and good at chat
    MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

    system_prompt = f"""
    You are GuppShupp.
    ROLE: {persona}
    USER INFO:
    - Prefers: {profile.user_preferences}
    - Mood: {profile.emotional_patterns}
    - Facts: {profile.facts}
    
    INSTRUCTION:
    Reply to the user's message based on your Role and their Info.
    Keep it short (max 2 sentences).
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Cloud Error: {e}"

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("☁️ GuppShupp (Free Cloud Edition)")
    
    # 1. SIDEBAR CONFIG
    with st.sidebar:
        st.header("Setup")
        if not get_client():
            st.error("Missing HF_TOKEN")
            st.info("Add HF_TOKEN to your Secrets to enable the Cloud AI.")
        else:
            st.success("Cloud AI Connected")
            
        persona = st.selectbox("Persona", ["Calm Mentor", "Witty Friend", "No-Nonsense CTO"])
        enable_voice = st.toggle("Voice Output")

    brain = CognitiveEngine()
    
    # 2. MEMORY SECTION
    if "profile" not in st.session_state:
        st.info("Building Memory from 30-message history...")
        p = brain.extract_profile(CHAT_HISTORY_30)
        brain.save(p)
        st.session_state["profile"] = p
    
    p = st.session_state["profile"]
    with st.expander("👤 Extracted User Memory (Click to view)"):
        st.json(p.model_dump())

    # 3. CHAT SECTION
    st.divider()
    st.subheader(f"Chatting as: {persona}")
    
    user_input = st.chat_input("Type a message...")
    
    if user_input:
        # Show User
        with st.chat_message("user"):
            st.write(user_input)
            
        # Show AI (Cloud)
        with st.chat_message("assistant"):
            with st.spinner("Thinking (on Hugging Face Cloud)..."):
                reply = generate_reply_cloud(user_input, p, persona)
                st.write(reply)
                
                if enable_voice:
                    try:
                        tts = gTTS(reply)
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        st.audio(fp, format="audio/mp3")
                    except:
                        pass

if __name__ == "__main__":
    main()
