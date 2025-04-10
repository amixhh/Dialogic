import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, LoraConfig
from huggingface_hub import hf_hub_download
import json
import time
import os
import sys
sys.path.append("..")  
from classify_query import classify_query

st.set_page_config(page_title="Dialogic Chat", layout="wide", initial_sidebar_state="collapsed")

#css layout
st.markdown("""
    <style>
        /* Fix the sidebar collapse button */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        /* Remove top whitespace */
        .stApp {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        .main-header {
            position: relative;
            top: 0;
            background-color: white;
            z-index: 998;
            padding: 10px 0;
            text-align: center;
            border-bottom: 1px solid #eee;
            width: 100%;
            margin-bottom: 0;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding-bottom: 90px; /* Extra space for input box */
            min-height: 50px;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 5px 20px 80px 20px;
            min-height: 50px;
            max-height: calc(100vh - 180px);
        }
        .chat-bubble.user {
            background-color: #007AFF;
            color: white;
            border-radius: 16px;
            padding: 10px 15px;
            margin: 10px 0;
            text-align: right;
            align-self: flex-end;
            width: fit-content;
            max-width: 75%;
            margin-left: auto;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }
        .chat-bubble.bot {
            background-color: #F1F0F0;
            color: black;
            border-radius: 16px;
            padding: 10px 15px;
            margin: 10px 0;
            text-align: left;
            align-self: flex-start;
            width: fit-content;
            max-width: 75%;
            margin-right: auto;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }
        .typing-indicator {
            display: flex;
            align-items: center;
            background-color: #F1F0F0;
            color: black;
            border-radius: 18px 18px 18px 0;
            padding: 12px 16px;
            margin: 8px 0;
            width: fit-content;
            margin-right: auto;
        }
        .typing-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #808080;
            margin-right: 4px;
            animation: typing 1.4s infinite ease-in-out;
        }
        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        .expansion-box {
            background-color: #f9f9f9;
            border-left: 5px solid #007AFF;
            padding: 10px 15px;
            margin: 10px 0 30px 0;
            border-radius: 10px;
            font-size: 0.9rem;
        }
        /* Input area fixed at bottom */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 15px 0;
            border-top: 1px solid #eee;
            z-index: 9999;
            box-shadow: 0px -2px 10px rgba(0,0,0,0.05);
        }
        
        /* Style for the form inside input area */
        .input-area form {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        /* Remove form padding */
        .stForm {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Style the text input */
        .stTextInput input {
            border-radius: 20px !important;
            padding: 10px 15px !important;
            font-size: 16px !important;
            border: 1px solid #ddd !important;
        }
        
        /* Style the button */
        .stButton button {
            background-color: #007AFF !important;
            color: white !important;
            border-radius: 20px !important;
            border: none !important;
            padding: 8px 20px !important;
            font-weight: bold !important;
        }
        
        /* Hide Streamlit branding elements */
        footer {
            visibility: hidden;
        }
        #MainMenu {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1 style="margin-bottom: 0.1rem; margin-top: 0;">🌐 Dialogic</h1>
        <h4 style="margin-top: 0; margin-bottom: 0;">Let's add logic to your dialogues</h4>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_expander_model():
    base_model_name = "google-t5/t5-base"
    lora_path = "amixh/t5-query-expansion-model"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config_file = hf_hub_download(repo_id=lora_path, filename="adapter_config.json")
    with open(config_file, "r") as f:
        raw_config = json.load(f)
    allowed_keys = {
        "r", "lora_alpha", "target_modules", "lora_dropout", "bias",
        "task_type", "inference_mode", "fan_in_fan_out", "modules_to_save",
        "init_lora_weights", "base_model_name_or_path", "rank_pattern", "alpha_pattern"
    }
    clean_config = {k: v for k, v in raw_config.items() if k in allowed_keys}
    lora_config = LoraConfig(**clean_config)
    model = PeftModel(base_model, lora_config)
    model.load_adapter(lora_path, adapter_name="default")
    return model, tokenizer

model, tokenizer = load_expander_model()

#I have taken a predefined conversation for the chatbot for testing purpose we can use our own bot here for realtime user-bot conversatoin
predefined_conversation = [
    {"user": "Who won the last Champions League?", "bot": "Real Madrid won the 2023-24 UEFA Champions League."},
    {"user": "How many times have they won?", "bot": "Real Madrid has won 15 Champions League titles."},
    {"user": "Who's their best player?", "bot": "Jude Bellingham is currently their standout performer."},
    {"user": "what about cricket?", "bot": "Virat Kohli is the best cricketer in the world."},
    {"user": "Who's the best cricketer right now in the world?", "bot": "Virat Kohli is widely regarded as one of the best in the world."},
    {"user": "How many centuries does he have?", "bot": "Virat Kohli has scored over 70 international centuries across all formats."},
]


if "chat" not in st.session_state:
    st.session_state.chat = []
if "expanded" not in st.session_state:
    st.session_state.expanded = {}
if "classified" not in st.session_state:
    st.session_state.classified = {}
if "waiting" not in st.session_state:
    st.session_state.waiting = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.markdown('<div class="chat-container"><div class="chat-messages">', unsafe_allow_html=True)

if len(st.session_state.chat) == 0:
    st.markdown("""
        <div style="text-align: center; padding: 40px 20px; color: #888;">
            <div style="font-size: 60px; margin-bottom: 20px;">💬</div>
            <div style="font-size: 18px; margin-bottom: 10px;">No messages yet</div>
            <div style="font-size: 14px;">Type a message to start chatting</div>
        </div>
    """, unsafe_allow_html=True)

# Display chat messages
for i in range(0, len(st.session_state.chat), 2):
    st.markdown(f"<div class='chat-bubble user'>{st.session_state.chat[i]['text']}</div>", unsafe_allow_html=True)
    if i == len(st.session_state.chat) - 2 and st.session_state.waiting:
        st.markdown("""
            <div class='typing-indicator'>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble bot'>{st.session_state.chat[i+1]['text']}</div>", unsafe_allow_html=True)
        if i == len(st.session_state.chat) - 2:
            st.markdown(f"""
                <div class='expansion-box'>
                    <b>📝 Expanded Query:</b><br>{st.session_state.expanded.get("text", "")}<br><br>
                    <b>📂 Topic:</b> {st.session_state.classified.get("topic", "")}<br>
                    <b>📁 Subtopic:</b> {st.session_state.classified.get("subtopic", "")}
                </div>
            """, unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown('<div class="input-area">', unsafe_allow_html=True)
with st.form(key="message_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Type your message:", 
            key="message_input",
            placeholder="Type your message here...",
            label_visibility="collapsed",
            value=st.session_state.user_input
        )
    with col2:
        submitted = st.form_submit_button(
            "Send",
            use_container_width=True
        )
st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_input:
    st.session_state.user_input = "" 
    next_index = len(st.session_state.chat) // 2
    expected = predefined_conversation[next_index]["user"] if next_index < len(predefined_conversation) else None
    if expected and user_input.strip() == expected:
        st.session_state.chat.append({"role": "user", "text": user_input.strip()})
        st.session_state.chat.append({"role": "bot", "text": ""})
        st.session_state.waiting = True
        st.rerun()
elif st.session_state.waiting:
    time.sleep(1.6)
    next_index = len(st.session_state.chat) // 2 - 1
    bot_msg = predefined_conversation[next_index]["bot"]
    st.session_state.chat[-1] = {"role": "bot", "text": bot_msg}
    st.session_state.waiting = False
    context_pairs = []
    for j in range(max(0, len(st.session_state.chat) - 22), len(st.session_state.chat) - 2, 2):
        u = st.session_state.chat[j]["text"]
        b = st.session_state.chat[j + 1]["text"]
        context_pairs.append(f"User: {u}\nBot: {b}")
    context = "\n".join(context_pairs)
    prompt = (
        f"Instruction: ONLY expand the ambiguous query below into a full, "
        f"self-contained question based on the dialogue context below. DO NOT answer it:\n"
        f"{context}\nUser: {st.session_state.chat[-2]['text']}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    expanded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.session_state.expanded = {"text": expanded}
    st.session_state.classified = classify_query(expanded)
    st.rerun()

st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', (event) => {
    // Focus on the input field
    setTimeout(() => {
        const inputField = document.querySelector('.stTextInput input');
        if (inputField) {
            inputField.focus();
        }
    }, 500);
});
</script>
""", unsafe_allow_html=True)