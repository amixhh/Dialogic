import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import os
import sys
sys.path.append("..") 
from classify_query import classify_query

st.set_page_config(page_title="Dialogic", layout="wide")
st.title("💎 Dialogic - let's add logic to your dialogues")

# Load T5 Base Model + LoRA Adapter
@st.cache_resource
def load_expander_model():
    base_model_name = "google-t5/t5-base"
    lora_path = "amixh/t5-query-expansion-model" 

    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    return model, tokenizer

model, tokenizer = load_expander_model()

# Query Expansion Function
def expand_query(context, ambiguous_query):
    prompt = f"Instruction: ONLY expand the ambiguous query below into a full, self-contained question based on the dialogue context below. DO NOT answer it:\\n{context.strip()}\\nUser: {ambiguous_query.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#streamlit interface
st.markdown("### Step 1: Paste the User-Bot conversation")
context = st.text_area("Conversation Context", height=300)

st.markdown("### Step 2: Enter the ambiguous query")
ambiguous_query = st.text_input("Ambiguous Query")

if st.button("Expand & Classify"):
    if not context.strip() or not ambiguous_query.strip():
        st.warning("Please provide both the conversation context and the ambiguous query.")
    else:
        with st.spinner("🔄 Expanding the query..."):
            expanded = expand_query(context, ambiguous_query)

        with st.spinner("🔍 Classifying the topic and subtopic..."):
            result = classify_query(expanded)

        st.success("✅ Query Expanded and Classified!")
        st.markdown(f"**📝 Expanded Query:** `{expanded}`")
        st.markdown(f"**📂 Topic:** `{result['topic']}`")
        st.markdown(f"**📁 Subtopic:** `{result['subtopic']}`")