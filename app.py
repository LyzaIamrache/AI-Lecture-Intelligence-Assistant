import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import json
import tempfile

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Lecture Intelligence Assistant",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 AI Lecture Intelligence Assistant")
st.write("Upload lecture audio and generate transcription and study content.")

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1

st.sidebar.write("Device:", device)

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload lecture audio", type=["mp3", "wav", "mp4"])

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=pipeline_device
    )

@st.cache_resource
def load_text():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def generate(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# MAIN
# -----------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    if st.button("Generate"):

        start = time.time()

        # -----------------------------
        # TRANSCRIPTION
        # -----------------------------
        st.subheader("Transcription")

        asr = load_asr()

        asr_start = time.time()

        # ✅ FIX HERE
        result = asr(audio_path, return_timestamps=True)

        asr_latency = round(time.time() - asr_start, 2)

        transcription = result["text"]
        st.write(transcription)

        # -----------------------------
        # TEXT GENERATION
        # -----------------------------
        tokenizer, model = load_text()

        summary = generate("Summarize:\n" + transcription, tokenizer, model)
        key_points = generate("Give 5 key points:\n" + transcription, tokenizer, model)
        questions = generate("Give 5 questions:\n" + transcription, tokenizer, model)

        gen_latency = round(time.time() - start, 2)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.subheader("Summary")
        st.write(summary)

        st.subheader("Key Points")
        st.write(key_points)

        st.subheader("Questions")
        st.write(questions)

        # -----------------------------
        # EVALUATION (SIMPLE)
        # -----------------------------
        st.subheader("Evaluation")

        st.metric("ASR Latency (s)", asr_latency)
        st.metric("Total Time (s)", gen_latency)
        st.metric("Summary Length", len(summary.split()))

        st.info("WER not shown due to lack of ground truth.")
