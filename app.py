import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import json
import tempfile
import os
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Lecture Intelligence Assistant",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 AI Lecture Intelligence Assistant")
st.write(
    "Upload lecture audio and generate transcription, summary, key points, "
    "study questions, and evaluation metrics."
)

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1

st.sidebar.header("System Info")
st.sidebar.write("Device:", device)
st.sidebar.write("Speech Model: Whisper Small")
st.sidebar.write("Text Model: FLAN-T5 Large")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload lecture audio",
    type=["wav", "mp3", "mp4", "m4a"]
)

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
def load_text_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def generate_text(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(**inputs, max_new_tokens=400, num_beams=6, no_repeat_ngram_size=3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def count_key_points(text):
    return len([line for line in text.split("\n") if "Key point" in line])

def count_questions(text):
    return text.count("?")

def quality_score(e):
    score = 0
    if e["transcription_word_count"] >= 20: score += 20
    if e["summary_word_count"] >= 30: score += 20
    if e["key_points_count"] >= 5: score += 20
    if e["study_questions_count"] >= 5: score += 20
    if e["total_latency_seconds"] <= 120: score += 20
    elif e["total_latency_seconds"] <= 240: score += 10
    return score

# -----------------------------
# MAIN APP
# -----------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name

    if st.button("Generate Lecture Intelligence"):

        start_time = time.time()

        # -----------------------------
        # TRANSCRIPTION
        # -----------------------------
        st.subheader("Step 1: Transcription")

        asr = load_asr()
        asr_start = time.time()
        result = asr(audio_path)
        asr_latency = round(time.time() - asr_start, 2)

        transcription = result["text"]
        st.text_area("Transcription", transcription, height=200)

        # -----------------------------
        # TEXT GENERATION
        # -----------------------------
        st.subheader("Step 2: AI Outputs")

        tokenizer, model = load_text_model()

        summary_prompt = f"Summarize the lecture in 5 sentences:\n{transcription}"
        key_points_prompt = f"Extract 5 key points:\n{transcription}"
        questions_prompt = f"Generate 5 study questions:\n{transcription}"

        gen_start = time.time()

        summary = generate_text(summary_prompt, tokenizer, model)
        key_points = generate_text(key_points_prompt, tokenizer, model)
        questions = generate_text(questions_prompt, tokenizer, model)

        generation_latency = round(time.time() - gen_start, 2)
        total_latency = round(time.time() - start_time, 2)

        # -----------------------------
        # DISPLAY OUTPUTS
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Summary")
            st.write(summary)

            st.subheader("Key Points")
            st.write(key_points)

        with col2:
            st.subheader("Study Questions")
            st.write(questions)

        # -----------------------------
        # EVALUATION
        # -----------------------------
        evaluation = {
            "transcription_word_count": len(transcription.split()),
            "summary_word_count": len(summary.split()),
            "key_points_count": count_key_points(key_points),
            "study_questions_count": count_questions(questions),
            "asr_latency_seconds": asr_latency,
            "generation_latency_seconds": generation_latency,
            "total_latency_seconds": total_latency
        }

        evaluation["overall_quality_score_out_of_100"] = quality_score(evaluation)

        # -----------------------------
        # DISPLAY EVALUATION (NICE UI)
        # -----------------------------
        st.subheader("Evaluation Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Transcription Words", evaluation["transcription_word_count"])
            st.metric("Summary Words", evaluation["summary_word_count"])
            st.metric("Key Points", evaluation["key_points_count"])

        with col2:
            st.metric("Questions", evaluation["study_questions_count"])
            st.metric("ASR Latency (s)", evaluation["asr_latency_seconds"])
            st.metric("Total Latency (s)", evaluation["total_latency_seconds"])

        st.metric("Quality Score (/100)", evaluation["overall_quality_score_out_of_100"])

        st.info("Evaluation measures performance, structure quality, and latency.")

        # JSON (optional)
        st.subheader("Detailed Evaluation (JSON)")
        st.json(evaluation)

        # -----------------------------
        # DOWNLOAD
        # -----------------------------
        final_output = f"""
Transcription:
{transcription}

Summary:
{summary}

Key Points:
{key_points}

Study Questions:
{questions}

Evaluation:
{json.dumps(evaluation, indent=4)}
"""

        st.download_button("Download Output", final_output)
