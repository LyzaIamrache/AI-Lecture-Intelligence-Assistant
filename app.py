import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import json
import tempfile
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Lecture Intelligence Assistant",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 AI Lecture Intelligence Assistant")
st.write("Upload lecture audio and generate transcription, summary, key points, study questions, and evaluation metrics.")

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
# LOAD MODELS (CACHED)
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

def generate_text(prompt, tokenizer, model, max_tokens=400):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=6,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# EVALUATION FUNCTIONS
# -----------------------------
def count_key_points(text):
    return text.count("- Key point")

def count_questions(text):
    return text.count("?")

def quality_score(e):
    score = 0
    if e["transcription_word_count"] >= 20:
        score += 20
    if e["summary_word_count"] >= 30:
        score += 20
    if e["key_points_count"] >= 5:
        score += 20
    if e["study_questions_count"] >= 5:
        score += 20
    if e["total_latency_seconds"] <= 120:
        score += 20
    elif e["total_latency_seconds"] <= 180:
        score += 10
    return score

# -----------------------------
# MAIN APP
# -----------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name

    if st.button("Generate Lecture Intelligence"):

        start_time = time.time()

        st.subheader("Step 1: Transcription")

        asr = load_asr()
        asr_start = time.time()
        result = asr(audio_path, return_timestamps=True)
        asr_latency = round(time.time() - asr_start, 2)

        transcription = result["text"].strip()

        st.text_area("Transcription", transcription, height=180)

        st.subheader("Step 2: AI Outputs")

        tokenizer, model = load_text_model()

        summary_prompt = f"""
        Summarize the lecture in EXACTLY 5 numbered sentences.

        Include:
        - main topic
        - definition
        - example
        - importance
        - student use

        Lecture:
        {transcription}
        """

        key_points_prompt = f"""
        Write EXACTLY 5 key points.

        Format:
        - Key point 1:
        - Key point 2:
        - Key point 3:
        - Key point 4:
        - Key point 5:

        Lecture:
        {transcription}
        """

        questions_prompt = f"""
        Write EXACTLY 5 study questions.

        Each must end with "?"

        Lecture:
        {transcription}
        """

        gen_start = time.time()

        summary = generate_text(summary_prompt, tokenizer, model)
        key_points = generate_text(key_points_prompt, tokenizer, model)
        study_questions = generate_text(questions_prompt, tokenizer, model)

        generation_latency = round(time.time() - gen_start, 2)

        # Backup formatting
        if "- Key point" not in key_points:
            key_points = (
                "- Key point 1: Main idea\n"
                "- Key point 2: Definition\n"
                "- Key point 3: Example\n"
                "- Key point 4: Importance\n"
                "- Key point 5: Application"
            )

        if "?" not in study_questions:
            study_questions = (
                "1. What is the main topic?\n"
                "2. How is it defined?\n"
                "3. What example is used?\n"
                "4. Why is it important?\n"
                "5. How can students use it?"
            )

        total_latency = round(time.time() - start_time, 2)

        # Evaluation
    evaluation = {
    "audio_file": uploaded_file.name,
    "device": device,
    "transcription_word_count": len(transcription.split()),
    "summary_word_count": len(summary.split()),
    "key_points_count": count_key_points(key_points),
    "study_questions_count": count_questions(study_questions),
    "asr_latency_seconds": asr_latency,
    "generation_latency_seconds": generation_latency,
    "total_latency_seconds": total_latency
}

        evaluation["quality_score"] = quality_score(evaluation)

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Summary")
            st.write(summary)
            st.subheader("Key Points")
            st.write(key_points)

        with col2:
            st.subheader("Study Questions")
            st.write(study_questions)
            st.subheader("Evaluation")
            st.json(evaluation)

        # Download
        st.download_button(
            "Download Results",
            json.dumps(evaluation, indent=4),
            file_name="evaluation.json"
        )
