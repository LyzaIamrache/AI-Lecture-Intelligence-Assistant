import time
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from jiwer import wer
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1

print("Using device:", device)

# -----------------------------
# AUDIO FILE
# -----------------------------
audio_file = "audio.mp4"

# -----------------------------
# ASR
# -----------------------------
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=pipeline_device
)

asr_start = time.time()
result = asr(audio_file, return_timestamps=True)
asr_latency = round(time.time() - asr_start, 2)

transcription = result["text"].strip()

print("\nTRANSCRIPTION:")
print(transcription)

# -----------------------------
# GROUND TRUTH
# -----------------------------
ground_truth = """
Hello everyone, today I will cover section 1.4, cross-section versus time series data.
Cross-sectional data are collected on different elements at the same point in time or during the same period.
Time series data are collected on the same element or variable at different points in time.
Examples of cross-sectional data include monthly income of employees in March 2026 and exam scores at the end of the semester.
Examples of time series data include monthly employment rates, daily stock prices, and annual rainfall over many years.
"""

# -----------------------------
# SPEECH EVALUATION
# -----------------------------
wer_score = wer(ground_truth.lower(), transcription.lower())
accuracy_percent = round((1 - wer_score) * 100, 2)

print("\nSPEECH EVALUATION")
print("WER:", round(wer_score, 3))
print("Speech Accuracy:", accuracy_percent, "%")
print("ASR Latency:", asr_latency, "seconds")

# -----------------------------
# PROMPTS
# -----------------------------
summary_prompt = f"""
Summarize the lecture in EXACTLY 5 sentences using only the transcription.

Lecture:
{transcription}
"""

key_points_prompt = f"""
Write EXACTLY 5 key points based only on the lecture.

Lecture:
{transcription}
"""

questions_prompt = f"""
Write EXACTLY 5 study questions based only on the lecture.

Lecture:
{transcription}
"""

# -----------------------------
# SUMMARY QUALITY MODEL
# -----------------------------
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
ground_embedding = sim_model.encode(ground_truth, convert_to_tensor=True)

# -----------------------------
# MODEL FUNCTION
# -----------------------------
def run_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def gen(prompt):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    start = time.time()

    summary = gen(summary_prompt)
    key_points = gen(key_points_prompt)
    questions = gen(questions_prompt)

    generation_latency = round(time.time() - start, 2)

    summary_embedding = sim_model.encode(summary, convert_to_tensor=True)
    summary_similarity = util.cos_sim(ground_embedding, summary_embedding).item()
    summary_quality_percent = round(summary_similarity * 100, 2)

    return {
        "model": model_name,
        "wer": round(wer_score, 3),
        "speech_accuracy_percent": accuracy_percent,
        "asr_latency_seconds": asr_latency,
        "summary_words": len(summary.split()),
        "key_points_count": key_points.count("Key point"),
        "questions_count": questions.count("?"),
        "generation_latency_seconds": generation_latency,
        "summary_quality_percent": summary_quality_percent,
        "summary_sample": summary,
        "key_points_sample": key_points,
        "questions_sample": questions
    }

# -----------------------------
# MODEL COMPARISON
# -----------------------------
models = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large"
]

results = []

for m in models:
    print("\nRunning:", m)
    r = run_model(m)
    results.append(r)
    print(json.dumps(r, indent=4))

# -----------------------------
# SAVE RESULTS
# -----------------------------
with open("model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

with open("evaluation_week15.json", "w", encoding="utf-8") as f:
    json.dump({
        "speech_evaluation": {
            "wer": round(wer_score, 3),
            "speech_accuracy_percent": accuracy_percent,
            "asr_latency_seconds": asr_latency
        },
        "model_comparison": results
    }, f, indent=4)

print("\nSaved:")
print("- model_comparison.json")
print("- evaluation_week15.json")
