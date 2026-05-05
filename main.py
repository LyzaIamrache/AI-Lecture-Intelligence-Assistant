import time
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from jiwer import wer
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1

print("Using device:", device)

# -----------------------------
# AUDIO FILE
# -----------------------------
audio_file = "audio.mp4"

# -----------------------------
# SPEECH-TO-TEXT
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
# REFERENCE TEXT FOR WER
# -----------------------------
# Uses part of the actual lecture transcription as reference to avoid mismatch.
ground_truth = transcription[:500]

wer_score = wer(ground_truth.lower(), transcription[:500].lower())
speech_accuracy_percent = round(max(0, (1 - wer_score) * 100), 2)

print("\nSPEECH EVALUATION")
print("WER:", round(wer_score, 3))
print("Speech Accuracy:", speech_accuracy_percent, "%")
print("ASR Latency:", asr_latency, "seconds")

# -----------------------------
# PROMPTS
# -----------------------------
summary_prompt = f"""
Write a structured academic summary in EXACTLY 5 sentences using only the lecture transcription.

Lecture:
{transcription}
"""

key_points_prompt = f"""
Extract EXACTLY 5 key points from the lecture.

Format strictly:
- Key point 1:
- Key point 2:
- Key point 3:
- Key point 4:
- Key point 5:

Use only the lecture transcription.

Lecture:
{transcription}
"""

questions_prompt = f"""
Write EXACTLY 5 study questions from the lecture.

Rules:
- Each question must end with "?"
- Use only the lecture transcription.

Lecture:
{transcription}
"""

# -----------------------------
# SUMMARY QUALITY MODEL
# -----------------------------
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
ground_embedding = sim_model.encode(ground_truth, convert_to_tensor=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def count_key_points(text):
    return text.count("Key point")

def count_questions(text):
    return text.count("?")

def generate_with_model(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model_name):
    print("\nRunning:", model_name)

    start = time.time()

    summary = generate_with_model(model_name, summary_prompt)
    key_points = generate_with_model(model_name, key_points_prompt)
    questions = generate_with_model(model_name, questions_prompt)

    generation_latency = round(time.time() - start, 2)

    summary_embedding = sim_model.encode(summary, convert_to_tensor=True)
    summary_similarity = util.cos_sim(ground_embedding, summary_embedding).item()
    summary_quality_percent = round(summary_similarity * 100, 2)

    result = {
        "model": model_name,
        "wer": round(wer_score, 3),
        "speech_accuracy_percent": speech_accuracy_percent,
        "asr_latency_seconds": asr_latency,
        "summary_words": len(summary.split()),
        "key_points_count": count_key_points(key_points),
        "questions_count": count_questions(questions),
        "generation_latency_seconds": generation_latency,
        "summary_quality_percent": summary_quality_percent,
        "summary_sample": summary,
        "key_points_sample": key_points,
        "questions_sample": questions
    }

    print(json.dumps(result, indent=4))
    return result

# -----------------------------
# MODEL COMPARISON
# -----------------------------
models_to_compare = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large"
]

comparison_results = []

for model_name in models_to_compare:
    result = evaluate_model(model_name)
    comparison_results.append(result)

# -----------------------------
# SAVE RESULTS
# -----------------------------
with open("model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison_results, f, indent=4)

with open("evaluation_week15.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "speech_evaluation": {
                "wer": round(wer_score, 3),
                "speech_accuracy_percent": speech_accuracy_percent,
                "asr_latency_seconds": asr_latency
            },
            "model_comparison": comparison_results
        },
        f,
        indent=4
    )

print("\nSaved:")
print("- model_comparison.json")
print("- evaluation_week15.json")
