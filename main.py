import os
import time
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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

start = time.time()
result = asr(audio_file, return_timestamps=True)
asr_latency = round(time.time() - start, 2)

transcription = result["text"].strip()

print("\nTRANSCRIPTION:")
print(transcription)

# -----------------------------
# PROMPTS
# -----------------------------
summary_prompt = f"""
Summarize the lecture in EXACTLY 5 sentences using only the transcription.
{transcription}
"""

key_points_prompt = f"""
Write EXACTLY 5 key points based only on the lecture.
{transcription}
"""

questions_prompt = f"""
Write EXACTLY 5 study questions based only on the lecture.
{transcription}
"""

# -----------------------------
# MODEL FUNCTION
# -----------------------------
def run_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def gen(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    start = time.time()

    summary = gen(summary_prompt)
    key_points = gen(key_points_prompt)
    questions = gen(questions_prompt)

    latency = round(time.time() - start, 2)

    return {
        "model": model_name,
        "summary_words": len(summary.split()),
        "key_points": key_points.count("Key point"),
        "questions": questions.count("?"),
        "latency": latency
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
    print(r)

# -----------------------------
# SAVE RESULTS
# -----------------------------
with open("model_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved model_comparison.json")
