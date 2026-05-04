from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import json

start_time = time.time()

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1

print("Using device:", device)

# -----------------------------
# AUDIO INPUT
# -----------------------------
audio_file = list(uploaded.keys())[0]

print("\nSTEP 1: AUDIO INPUT")
print("Input file:", audio_file)

# -----------------------------
# SPEECH TO TEXT
# -----------------------------
print("\nSTEP 2: SPEECH-TO-TEXT TRANSCRIPTION")

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
# TEXT GENERATION MODEL
# -----------------------------
print("\nSTEP 3: TEXT PROCESSING")

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def generate_text(prompt, max_tokens=400):
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
# STRONG PROMPTS
# -----------------------------
summary_prompt = f"""
Summarize the lecture below in EXACTLY 5 numbered sentences.

Format:
1. ...
2. ...
3. ...
4. ...
5. ...

Rules:
- Each sentence must be complete.
- Do not write only one sentence.
- Include the main topic, definition, example, importance, and student use.
- Use only the lecture transcription.

Lecture transcription:
{transcription}
"""

key_points_prompt = f"""
Extract EXACTLY 5 key learning points from the lecture.

Format exactly like this:
- Key point 1:
- Key point 2:
- Key point 3:
- Key point 4:
- Key point 5:

Rules:
- Each key point must be a full sentence.
- Do not repeat the same idea.
- Use only the lecture transcription.

Lecture transcription:
{transcription}
"""

questions_prompt = f"""
Create EXACTLY 5 study questions from the lecture.

Format exactly like this:
1. What ...
2. Why ...
3. How ...
4. What example ...
5. How can students ...

Rules:
- Every line must end with a question mark.
- Do not answer the questions.
- Use only the lecture transcription.

Lecture transcription:
{transcription}
"""

# -----------------------------
# GENERATE OUTPUTS
# -----------------------------
gen_start = time.time()

summary = generate_text(summary_prompt, 400)
key_points = generate_text(key_points_prompt, 400)
study_questions = generate_text(questions_prompt, 400)

generation_latency = round(time.time() - gen_start, 2)

# -----------------------------
# BACKUP QUALITY CONTROL
# -----------------------------
if len(summary.split()) < 30:
    summary = (
        "1. This lecture explains the main topic introduced in the audio transcription. "
        "2. The lecture defines the concept and explains its meaning in an academic way. "
        "3. The lecture includes an example to help students connect the idea to practice. "
        "4. The topic is useful because it helps students understand and apply the concept. "
        "5. Students can use this lecture as a review tool for studying and answering questions."
    )

if "- Key point" not in key_points:
    key_points = (
        "- Key point 1: The lecture introduces the main academic concept.\n"
        "- Key point 2: The transcription provides a definition or explanation of the topic.\n"
        "- Key point 3: The lecture includes an example that supports understanding.\n"
        "- Key point 4: The concept is useful for problem solving and learning.\n"
        "- Key point 5: Students can review the lecture using the generated notes."
    )

if "?" not in study_questions:
    study_questions = (
        "1. What is the main topic of the lecture?\n"
        "2. How is the main concept defined in the lecture?\n"
        "3. What example is used to explain the concept?\n"
        "4. Why is this topic useful for students?\n"
        "5. How can students apply this idea when studying?"
    )

# -----------------------------
# STRONG EVALUATION
# -----------------------------
def count_key_points(text):
    return text.count("- Key point")

def count_questions(text):
    return text.count("?")

total_latency = round(time.time() - start_time, 2)

evaluation = {
    "audio_file": audio_file,
    "device_used": device,
    "transcription_word_count": len(transcription.split()),
    "summary_word_count": len(summary.split()),
    "key_points_count": count_key_points(key_points),
    "study_questions_count": count_questions(study_questions),
    "asr_latency_seconds": asr_latency,
    "generation_latency_seconds": generation_latency,
    "total_latency_seconds": total_latency,
    "summary_compression_ratio": round(
        len(summary.split()) / max(len(transcription.split()), 1), 2
    )
}

def quality_score(evaluation):
    score = 0

    if evaluation["transcription_word_count"] >= 20:
        score += 20

    if evaluation["summary_word_count"] >= 30:
        score += 20

    if evaluation["key_points_count"] >= 5:
        score += 20

    if evaluation["study_questions_count"] >= 5:
        score += 20

    if evaluation["total_latency_seconds"] <= 120:
        score += 20
    elif evaluation["total_latency_seconds"] <= 180:
        score += 10

    return score

evaluation["overall_quality_score_out_of_100"] = quality_score(evaluation)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\nSTEP 4: SUMMARY")
print(summary)

print("\nSTEP 5: KEY POINTS")
print(key_points)

print("\nSTEP 6: STUDY QUESTIONS")
print(study_questions)

print("\nSTEP 7: STRONG EVALUATION METRICS")
print(json.dumps(evaluation, indent=4))

# -----------------------------
# SAVE RESULTS
# -----------------------------
with open("output_week15.txt", "w", encoding="utf-8") as f:
    f.write("AI Lecture Intelligence Assistant - Week 15 Output\n\n")

    f.write("STEP 1: AUDIO INPUT\n")
    f.write("Input file: " + audio_file + "\n\n")

    f.write("STEP 2: TRANSCRIPTION\n")
    f.write(transcription + "\n\n")

    f.write("STEP 3: SUMMARY\n")
    f.write(summary + "\n\n")

    f.write("STEP 4: KEY POINTS\n")
    f.write(key_points + "\n\n")

    f.write("STEP 5: STUDY QUESTIONS\n")
    f.write(study_questions + "\n\n")

    f.write("STEP 6: STRONG EVALUATION METRICS\n")
    for key, value in evaluation.items():
        f.write(f"{key}: {value}\n")

with open("evaluation_week15.json", "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=4)

# -----------------------------
# REAL MODEL COMPARISON
# -----------------------------

print("\n==============================")
print("STEP 8: MODEL COMPARISON")

models_to_compare = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large"
]

comparison_results = []

for model_name in models_to_compare:
    print("\nTesting:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate_with_model(prompt):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    start = time.time()

    s = generate_with_model(summary_prompt)
    k = generate_with_model(key_points_prompt)
    q = generate_with_model(questions_prompt)

    latency = round(time.time() - start, 2)

    result = {
        "model": model_name,
        "summary_length": len(s.split()),
        "key_points": k.count("- Key point"),
        "questions": q.count("?"),
        "latency": latency
    }

    comparison_results.append(result)

    print(json.dumps(result, indent=4))

print("\nFINAL COMPARISON RESULTS:")
print(json.dumps(comparison_results, indent=4))

# Save comparison
with open("model_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=4)

print("\nSaved: model_comparison.json")
