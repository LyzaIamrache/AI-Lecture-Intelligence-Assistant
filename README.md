# 🎧 AI Lecture Intelligence Assistant

An AI-powered system that transforms lecture audio into structured learning content, including transcription, summary, key points, study questions, and evaluation metrics.

---

## 🚀 Project Overview

The **AI Lecture Intelligence Assistant** is an end-to-end AI pipeline designed to help students better understand and review lecture material.

The system takes lecture audio as input and automatically generates:

- 📜 Transcription (speech → text)
- 📝 Summary (5 structured sentences)
- 🔑 Key Points (5 important concepts)
- ❓ Study Questions (5 questions)
- 📊 Evaluation Metrics (latency, quality, structure)

---

## 🧠 Architecture
Audio → Whisper → Transcription → FLAN-T5 → Summary / Key Points / Questions → Evaluation

---

## 🧩 Technologies Used

- **Python**
- **Streamlit** (UI)
- **Hugging Face Transformers**
  - Whisper (Speech Recognition)
  - FLAN-T5 (Text Generation)
- **Torch**
- **Sentence Transformers** (evaluation)

---

## ⚙️ Features

- Upload lecture audio (MP3, WAV, MP4, M4A)
- Automatic transcription using Whisper
- AI-generated structured outputs
- Clean UI with Streamlit
- Built-in evaluation dashboard
- Model comparison (main.py)

---

## 📊 Evaluation

The system evaluates performance using:

- **WER (Word Error Rate)** *(approximate)*
- **Speech Accuracy**
- **Latency (ASR + Generation)**
- **Summary Quality (semantic similarity)**
- **Output Structure (key points & questions count)**

> ⚠️ Note: WER is approximate due to lack of ground truth transcript.

---

##  Model Comparison

Tested models:

- FLAN-T5 Small → Fast, lower quality
- FLAN-T5 Base → Balanced
- FLAN-T5 Large → Best quality (used in final system)

---

##  How to Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt

### 2. Run the application
python -m streamlit run app.py

## Project Structure
├── app.py
├── main.py
├── requirements.txt
├── audio.mp4
├── Video

## Result
Summary Quality: ~81.68%
Key Points Generated: 5
Study Questions Generated: 5

## Future Work
Real-time lecture processing
Multilingual support
Mobile application
Voice output (text-to-speech)
Improved evaluation with real ground truth

## Limitations
Processing time can be high for large models
Output quality depends on audio clarity

## Business Value
### 1. Target users:
Students
Universities
Online learning platforms

### 2. Benefits:
Faster studying
Better lecture understanding
Automated note-taking

## Conclusion
This project demonstrates how AI can transform lecture audio into structured learning content, improving efficiency and accessibility in education.


