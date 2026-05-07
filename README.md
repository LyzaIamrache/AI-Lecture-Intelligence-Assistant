# AI Lecture Intelligence Assistant

An AI-powered educational system that transforms lecture audio into structured learning content using speech recognition and natural language processing.

The system automatically generates:
- Transcription
- Translated Transcription
- Summary
- Key Points
- Study Questions
- Evaluation Metrics

This project improves learning efficiency, accessibility, and multilingual support for students.

---

# Project Overview

The AI Lecture Intelligence Assistant is an end-to-end AI pipeline designed to help students better understand and review lecture material.

The system takes lecture audio as input and automatically generates:
- Lecture Transcription (Speech → Text)
- Translated Transcription
- Structured Summary
- Key Points Extraction
- Study Questions
- Evaluation Metrics

The project combines speech recognition and text generation models into an intelligent educational assistant.

---

# System Architecture

Audio Input  
↓  
Whisper (Speech Recognition)  
↓  
Transcription  
↓  
Translation Module  
↓  
FLAN-T5 (Text Generation)  
↓  
Summary / Key Points / Questions  
↓  
Evaluation Dashboard

---

# Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- Whisper
- FLAN-T5
- Torch
- Sentence Transformers
- Deep Translator
- Pandas

---

# Features

- Upload lecture audio files (MP3, WAV, MP4, M4A)
- Automatic transcription using Whisper
- Translation of transcription into multiple languages
- AI-generated summaries
- Automatic key point extraction
- Study question generation
- Interactive Streamlit interface
- Evaluation dashboard
- Model comparison functionality

---

# Supported Outputs

## 1. Transcription
Converts lecture audio into readable text using Whisper.

## 2. Translated Transcription
Translates the transcription into different languages for multilingual accessibility.

## 3. Summary
Generates a structured summary of the lecture.

## 4. Key Points
Extracts the most important concepts discussed in the lecture.

## 5. Study Questions
Automatically generates review questions for students.

## 6. Evaluation Metrics
Measures:
- Speech recognition quality
- Generation latency
- Semantic similarity
- Output structure quality

---

# Model Comparison

| Model | Performance |
|---|---|
| FLAN-T5 Small | Faster generation but lower quality |
| FLAN-T5 Base | Balanced speed and quality |
| FLAN-T5 Large | Best quality and coherence (selected model) |

---

# Evaluation Results

Example system performance:
- Summary Quality: ~81.68%
- Key Points Generated: 5
- Study Questions Generated: 5

The evaluation dashboard measures:
- ASR latency
- Text generation latency
- Semantic similarity scores
- Output completeness

---

# How to Run the Project

## 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 2. Run the Application

```bash
python -m streamlit run app.py
```

---

# Project Structure

```text
├── app.py
├── main.py
├── requirements.txt
├── audio.mp4
├── outputs/
├── Video/
└── README.md
```

---

# Business Value

## Target Users
- Students
- Universities
- Online learning platforms
- Language learners

## Benefits
- Faster studying and revision
- Better lecture understanding
- Automated note-taking
- Multilingual accessibility
- AI-assisted learning support

---

# Future Work

- Real-time lecture processing
- Mobile application support
- Voice output (Text-to-Speech)
- Improved multilingual translation
- Better evaluation using real ground truth datasets
- Cloud deployment

---

# Limitations

- Processing time may increase for large AI models
- Output quality depends on audio clarity
- Evaluation is limited without complete ground truth datasets
- Large models require higher computational resources

---

# Conclusion

The AI Lecture Intelligence Assistant demonstrates how artificial intelligence can transform lecture audio into structured educational content through transcription, translated transcription, summarization, question generation, and evaluation.

By combining speech recognition, natural language processing, and multilingual support, the system improves educational accessibility and learning efficiency.
