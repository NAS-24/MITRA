# MITRA – Memory Identification & Timeline-based Recall Assistant

MITRA is a privacy-first, camera-based assistive system designed to help individuals with memory loss (such as dementia) recognize familiar people and recall contextual information during daily interactions.

---

## 🚀 Project Overview

MITRA uses real-time face recognition from a device camera to identify known individuals from a closed, consent-based database. Once a person is recognized, the system displays relevant information such as name, relationship, last interaction, and notes. It also provides natural language narration with optional voice output.

This is a **software-only solution** that works on laptops or mobile devices without requiring specialized hardware.

---

## 🧩 System Architecture

The project follows a modular architecture:

- **Frontend** → UI, camera interface, display, audio playback  
- **Backend** → APIs, database, system integration  
- **AI Module** → Face detection and recognition  
- **GenAI Module** → Narration, timeline logic, voice output  

---

## ⭐ Features

- Real-time face detection and recognition  
- Private, consent-based database  
- Person information card (name, relationship, notes, last met)  
- Timeline-based interaction tracking  
- Unknown person detection  
- Confidence-based recognition system  
- Caregiver management panel  
- GenAI-powered natural narration  
- Optional voice output (Text-to-Speech)  
- Privacy-first local processing  

---

## 🔁 System Flow

Camera → Face Detection → Face Recognition → Backend → GenAI → Frontend → Voice Output

---

## 🛠 Tech Stack

- **Frontend:** React / HTML-CSS-JS  
- **Backend:** FastAPI / Flask  
- **AI:** OpenCV, MediaPipe, InsightFace / FaceNet  
- **Database:** SQLite  
- **GenAI:** Groq API (Llama 3) / HuggingFace  
- **TTS:** Deepgram / Sarvam AI / Browser SpeechSynthesis  

---

## 🔐 Privacy Approach

- All face recognition is performed locally  
- Uses a closed, consent-based database  
- No external face recognition APIs  
- Stores embeddings instead of raw images (where possible)  
- Sensitive data is not shared without consent  

---

## 📂 Project Structure

```
MITRA/
│
├── frontend/        # UI (camera, cards, dashboard)
├── backend/         # APIs, database logic
├── ai_core/         # face detection & recognition
├── gen_ai/          # narration, prompts, TTS
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## ⚠️ Note

MITRA is an assistive system and does not provide medical diagnosis or treatment.

---

## 📌 One-Line Summary

MITRA is a camera-based assistive system that combines face recognition and AI-driven narration to help users recognize people and recall past interactions in real time.
