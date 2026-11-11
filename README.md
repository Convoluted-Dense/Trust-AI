# Trust-AI: AI Witness Integrity Layer & Surveillance Cross-Examination

Transparent Explainable AI (XAI) surveillance audit dashboard combining real-time incident detection with local Large Language Model cross-examination and visual evidence auditing.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)](https://ollama.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

Artificial Intelligence decisions in high-stakes surveillance and security environments must remain transparent, verifiable, and subject to continuous human oversight. Trust-AI implements an AI Witness Integrity Layer that allows legal auditors, security analysts, and operators to cross-examine computer vision detection results through an interactive natural language interface.

When anomalous physical altercations or security violations are detected, Trust-AI generates temporal incident logs, visual crops, and bounding boxes. Operators can then query an onboard Large Language Model (via Ollama) to cross-examine the AI witness, challenge confidence thresholds, inspect conflicting hypotheses, and verify evidence integrity.

---

## Key Capabilities

- Automated Incident Flagging (seg.py): Computer vision and segmentation pipeline monitoring video streams to detect anomalies (such as physical altercations or unauthorized access) with timestamped event logging.
- Interactive Cross-Examination Interface (app.py): Streamlit web dashboard enabling users to cross-examine the AI system regarding specific flagged frames and reasoning rationale.
- Local Private LLM Integration (Ollama): Utilizes local LLMs (such as LLaMA 3, Mistral, or Gemma) to provide grounded conversational explanations without transmitting sensitive surveillance footage to third-party APIs.
- Evidence Verification & Audit Trails: Maintains structured JSON logs documenting model confidence scores, detected bounding boxes, temporal sequences, and operator annotations.

---

## Architecture Flow

```
[ Video Feed ] ---> [ seg.py Incident Detection ] ---> [ incident_log.json + Frame Crops ]
                                                                      |
                                                                      v
[ Human Auditor ] <---> [ Streamlit UI (app.py) ] <---> [ Ollama Local LLM Audit Layer ]
```

---

## Repository Structure

```
Trust-AI/
├── app.py                 # Streamlit cross-examination dashboard and LLM audit layer
├── seg.py                 # Real-time video processing, altercation segmentation, and incident logger
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Local installation of Ollama (https://ollama.ai/)
- Pull your chosen Ollama model (e.g., llama3):
```bash
ollama pull llama3
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Convoluted-Dense/Trust-AI.git
cd Trust-AI
```

2. Create and activate a Python virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run the Video Detection Pipeline
Process surveillance streams or test video files to detect incidents:
```bash
python seg.py --source video_sample.mp4
```
Detected events and cropped evidence frames are written to the local incident buffer.

### 2. Launch the AI Witness Cross-Examination Dashboard
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser at http://localhost:8501. You can select flagged incident events, inspect frame-by-frame visual evidence, and cross-examine the AI witness on its confidence, visual cues, and environmental context.

---

## License

This project is licensed under the MIT License.
