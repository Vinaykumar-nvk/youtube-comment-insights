# YouTube Comment Insights

Multilabel YouTube comment classification using XLM-RoBERTa with ONNX optimization, built with FastAPI (backend) and Tailwind (frontend). [2][1]

## Overview
Classifies comments into seven categories: Feedback, Questions, Praise, Suggestions, Criticism, Complaints, and Off-Topic/Spam; includes bulk actions, clustering (~85% accuracy), and user corrections. Weighted F1-score: 0.89; ~1,000 comments in ~8.5s. [3][13]

## Features
- Multilabel classification across 7 categories. [2]
- Real-time YouTube Data API fetch and processing. [5]
- Bulk reply/update/delete for video owners. [2]
- User corrections to improve predictions. [3]
- OAuth2-secured owner actions. [12]
- ONNX quantization for 2–3x faster inference. [13][16]

## Prerequisites
- Python 3.12 or 3.13. [1]
- Git (optional, for cloning). [11]
- Internet (API/model downloads). [14]

## Setup
1. Clone: `git clone https://github.com/yourusername/youtube-comment-insights.git && cd youtube-comment-insights` [11]
2. Env: `python -m venv yt_analysis_env && (yt_analysis_env\Scripts\activate || source yt_analysis_env/bin/activate)` [11]
3. Install: `pip install -r requirements.txt` (or install FastAPI, Uvicorn, Transformers, Optimum[onnxruntime], Torch, scikit-learn, Google API libs). [6][12]
4. Env file: create `.env` with `YOUTUBE_API_KEY=your_api_key`. [11]
5. OAuth: download `client_secrets.json` from Google Cloud Console to project root. [12]
6. Run backend: `uvicorn main:app --host 127.0.0.1 --port 8080 --reload`. [12][9]
7. Run frontend: `cd frontend && python -m http.server 8081`. [11]

## Usage
- Open http://127.0.0.1:8081/index.html, enter a YouTube URL, then click Analyze. [12]

## Files
- `main.py`: FastAPI backend. [12]
- `dataset.py`: Dataset utilities. [11]
- `frontend/index.html`: UI. [11]
- `dataset.json`: Sample data. [11]
- `model_xlmr_base/`, `onnx_model/`, `onnx_model_quantized/`: Model files. [13]

## Notes
- Keep `.env`, `client_secrets.json`, `token.json` private; never commit them. [11]
