# YouTube Comment Insights

Multilabel YouTube comment classification using XLM-RoBERTa with ONNX optimization, built with FastAPI (backend) and Tailwind (frontend).

## Overview
Classifies comments into seven categories: Feedback, Questions, Praise, Suggestions, Criticism, Complaints, and Off-Topic/Spam; includes bulk actions, clustering (~85% accuracy), and user corrections. Weighted F1-score: 0.89; ~1,000 comments in ~8.5s. 

## Features
- Multilabel classification across 7 categories. 
- Real-time YouTube Data API fetch and processing. 
- Bulk reply/update/delete for video owners. 
- User corrections to improve predictions.
- OAuth2-secured owner actions. 
- ONNX quantization for 2–3x faster inference. 

## Prerequisites
- Python 3.12 or 3.13. 
- Git (optional, for cloning).
- Internet (API/model downloads).

## Setup
1. Clone: `git clone https://github.com/Vinaykumar-nvk/youtube-comment-insights.git && cd youtube-comment-insights` 
2. Env: `python -m venv yt_analysis_env && (yt_analysis_env\Scripts\activate || source yt_analysis_env/bin/activate)`
3. Install: `pip install -r requirements.txt` (or install FastAPI, Uvicorn, Transformers, Optimum[onnxruntime], Torch, scikit-learn, Google API libs).
4. Env file: create `.env` with `YOUTUBE_API_KEY=your_api_key`. 
5. OAuth: download `client_secrets.json` from Google Cloud Console to project root. 
6. Run backend: `uvicorn main:app --host 127.0.0.1 --port 8080 --reload`. 
7. Run frontend: `cd frontend && python -m http.server 8081`.

## Usage
- Open http://127.0.0.1:8081/index.html, enter a YouTube URL, then click Analyze.

## Files
- `main.py`: FastAPI backend. 
- `dataset.py`: Dataset utilities.
- `index.html`: UI.
- `dataset.json`: Sample data.
- `model_xlmr_base/`, `onnx_model/`, `onnx_model_quantized/`: Model files.Notes
- Keep `.env`, `client_secrets.json`, `token.json` private; never commit them. [11]
