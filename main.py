# -*- coding: utf-8 -*-
import requests
import logging
import torch
from fastapi import FastAPI, HTTPException, Body, Depends, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from collections import Counter
from typing import Dict, List
import re
import json
import os
import time
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig, AutoQuantizationConfig
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import asyncio
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
import uvicorn
import cProfile
import os
from dotenv import load_dotenv
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Constants
load_dotenv()

# Get API key from environment
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in environment variables. Please set it in a .env file.")
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CREDENTIALS_FILE = "client_secrets.json"  # Ensure this file exists with OAuth credentials
BACKEND_PORT = 8080
FRONTEND_URL = "http://127.0.0.1:8081/index.html"

# Model paths (update these to your actual paths)
MODEL_PATH = r"C:\Users\nagar\Desktop\final\model_xlmr_base"
ONNX_MODEL_PATH = r"C:\Users\nagar\Desktop\final\onnx_model"
QUANTIZED_ONNX_PATH = ONNX_MODEL_PATH + "_quantized"

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer
logger.info(f"Loading tokenizer from {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logger.info(f"Tokenizer loaded from: {tokenizer.name_or_path}")
except Exception as e:
    logger.error(f"Tokenizer loading failed: {str(e)}. Falling back to xlm-roberta-base.")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    logger.info(f"Tokenizer fallback loaded from: {tokenizer.name_or_path}")

# Load and quantize ONNX model
logger.info(f"Loading optimized ONNX model from {QUANTIZED_ONNX_PATH}")
try:
    if not os.path.exists(QUANTIZED_ONNX_PATH):
        logger.info("Quantizing model for faster inference...")
        model = ORTModelForSequenceClassification.from_pretrained(MODEL_PATH, export=True)
        quantizer = ORTQuantizer.from_pretrained(model)
        quantizer.quantize(
            save_dir=QUANTIZED_ONNX_PATH,
            quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        )
    model = ORTModelForSequenceClassification.from_pretrained(
        QUANTIZED_ONNX_PATH,
        session_options=AutoOptimizationConfig.O1()
    )
    logger.info(f"Quantized ONNX model loaded from: {QUANTIZED_ONNX_PATH}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}. Falling back to xlm-roberta-base.")
    model = ORTModelForSequenceClassification.from_pretrained("xlm-roberta-base", export=True)
    model.save_pretrained(ONNX_MODEL_PATH)
    model = ORTModelForSequenceClassification.from_pretrained(ONNX_MODEL_PATH)
    logger.info(f"ONNX model fallback loaded from: {ONNX_MODEL_PATH}")

class CommentAnalyzer:
    def __init__(self):
        self.categories = ["Feedback", "Questions", "Praise", "Suggestions", "Criticism", "Complaints", "Off-Topic/Spam"]
        self.label2id = {cat: i for i, cat in enumerate(self.categories)}
        self.rules = {
            "Feedback": ["comment", "thought", "opinion", "understood", "explanation", "song", "soulful", "vinnanu", "telusu", "cheppu", "review", "input", "take", "view", "perspective"],
            "Questions": ["?", "how", "what", "where", "why", "when", "anti", "enti", "eppudu", "ela", "who", "which", "can you", "do you", "is it"],
            "Praise": ["great", "awesome", "love", "amazing", "good", "best", "thanks", "super", "superb", "soulful", "rocked", "melody", "vere level", "all the best", "nachindi", "baga", "chala", "machi", "kummindi", "bagundi", "excellent", "fantastic", "brilliant", "top", "perfect"],
            "Suggestions": ["should", "recommend", "try", "suggest", "next", "please", "add", "version", "ledu", "vinalanipistundi", "could", "maybe", "would", "hope", "consider"],
            "Criticism": ["bad", "poor", "hate", "worst", "dislike", "not", "chetha", "gadida", "awful", "terrible", "boring", "lame", "waste", "disappointing"],
            "Complaints": ["sucks", "broken", "lag", "fail", "error", "fix", "pichi", "thikkodi", "issue", "problem", "annoying", "crash", "stupid", "ridiculous"],
            "Off-Topic/Spam": ["http", "subscribe", "check my", "promo", "link", "click", "follow", "buy", "free", "offer"]
        }
        self.emoji_map = {"❤": "love", "🥰": "love", "👍": "good", "👎": "bad", "😊": "good", "😡": "bad"}
        self.slang_map = {"machi": "good", "kummindi": "awesome", "bagundi": "good", "picha": "crazy", "keka": "great"}
        self.dataset_file = 'dataset.json'
        self.corrections_file = 'corrections.json'

    def preprocess_text_single(self, text: str) -> str:
        """Preprocess a single comment."""
        for emoji, meaning in self.emoji_map.items():
            text = text.replace(emoji, f" {meaning} ")
        for slang, meaning in self.slang_map.items():
            text = text.replace(slang, f" {meaning} ")
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.strip()

    def preprocess_text(self, texts: List[str]) -> List[str]:
        """Sequential preprocessing for small batches."""
        return [self.preprocess_text_single(text) for text in texts]

    def rule_based_fallback(self, text: str, description: str = "") -> List[str]:
        """Optimized rule-based fallback."""
        text_lower = text.lower()
        desc_lower = description.lower() if description else ""
        categories = set()
        for category, keywords in self.rules.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.add(category)
        return list(categories) if categories else ["Off-Topic/Spam"]

    def zero_shot_categorize(self, text: str) -> List[str]:
        """Zero-shot classification placeholder (kept for compatibility)."""
        return self.rule_based_fallback(text)

    def categorize_comment(self, text: str, description: str) -> List[str]:
        """Categorize a single comment using the model or fallback."""
        processed_text = self.preprocess_text_single(text)
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=32)
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits.detach()).numpy()[0]
        if max(probs) < 0.7:
            return self.rule_based_fallback(text, description)
        else:
            labels = [self.categories[i] for i, p in enumerate(probs) if p >= 0.5]
            return labels if labels else [self.categories[np.argmax(probs)]]

    def categorize_comments(self, texts: List[str], description: str = "") -> List[List[str]]:
        """Optimized batch processing with ONNX Runtime."""
        start_time = time.time()
        try:
            if not texts:
                return []
            processed_texts = self.preprocess_text(texts)
            batch_size = 16
            results = []

            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=32)
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits.detach()).numpy()
                for text, prob in zip(batch, probs):
                    if max(prob) < 0.7:
                        results.append(self.rule_based_fallback(text, description))
                    else:
                        labels = [self.categories[j] for j, p in enumerate(prob) if p >= 0.5]
                        results.append(labels if labels else [self.categories[np.argmax(prob)]])

            logger.info(f"Categorized {len(texts)} comments in {time.time() - start_time:.2f}s")
            return results
        except Exception as e:
            logger.error(f"Model categorization failed: {str(e)}")
            return [self.rule_based_fallback(text, description) for text in texts]

    async def fetch_video_description(self, video_id: str) -> str:
        """Fetch video description from YouTube API."""
        start_time = time.time()
        params = {"part": "snippet", "id": video_id, "key": YOUTUBE_API_KEY}
        response = requests.get(YOUTUBE_VIDEO_URL, params=params)
        if response.status_code != 200:
            logger.error(f"YouTube API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=400, detail=f"YouTube API error: {response.text}")
        data = response.json()
        desc = data["items"][0]["snippet"]["description"] if data.get("items") else ""
        logger.info(f"Fetched description in {time.time() - start_time:.2f}s")
        return desc

    async def fetch_all_comments(self, video_id: str, limit_one_page=True, youtube_service=None) -> List[Dict]:
        """Fetch all comments for a video, including user replies if authenticated."""
        comments = []
        page_token = None
        try:
            description = await self.fetch_video_description(video_id)
            while True:
                params = {
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": 100,
                    "key": YOUTUBE_API_KEY if not youtube_service else None,
                    "pageToken": page_token
                }
                if youtube_service:
                    response = youtube_service.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=page_token
                    ).execute()
                else:
                    response = requests.get(YOUTUBE_API_URL, params=params)
                    if response.status_code != 200:
                        logger.error(f"YouTube API error: {response.status_code} - {response.text}")
                        raise HTTPException(status_code=400, detail=f"YouTube API failed: {response.text}")
                    response = response.json()
                data = response if youtube_service else response
                comment_batch = []
                for item in data.get("items", []):
                    comment_id = item["snippet"]["topLevelComment"]["id"]
                    text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    author_display_name = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                    author_channel_id = item["snippet"]["topLevelComment"]["snippet"]["authorChannelId"]["value"]
                    categories = self.categorize_comment(text, description)
                    zero_shot = self.zero_shot_categorize(text)
                    has_replied = False
                    user_replies = []
                    if youtube_service:
                        try:
                            reply_check = youtube_service.comments().list(part="snippet", parentId=comment_id).execute()
                            user_channel_id = youtube_service.channels().list(mine=True, part="id").execute()["items"][0]["id"]
                            for reply in reply_check.get("items", []):
                                if reply["snippet"]["authorChannelId"]["value"] == user_channel_id:
                                    has_replied = True
                                    user_replies.append({
                                        "reply_id": reply["id"],
                                        "text": reply["snippet"]["textOriginal"],
                                        "timestamp": reply["snippet"]["publishedAt"]
                                    })
                        except Exception as e:
                            logger.error(f"Error checking replies for comment {comment_id}: {str(e)}")
                    comment_batch.append({
                        "comment_id": comment_id,
                        "comment_text": text,
                        "author_display_name": author_display_name,
                        "author_channel_id": author_channel_id,
                        "categories": categories,
                        "zero_shot_categories": zero_shot,
                        "replied": has_replied,
                        "replies": user_replies
                    })
                comments.extend(comment_batch)
                page_token = data.get("nextPageToken")
                if limit_one_page or not page_token:
                    break
            logger.info(f"Fetched total {len(comments)} comments")
            return comments
        except Exception as e:
            logger.error(f"Error fetching comments: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def fetch_my_replies(self, video_id: str, youtube_service=None) -> List[Dict]:
        """Fetch all replies posted by the authenticated account."""
        start_time = time.time()
        my_replies = []
        page_token = None

        if not youtube_service:
            logger.error("YouTube service required for fetching my replies")
            raise HTTPException(status_code=401, detail="Authentication required")

        try:
            my_channel_response = youtube_service.channels().list(part="id", mine=True).execute()
            my_channel_id = my_channel_response["items"][0]["id"]
            logger.info(f"My channel ID: {my_channel_id}")

            total_threads = 0
            while True:
                params = {
                    "part": "snippet,replies",
                    "videoId": video_id,
                    "maxResults": 100,
                    "pageToken": page_token,
                    "textFormat": "plainText"
                }
                response = youtube_service.commentThreads().list(**params).execute()
                total_threads += len(response.get("items", []))

                for item in response.get("items", []):
                    top_comment_id = item["snippet"]["topLevelComment"]["id"]
                    if "replies" in item:
                        for reply in item["replies"]["comments"]:
                            reply_snippet = reply["snippet"]
                            if reply_snippet["authorChannelId"]["value"] == my_channel_id:
                                my_replies.append({
                                    "comment_id": reply["id"],
                                    "comment_text": reply_snippet["textDisplay"],
                                    "author_display_name": reply_snippet["authorDisplayName"],
                                    "author_channel_id": reply_snippet["authorChannelId"]["value"],
                                    "parent_comment_id": reply_snippet["parentId"],
                                    "published_at": reply_snippet["publishedAt"]
                                })
                                logger.debug(f"Found reply: {reply['id']} at {reply_snippet['publishedAt']} to comment {top_comment_id}")

                page_token = response.get("nextPageToken")
                logger.info(f"Processed page, found {len(my_replies)} replies so far, total threads: {total_threads}")
                if not page_token:
                    break

            logger.info(f"Fetched {len(my_replies)} replies by me in {time.time() - start_time:.2f}s")
            return my_replies
        except Exception as e:
            logger.error(f"Error fetching my replies: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch replies: {str(e)}")

    def update_dataset(self, new_comments: List[Dict], corrections: List[Dict] = None) -> List[Dict]:
        """Update the dataset with new comments and corrections."""
        try:
            logger.info("Updating dataset")
            if not os.path.exists(self.dataset_file):
                with open(self.dataset_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_texts = {entry['text'].lower() for entry in existing_data if isinstance(entry, dict) and 'text' in entry}

            new_entries = []
            for comment in new_comments:
                text = comment['comment_text'].strip()
                if text.lower() not in existing_texts and len(text.split()) >= 3:
                    categories = comment.get('categories', self.rule_based_fallback(text))
                    new_entries.append({"text": text, "categories": categories})
                    existing_texts.add(text.lower())

            if corrections:
                new_entries.extend(corrections)

            if new_entries:
                updated_data = existing_data + [{"#newly added comments": []}] + new_entries
                with open(self.dataset_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Added {len(new_entries)} new comments to dataset")
            return new_entries
        except Exception as e:
            logger.error(f"Failed to update dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def cluster_comments(self, comments: List[Dict]) -> Dict[str, List[str]]:
        """Cluster comments if there are more than 100."""
        start_time = time.time()
        if len(comments) > 100:
            texts = [comment["comment_text"] for comment in comments]
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=32)
            outputs = model(**inputs)
            embeddings = outputs.logits
            kmeans = MiniBatchKMeans(n_clusters=min(5, len(comments)), random_state=0, batch_size=100).fit(embeddings)
            clusters = {}
            for i, label in enumerate(kmeans.labels_):
                cluster_name = f"Cluster {label}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(texts[i])
            logger.info(f"Clustered {len(comments)} comments in {time.time() - start_time:.2f}s")
            return clusters
        logger.info("Skipping clustering for <=100 comments")
        return {}

    async def analyze_comments(self, video_id: str, youtube_service=None) -> Dict:
        """Analyze comments and return categorized and frequent comments."""
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        try:
            comments = await self.fetch_all_comments(video_id, limit_one_page=False, youtube_service=youtube_service)

            categorized_comments = {cat: [] for cat in self.categories}
            comment_freq = Counter()
            for comment in comments:
                text = comment["comment_text"]
                for category in comment["categories"]:
                    if category in self.categories:
                        categorized_comments[category].append(comment)
                comment_freq[text.lower()] += 1

            frequent_comments = self.extract_frequent_comments(comments, comment_freq)
            clusters = self.cluster_comments(comments)
            result = {
                "video_id": video_id,
                "categorized": categorized_comments,
                "frequent_comments": frequent_comments,
                "category_breakdown": {cat: len(categorized_comments[cat]) for cat in self.categories},
                "new_comments": [{"text": c["comment_text"], "author_display_name": c["author_display_name"], "categories": c["categories"], "replied": c["replied"], "replies": c["replies"]} for c in comments],
                "clusters": clusters
            }
            logger.info(f"Analyzed {len(comments)} comments in {time.time() - start_time:.2f}s")

            profiler.disable()
            profiler.print_stats(sort='cumulative')
            return result
        except Exception as e:
            logger.error(f"Error in analyze_comments: {str(e)}")
            profiler.disable()
            raise HTTPException(status_code=500, detail=str(e))

    def extract_frequent_comments(self, comments: List[Dict], comment_freq: Counter) -> Dict[str, List[Dict]]:
        """Extract frequent comments (appearing 2+ times)."""
        frequent = {cat: [] for cat in self.categories}
        for comment in comments:
            text = comment["comment_text"].lower()
            count = comment_freq[text]
            if count >= 2:
                for category in comment["categories"]:
                    if category in self.categories:
                        frequent[category].append({
                            "text": comment["comment_text"],
                            "count": count,
                            "comment_ids": [c["comment_id"] for c in comments if c["comment_text"].lower() == text],
                            "categories": comment["categories"],
                            "replied": comment["replied"],
                            "replies": comment["replies"]
                        })
        for category in frequent:
            seen = set()
            frequent[category] = [item for item in frequent[category] if not (item["text"].lower() in seen or seen.add(item["text"].lower()))]
        return frequent

# FastAPI Endpoints
@app.get("/health")
async def health_check():
    """Check if the backend is running."""
    return {"status": "ok", "message": "Backend is running"}

@app.get("/auth")
async def authenticate():
    """Initiate OAuth2 authentication flow."""
    logger.info("Auth route accessed")
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri=f"http://127.0.0.1:{BACKEND_PORT}/oauth2callback"
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return RedirectResponse(url=auth_url)

@app.get("/oauth2callback")
async def oauth2callback(request: FastAPIRequest):
    """Handle OAuth2 callback and save credentials."""
    logger.info("OAuth2 callback route accessed")
    try:
        code = request.query_params.get("code")
        if not code:
            logger.error("No authorization code provided")
            raise ValueError("No authorization code provided")
        flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIALS_FILE,
            SCOPES,
            redirect_uri=f"http://127.0.0.1:{BACKEND_PORT}/oauth2callback"
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials
        youtube_service = build('youtube', 'v3', credentials=credentials)
        youtube_service.channels().list(mine=True, part="id").execute()
        logger.info("YouTube service verified successfully")
        with open("token.json", "w") as token_file:
            token_file.write(credentials.to_json())
        return RedirectResponse(url=FRONTEND_URL)
    except Exception as e:
        logger.error(f"OAuth2 callback error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"message": f"Authentication failed: {str(e)}", "redirect": FRONTEND_URL},
            headers={"Location": FRONTEND_URL}
        )

@app.get("/check_auth")
async def check_auth():
    """Check if the user is authenticated."""
    try:
        credentials = Credentials.from_authorized_user_file("token.json", SCOPES)
        if credentials.expired and credentials.refresh_token:
            logger.info("Access token expired, refreshing")
            credentials.refresh(GoogleRequest())
            with open("token.json", "w") as token_file:
                token_file.write(credentials.to_json())
        return {"status": "authenticated"}
    except FileNotFoundError:
        return {"status": "unauthenticated", "detail": "Please authenticate first via /auth"}
    except Exception as e:
        return {"status": "unauthenticated", "detail": str(e)}

@app.post("/signout")
async def signout():
    """Sign out the user by removing the token."""
    try:
        if os.path.exists("token.json"):
            os.remove("token.json")
        logger.info("User signed out, token removed")
        return {"status": "signed_out"}
    except Exception as e:
        logger.error(f"Error signing out: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sign out failed: {str(e)}")

async def get_youtube_service():
    """Dependency to get authenticated YouTube service."""
    logger.info("Attempting to build YouTube service")
    try:
        if not os.path.exists("token.json"):
            logger.warning("No token.json found - user must authenticate")
            raise HTTPException(status_code=401, detail="Please authenticate first via /auth")

        credentials = Credentials.from_authorized_user_file("token.json", SCOPES)
        logger.info(f"Loaded credentials: client_id={credentials.client_id}, expiry={credentials.expiry}")

        if credentials.expired and credentials.refresh_token:
            logger.info("Access token expired, attempting to refresh")
            credentials.refresh(GoogleRequest())
            with open("token.json", "w") as token_file:
                token_file.write(credentials.to_json())
            logger.info("Token refreshed and saved successfully")
        elif credentials.expired:
            logger.warning("Token expired and no refresh token available")
            raise HTTPException(status_code=401, detail="Token expired, please re-authenticate")

        youtube_service = build('youtube', 'v3', credentials=credentials)
        logger.info("YouTube service built successfully")
        return youtube_service
    except Exception as e:
        logger.error(f"Error building YouTube service: {str(e)}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

def check_ownership(youtube_service, video_id: str) -> bool:
    """Check if the authenticated user owns the video."""
    try:
        video_response = youtube_service.videos().list(part="snippet", id=video_id).execute()
        if not video_response.get("items"):
            logger.warning(f"No video found for ID: {video_id}")
            return False
        video_channel_id = video_response["items"][0]["snippet"]["channelId"]
        user_channels_response = youtube_service.channels().list(mine=True, part="id", maxResults=50).execute()
        if not user_channels_response.get("items"):
            logger.warning("No channels found for user")
            return False
        user_channel_ids = [channel["id"] for channel in user_channels_response["items"]]
        is_owner = video_channel_id in user_channel_ids
        logger.info(f"Ownership check - Video Channel ID: {video_channel_id}, User Channel IDs: {user_channel_ids}, Is Owner: {is_owner}")
        return is_owner
    except Exception as e:
        logger.error(f"Ownership check failed: {str(e)}")
        return False

@app.get("/check_ownership/")
async def check_ownership_endpoint(video_id: str, youtube_service=Depends(get_youtube_service)):
    """Endpoint to check video ownership."""
    try:
        is_owner = check_ownership(youtube_service, video_id)
        logger.info(f"Ownership check for video {video_id}: {is_owner}")
        return {"is_owner": is_owner}
    except Exception as e:
        logger.error(f"Ownership check failed for video {video_id}: {str(e)}")
        return {"is_owner": False}

@app.post("/analyze_comments/")
async def analyze_comments_endpoint(data: dict = Body(...), youtube_service=Depends(get_youtube_service)):
    """Analyze comments for a given video URL."""
    video_url = data.get("video_url")
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube Video ID")
    analyzer = CommentAnalyzer()
    logger.info(f"Analyzing comments for video ID: {video_id.group(1)}")
    result = await analyzer.analyze_comments(video_id.group(1), youtube_service=youtube_service)
    logger.info("Analysis completed")
    return result

@app.post("/reply_comment/")
async def reply_comment(data: dict = Body(...), youtube_service=Depends(get_youtube_service)):
    """Post a reply to a comment."""
    comment_id = data.get("comment_id")
    reply_text = data.get("reply_text")
    if not comment_id or not reply_text:
        raise HTTPException(status_code=400, detail="Comment ID and reply text are required")
    try:
        response = youtube_service.comments().insert(
            part="snippet",
            body={
                "snippet": {
                    "parentId": comment_id,
                    "textOriginal": reply_text
                }
            }
        ).execute()
        reply_id = response["id"]
        logger.info(f"Replied to comment {comment_id} with reply ID {reply_id}")
        return {"message": f"Replied to comment {comment_id}", "reply_id": reply_id}
    except Exception as e:
        logger.error(f"Failed to reply to comment {comment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk_reply/")
async def bulk_reply(data: dict = Body(...), youtube_service=Depends(get_youtube_service)):
    """Bulk reply to multiple comments."""
    logger.info(f"Bulk reply request: {json.dumps(data, indent=2)}")
    comment_ids = data.get("comment_ids", [])
    reply_text = data.get("reply_text")
    video_id = data.get("video_id")
    if not video_id or not comment_ids or not reply_text:
        logger.error(f"Invalid request: video_id={video_id}, comment_ids={comment_ids}, reply_text={reply_text}")
        raise HTTPException(status_code=400, detail="Video ID, comment IDs, and reply text are required")

    if not check_ownership(youtube_service, video_id):
        logger.error(f"User is not the owner of video ID: {video_id}")
        raise HTTPException(status_code=403, detail="You must be the video owner to bulk reply")

    try:
        success_count = 0
        for comment_id in comment_ids:
            logger.debug(f"Replying to comment ID: {comment_id}")
            youtube_service.comments().insert(
                part="snippet",
                body={
                    "snippet": {
                        "parentId": comment_id,
                        "textOriginal": reply_text
                    }
                }
            ).execute()
            success_count += 1
            logger.info(f"Successfully replied to comment ID: {comment_id}")
        logger.info(f"Bulk reply completed: Successfully replied to {success_count} of {len(comment_ids)} comments")
        return {"message": f"Replied to {success_count} comments", "success_count": success_count}
    except Exception as e:
        logger.error(f"Bulk reply failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk reply failed: {str(e)}")

@app.post("/delete_reply/")
async def delete_reply(data: dict = Body(...), youtube_service=Depends(get_youtube_service)):
    """Delete a specific reply."""
    reply_id = data.get("reply_id")
    if not reply_id:
        raise HTTPException(status_code=400, detail="Reply ID is required")
    try:
        youtube_service.comments().delete(id=reply_id).execute()
        logger.info(f"Deleted reply {reply_id}")
        return {"message": f"Reply {reply_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete reply {reply_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correct_comment/")
async def correct_comment(data: dict = Body(...)):
    """Save a correction for a comment's categories."""
    comment_text = data.get("comment_text")
    corrected_categories = data.get("corrected_categories", [])
    try:
        if not os.path.exists("corrections.json"):
            with open("corrections.json", 'w', encoding='utf-8') as f:
                json.dump([], f)
        with open("corrections.json", 'r', encoding='utf-8') as f:
            corrections = json.load(f)
        corrections.append({"text": comment_text, "categories": corrected_categories})
        with open("corrections.json", 'w', encoding='utf-8') as f:
            json.dump(corrections, f, ensure_ascii=False, indent=2)
        logger.info(f"Correction saved for '{comment_text}'")
        return {"message": f"Correction saved for '{comment_text}'"}
    except Exception as e:
        logger.error(f"Failed to save correction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_comment/")
async def delete_comment(data: dict = Body(...), youtube_service=Depends(get_youtube_service)):
    """Delete a comment (only if the user owns the video)."""
    comment_id = data.get("comment_id")
    video_id = data.get("video_id")
    if not comment_id or not video_id:
        raise HTTPException(status_code=400, detail="Comment ID and video ID are required")
    if not check_ownership(youtube_service, video_id):
        logger.error(f"User is not the owner of video ID: {video_id}")
        raise HTTPException(status_code=403, detail="You must be the video owner to delete comments")
    try:
        youtube_service.comments().delete(id=comment_id).execute()
        logger.info(f"Deleted comment {comment_id}")
        return {"message": f"Comment {comment_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete comment {comment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=BACKEND_PORT, reload=True)