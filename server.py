import json
import re
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from features import add_features, FEATURE_COLS

app = FastAPI(title="SpamLens")
model = joblib.load("spam_model.joblib")
meta = json.loads(Path("results.json").read_text())

URGENT = {"urgent", "now", "free", "win", "winner", "won", "claim", "prize", "cash",
          "offer", "limited", "expires", "immediately", "congratulations", "reward",
          "guaranteed", "call", "txt"}


class Msg(BaseModel):
    text: str


def frame(texts):
    return add_features(pd.DataFrame({"text": texts}))[["text"] + FEATURE_COLS]


@app.post("/api/analyze")
def analyze(m: Msg):
    text = " ".join(m.text.split())[:1000]
    if not text:
        raise HTTPException(400, "Message is empty")
    words = text.split()
    base = frame([text])
    p = float(model.predict_proba(base)[0, 1])
    # remove each word in turn: the drop in spam probability is that word's impact
    without = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    impact = p - model.predict_proba(frame(without))[:, 1]
    f = base.iloc[0]
    clean = [re.sub(r"\W", "", w.lower()) for w in words]
    return {
        "probability": p,
        "words": [{"w": w, "impact": float(i)} for w, i in zip(words, impact)],
        "signals": {
            "url": bool(f["has_url"]),
            "phone": bool(re.search(r"\d[\d\s-]{6,}\d", text)),
            "money": bool(f["has_currency"]),
            "urgency": sorted({c for c in clean if c in URGENT}),
            "caps": round(float(f["caps_ratio"]), 2),
        },
        "stats": {"chars": len(text), "words": len(words)},
    }


@app.get("/api/metrics")
def metrics():
    return {"best": meta["best_model"], "results": meta["results"]}


@app.get("/")
def home():
    return FileResponse("static/index.html")