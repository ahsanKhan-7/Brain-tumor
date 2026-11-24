"""FastAPI backend: ONNX brain MRI classifier + CAM heatmap + LLM explanation.

Runs on Vercel as a single serverless function (api/index.py exposes `app`).
"""
import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():  # local dev only; Vercel sets real env vars
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
META = json.loads((MODEL_DIR / "classes.json").read_text())
CLASSES = META["classes"]
SIZE = META["size"]
MEAN = np.array(META["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array(META["std"], dtype=np.float32).reshape(1, 3, 1, 1)
CAM_W = np.load(MODEL_DIR / "cam_weights.npy")  # (num_classes, channels)

_session = None


def session():
    global _session
    if _session is None:  # lazy load, keeps cold start fast
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    return _session


# Plain language facts per class. The LLM gets these as grounding, it must not invent beyond them.
INFO = {
    "glioma_tumor": {
        "title": "Glioma",
        "what": "A tumor that starts from glial cells, the supporting cells of the brain.",
        "doctors": ["Neurosurgeon", "Neuro-oncologist", "Neurologist", "Radiologist (to review the full scan)"],
        "risk": "high",
    },
    "meningioma_tumor": {
        "title": "Meningioma",
        "what": "A tumor that grows from the meninges, the thin layers that cover the brain. Many are slow growing.",
        "doctors": ["Neurosurgeon", "Neurologist", "Radiologist (to review the full scan)"],
        "risk": "medium",
    },
    "pituitary_tumor": {
        "title": "Pituitary tumor",
        "what": "A growth in the pituitary gland at the base of the brain, which controls many hormones.",
        "doctors": ["Neurosurgeon", "Endocrinologist (hormone specialist)", "Neurologist"],
        "risk": "medium",
    },
    "no_tumor": {
        "title": "No tumor pattern found",
        "what": "The model did not find a tumor pattern among the 3 tumor types it knows.",
        "doctors": ["General physician or Neurologist, only if symptoms continue"],
        "risk": "low",
    },
}

app = FastAPI(title="Brain MRI Assistant")


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return img, (arr - MEAN) / STD


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def jet(v):
    """v in [0,1] -> RGB uint8, simple jet-like colormap without matplotlib."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def make_cam(features, class_idx, base_img):
    """Class Activation Map: weighted sum of last feature maps. Returns (overlay_b64, region_text, focus_share)."""
    f = features[0]  # (C, h, w)
    cam = np.tensordot(CAM_W[class_idx], f, axes=(0, 0))
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    cam_big = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = jet(cam_big).astype(np.float32)
    alpha = (cam_big[..., None] * 0.65)
    overlay = (np.asarray(base_img, dtype=np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=85)

    hot = cam_big >= 0.6
    focus_share = float(hot.mean())
    if hot.any():
        ys, xs = np.nonzero(hot)
        cy, cx = ys.mean() / SIZE, xs.mean() / SIZE
        v = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
        h = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
        region = "center of the scan" if (v == "middle" and h == "center") else f"{v} {h} part of the scan"
    else:
        region = "no single clear area"
    return base64.b64encode(buf.getvalue()).decode(), region, focus_share


def image_warnings(img: Image.Image, top_prob: float):
    warns = []
    small = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    colorfulness = float(np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean())
    if colorfulness > 12:
        warns.append("This looks like a colour photo, not a grayscale medical scan. Results may be meaningless.")
    if top_prob < 0.60:
        warns.append("The model is not confident about this image. Please do not rely on this result.")
    return warns


SYSTEM_PROMPT = """You are a careful medical information assistant inside a brain MRI screening demo.
You receive the output of an image classifier. Explain it to an ordinary person.

Hard rules:
- Write simple, calm, kind language. Short sentences. No jargon without a one line explanation.
- NEVER name, suggest or hint at any medicine, drug, supplement, dose or home remedy.
- NEVER say the person definitely has or does not have a disease. Say the AI found a pattern that may match, and only a doctor can confirm.
- Say which type of doctor to visit (use the doctor list given) and why.
- Give general next steps only: show the full scan to a specialist, carry old reports, write down symptoms, ask questions. Mention emergency care for sudden severe headache, seizure, weakness on one side, trouble speaking, or loss of consciousness.
- Mention the focus area only as "the highlighted area of the image".
- If warnings are given, start by explaining them.
- Do not use dashes as punctuation. Do not use markdown tables. Use short headings with ** ** and short bullet lines starting with a bullet character.
- Maximum 230 words.
- Reply in English."""


def fallback_report(top, warns):
    info = INFO[top]
    lines = [f"**What the AI found**", f"The AI found a pattern that may match: {info['title']}. {info['what']}", ""]
    if warns:
        lines += ["**Please note**"] + [f"• {w}" for w in warns] + [""]
    lines += ["**Which doctor to see**"] + [f"• {d}" for d in info["doctors"]] + [""]
    lines += [
        "**What to do next**",
        "• Show the full scan and old reports to a specialist. Only a doctor can confirm any diagnosis.",
        "• Write down symptoms, when they started, and questions to ask.",
        "• Go to emergency care for sudden severe headache, seizure, weakness on one side, or trouble speaking.",
    ]
    return "\n".join(lines)


BAD_WORDS = re.compile(r"\b(\d+\s?(mg|ml|mcg)|tablet|capsule|dose|dosage|prescri(be|ption)|ibuprofen|paracetamol|acetaminophen|dexamethasone|temozolomide|cabergoline)\b", re.I)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


async def groq_chat(messages: list, max_tokens: int = 700) -> str:
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "messages": messages},
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


async def llm_report(payload: dict):
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        text = await groq_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classifier output:\n{json.dumps(payload, indent=2)}"},
        ])
        if not text or BAD_WORDS.search(text):  # safety net: no medicine talk, ever
            return None
        return text
    except Exception:
        return None


CHAT_SYSTEM_PROMPT = """You are a careful, kind medical information assistant inside a brain MRI screening demo.
You are chatting with someone about the AI scan result(s) they just received (given to you below as JSON, one or more scans).

Hard rules:
- Simple, calm, kind language. Short answers, under 120 words.
- NEVER name, suggest or hint at any medicine, drug, supplement, dose or home remedy.
- NEVER confirm or rule out a diagnosis. Only a doctor can confirm anything.
- You may explain the result, the highlighted area, what terms mean, and what to expect when seeing a doctor.
- For emergency symptoms (sudden severe headache, seizure, weakness on one side, trouble speaking, loss of consciousness) say to seek emergency care immediately.
- If asked something unrelated to this scan or outside your knowledge, say so briefly and redirect to a doctor.
- No markdown tables. Plain short sentences, occasional bullet lines with a bullet character."""

CHAT_BLOCKED_REPLY = (
    "I can't discuss medicines, dosages or prescriptions here. Please ask a doctor about treatment. "
    "Is there something else about your result I can explain?"
)


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": bool(os.environ.get("GROQ_API_KEY")), "model_accuracy": META.get("test_accuracy")}


def classify_one(raw: bytes):
    """Runs the ONNX model on one image. Returns (result_dict_without_report, llm_payload)."""
    if len(raw) > 6 * 1024 * 1024:
        raise HTTPException(413, "One of the images is too large. Please upload images under 6 MB each.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not read one of the files. Please upload JPG or PNG images.")

    base, x = preprocess(img)
    logits, feats = session().run(None, {"image": x})
    probs = softmax(logits[0])
    order = np.argsort(-probs)
    top_i = int(order[0])
    top = CLASSES[top_i]
    warns = image_warnings(img, float(probs[top_i]))
    overlay_b64, region, focus = make_cam(feats, top_i, base)

    predictions = [{"label": INFO[CLASSES[i]]["title"], "key": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in order]
    info = INFO[top]
    risk = info["risk"] if not warns else "uncertain"

    payload = {
        "top_finding": info["title"],
        "meaning": info["what"],
        "confidence_percent": round(float(probs[top_i]) * 100, 1),
        "all_probabilities_percent": {p["label"]: round(p["probability"] * 100, 1) for p in predictions},
        "highlighted_area": region,
        "doctors_to_see": info["doctors"],
        "warnings": warns,
        "model_limits": "Trained only on brain MRI images for 3 tumor types. Not for CT or X-ray. Not a diagnosis.",
    }
    result = {
        "prediction": info["title"],
        "key": top,
        "confidence": round(float(probs[top_i]), 4),
        "risk": risk,
        "predictions": predictions,
        "highlighted_area": region,
        "heatmap": "data:image/jpeg;base64," + overlay_b64,
        "doctors": info["doctors"],
        "warnings": warns,
        "disclaimer": "Educational demo. Not a medical device and not a diagnosis. Please consult a qualified doctor.",
    }
    return result, payload


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    if not (1 <= len(files) <= 3):
        raise HTTPException(400, "Please upload between 1 and 3 images.")
    raws = [await f.read() for f in files]
    pairs = [classify_one(r) for r in raws]
    reports = await asyncio.gather(*[llm_report(payload) for _, payload in pairs])
    results = []
    for (result, _), report in zip(pairs, reports):
        result["report"] = report if report is not None else fallback_report(result["key"], result["warnings"])
        result["report_by_llm"] = report is not None
        results.append(result)
    return {"results": results}


@app.post("/api/chat")
async def chat(body: dict = Body(...)):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(503, "AI chat is not available right now.")
    history = [
        {"role": m.get("role"), "content": str(m.get("content", ""))[:1500]}
        for m in (body.get("messages") or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ][-12:]
    if not history:
        raise HTTPException(400, "No message provided.")
    context = body.get("context") or {}
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT + f"\n\nScan result(s) for this conversation:\n{json.dumps(context, indent=2)}"},
    ] + history
    try:
        text = await groq_chat(messages, max_tokens=400)
    except Exception:
        raise HTTPException(502, "AI chat failed. Please try again.")
    if BAD_WORDS.search(text):
        text = CHAT_BLOCKED_REPLY
    return {"reply": text}


# Local run only: serve the web page from the same server. On Vercel, public/ is served by the CDN.
_public = Path(__file__).resolve().parent.parent / "public"
if _public.exists():
    app.mount("/", StaticFiles(directory=str(_public), html=True), name="public")
