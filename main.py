from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import base64
import json
from typing import Optional

app = FastAPI(title="Half-Pint Resale API")

# CORS — allow the PWA and any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client — reads OPENAI_API_KEY from environment automatically
openai_client = OpenAI()

# Google Vision setup — reads GOOGLE_CREDENTIALS_JSON env var (base64-encoded JSON)
def get_vision_client():
    try:
        from google.cloud import vision
        from google.oauth2 import service_account

        creds_b64 = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not creds_b64:
            raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set")

        creds_json = base64.b64decode(creds_b64).decode("utf-8")
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        print(f"Vision client init error: {e}")
        return None

# Access code database
# Format: code -> { "total_uses": int, "used": int, "type": "trial"|"season" }
# In production you'd use a real database (Railway has Postgres).
# For now this works fine for your scale.
ACCESS_CODES: dict = {}

def load_codes():
    """Load codes from CODES_JSON environment variable (JSON string)."""
    codes_json = os.environ.get("CODES_JSON", "{}")
    try:
        return json.loads(codes_json)
    except Exception as e:
        print(f"Error loading codes: {e}")
        return {}

def save_codes(codes: dict):
    """
    NOTE: In-memory only. On Railway, env vars can't be written at runtime.
    For persistence across restarts, swap this for a database call.
    For the sale season, this is fine as long as the server stays up.
    """
    global ACCESS_CODES
    ACCESS_CODES = codes

# --- Request/Response Models ---

class CodeValidationRequest(BaseModel):
    code: str

class AnalyzeRequest(BaseModel):
    code: str
    image1: str  # base64 JPEG — item photo
    image2: str  # base64 JPEG — label photo

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    global ACCESS_CODES
    ACCESS_CODES = load_codes()
    print(f"Loaded {len(ACCESS_CODES)} access codes")

@app.get("/")
async def root():
    return {"message": "Half-Pint Resale API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/validate-code")
async def validate_code(request: CodeValidationRequest):
    """Validate an access code and return remaining uses."""
    code = request.code.strip().upper()

    if code not in ACCESS_CODES:
        raise HTTPException(status_code=404, detail="Invalid access code. Please check and try again.")

    entry = ACCESS_CODES[code]
    total = entry.get("total_uses", 10)
    used = entry.get("used", 0)
    remaining = max(0, total - used)

    return {
        "valid": True,
        "uses_remaining": remaining,
        "total_uses": total,
        "type": entry.get("type", "trial")
    }

@app.post("/analyze-photo")
async def analyze_photo(request: AnalyzeRequest):
    """
    Analyze two photos using AI and return structured clothing item data.
    Photo 1 (image1): The clothing item — analyzed by OpenAI Vision
    Photo 2 (image2): The clothing label — analyzed by Google Cloud Vision OCR
    """
    code = request.code.strip().upper()

    # Validate code
    if code not in ACCESS_CODES:
        raise HTTPException(status_code=403, detail="Invalid or expired access code.")

    entry = ACCESS_CODES[code]
    total = entry.get("total_uses", 10)
    used = entry.get("used", 0)
    remaining = max(0, total - used)

    if remaining <= 0:
        raise HTTPException(status_code=403, detail="No uses remaining on this code.")

    # --- Step 1: Analyze item photo with OpenAI Vision ---
    item_data = {"brand": "", "description": "", "size": "", "gender": "neutral"}

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a children's clothing item for a consignment sale. "
                                "Look at the item and respond ONLY with a JSON object (no markdown, no explanation) with these keys:\n"
                                "- brand: brand name if visible on item, otherwise empty string\n"
                                "- description: concise description with primary color/pattern + any visible brand text/logos on the item + item type (e.g. 'grey Nike sweatpants', 'pink floral dress', 'blue striped polo'). Keep it 2-4 words.\n"
                                "- size: size if visible on item tag/label (e.g. '4T', '2T', 'XS', '6-12m'), otherwise empty string\n"
                                "- gender: 'boy', 'girl', or 'neutral' based on the item's style\n"
                                "Examples: {\"brand\": \"Nike\", \"description\": \"grey Nike sweatpants\", \"size\": \"12-13 YRS\", \"gender\": \"neutral\"}, {\"brand\": \"Carter's\", \"description\": \"pink floral dress\", \"size\": \"4T\", \"gender\": \"girl\"}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image1}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        item_data.update({
            "brand": parsed.get("brand", ""),
            "description": parsed.get("description", ""),
            "size": parsed.get("size", ""),
            "gender": parsed.get("gender", "neutral")
        })
    except Exception as e:
        print(f"OpenAI Vision error: {e}")
        # Non-fatal — continue with label photo

    # --- Step 2: Extract text from label photo with Google Cloud Vision ---
    try:
        vision_client = get_vision_client()
        if vision_client:
            from google.cloud import vision as gvision
            image_bytes = base64.b64decode(request.image2)
            image = gvision.Image(content=image_bytes)
            result = vision_client.text_detection(image=image)

            if result.text_annotations:
                label_text = result.text_annotations[0].description

                # Use GPT to parse the label text into structured data
                parse_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=150,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"This text was extracted from a children's clothing label:\n\n{label_text}\n\n"
                                "Extract and respond ONLY with a JSON object (no markdown) with these keys:\n"
                                "- brand: brand name from label, or empty string\n"
                                "- size: size from label (e.g. '4T', '2T', 'XS', '6-12m'), or empty string\n"
                                "Example: {\"brand\": \"Carter's\", \"size\": \"4T\"}"
                            )
                        }
                    ]
                )

                raw2 = parse_response.choices[0].message.content.strip()
                if raw2.startswith("```"):
                    raw2 = raw2.split("```")[1]
                    if raw2.startswith("json"):
                        raw2 = raw2[4:]
                label_data = json.loads(raw2)

                # Label data overrides item photo data (more reliable)
                if label_data.get("brand"):
                    item_data["brand"] = label_data["brand"]
                if label_data.get("size"):
                    item_data["size"] = label_data["size"]

    except Exception as e:
        print(f"Google Vision / label parse error: {e}")
        # Non-fatal — use whatever we got from photo 1

    # --- Increment usage ---
    ACCESS_CODES[code]["used"] = used + 1
    new_remaining = max(0, total - (used + 1))

    return {
        "brand": item_data.get("brand", ""),
        "description": item_data.get("description", ""),
        "size": item_data.get("size", ""),
        "gender": item_data.get("gender", "neutral"),
        "uses_remaining": new_remaining
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
