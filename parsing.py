import os
import base64
import json
import time
import io
from pathlib import Path
from openai import OpenAI
from PIL import Image  # optional: to check/resize images
from dotenv import load_dotenv

load_dotenv()


# ================== CONFIG ==================
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),        # your xai-... key
    base_url="https://api.x.ai/v1"
)

MODEL = "grok-4"          # or "grok-2-vision-1212" if you want the dedicated vision model
SCREENSHOTS_FOLDER = "Dl screenshots"   # put your phone screenshots here
OUTPUT_FOLDER = "driving_data/parsed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================== PROMPT (English output only) ==================
SYSTEM_PROMPT = """You are an expert DGT Permiso B instructor. The user is preparing for the official English theory exam.

Analyze the screenshot and return ONLY valid JSON:

{
  "question": "full question translated to natural English",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],   // translated
  "correct": "C",   // or null
  "explanation": "full explanation translated to clear, natural English",
  "sign_description": "description of any road sign in English"
}
"""
PARSE_MAX_DIM = int(os.getenv("PARSE_MAX_DIM", "1280"))  # max width/height
PARSE_JPEG_QUALITY = int(os.getenv("PARSE_JPEG_QUALITY", "75"))
PARSE_DETAIL = os.getenv("PARSE_DETAIL", "low").lower()


def encode_image(image_path: str) -> str:
    # Resize/compress to reduce upload size and latency
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if PARSE_MAX_DIM > 0:
            img.thumbnail((PARSE_MAX_DIM, PARSE_MAX_DIM))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=PARSE_JPEG_QUALITY, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def parse_screenshot(image_path: str):
    t0 = time.perf_counter()
    print(f"Processing: {image_path}")
    
    t_encode_start = time.perf_counter()
    base64_image = encode_image(image_path)
    t_encode = time.perf_counter() - t_encode_start
    
    t_api_start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": PARSE_DETAIL
                    }
                }
            ]
        }],
        max_tokens=1200,
        temperature=0.0   # máximo precisión y consistencia
    )
    t_api = time.perf_counter() - t_api_start
    
    raw_text = response.choices[0].message.content.strip()
    
    # Intentar extraer JSON (Grok suele devolverlo limpio)
    t_parse_start = time.perf_counter()
    try:
        # Si viene con ```json ... ``` lo limpiamos
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
        
        data = json.loads(raw_text)
    except Exception:
        data = {"error": "Failed to parse JSON", "raw": raw_text}
    t_parse = time.perf_counter() - t_parse_start
    
    t_save_start = time.perf_counter()
    # Guardar JSON
    output_path = Path(OUTPUT_FOLDER) / f"{Path(image_path).stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Also save a readable .txt for your RAG
    txt_path = Path(OUTPUT_FOLDER) / f"{Path(image_path).stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Question: {data.get('question', '')}\n\n")
        for opt in data.get("options", []):
            f.write(f"{opt}\n")
        f.write(f"\nCorrect answer: {data.get('correct', 'Not indicated')}\n")
        f.write(f"\nExplanation:\n{data.get('explanation', '')}\n")
        if data.get("sign_description"):
            f.write(f"\nSign: {data['sign_description']}\n")
    t_save = time.perf_counter() - t_save_start
    
    total = time.perf_counter() - t0
    print(
        f"Saved: {output_path.name} and .txt | "
        f"encode={t_encode:.2f}s api={t_api:.2f}s parse={t_parse:.2f}s save={t_save:.2f}s total={total:.2f}s\n"
    )
    return data

# ================== RUN ON FOLDER ==================
if __name__ == "__main__":
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(Path(SCREENSHOTS_FOLDER).glob(ext))
    
    print(f"Found {len(all_images)} screenshots to process...\n")
    
    for img_path in sorted(all_images):
        parse_screenshot(str(img_path))
    
    print("Done! All files are in driving_data/parsed/")
    print("You can add them to your RAG index by running your LlamaIndex script again.")
