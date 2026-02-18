import os
import base64
import json
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
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_screenshot(image_path: str):
    print(f"Processing: {image_path}")
    
    base64_image = encode_image(image_path)
    
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
                        "detail": "high"
                    }
                }
            ]
        }],
        max_tokens=1200,
        temperature=0.0   # máximo precisión y consistencia
    )
    
    raw_text = response.choices[0].message.content.strip()
    
    # Intentar extraer JSON (Grok suele devolverlo limpio)
    try:
        # Si viene con ```json ... ``` lo limpiamos
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
        
        data = json.loads(raw_text)
    except Exception:
        data = {"error": "Failed to parse JSON", "raw": raw_text}
    
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
    
    print(f"Saved: {output_path.name} and .txt\n")
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
