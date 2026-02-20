import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
import asyncio
import logging
import time
from telegram import Update
from telegram.error import RetryAfter
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from parsing import parse_screenshot
from rag import get_query_engine

ENABLE_RAG = os.getenv("ENABLE_RAG", "0") == "1"
query_engine = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram-rag-bot")
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment.")

BOT_MODE = os.getenv("BOT_MODE", "polling").lower()
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook").strip() or "/webhook"
PORT = int(os.getenv("PORT", "8080"))
SCREENSHOTS_DIR = os.getenv("SCREENSHOTS_DIR", "./screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def format_result(data: dict) -> str:
    if "error" in data:
        return f"Error: {data.get('error')}\n\nRaw output:\n{data.get('raw', '')}"

    lines = []
    question = data.get("question", "").strip()
    if question:
        lines.append(f"Question: {question}")

    options = data.get("options", [])
    if options:
        lines.append("")
        lines.extend([opt for opt in options if isinstance(opt, str)])

    correct = data.get("correct")
    if correct:
        lines.append("")
        lines.append(f"Correct answer: {str(correct)}")

    explanation = (data.get("explanation") or "").strip()
    if explanation:
        lines.append("")
        lines.append("Explanation:")
        lines.append(explanation)

    sign_description = (data.get("sign_description") or "").strip()
    if sign_description:
        lines.append("")
        lines.append(f"Sign: {sign_description}")

    if not lines:
        return "No useful text could be extracted from the image."

    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a screenshot (photo or file) and I will extract the question and options."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    logger.info("Received image update_id=%s chat_id=%s", update.update_id, message.chat_id)

    tg_file = None
    if message.photo:
        tg_file = await message.photo[-1].get_file()
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        tg_file = await message.document.get_file()
    else:
        await message.reply_text("Please send an image (photo or file).")
        return

    suffix = Path(tg_file.file_path or "").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    save_path = str(Path(SCREENSHOTS_DIR) / filename)
    await tg_file.download_to_drive(custom_path=save_path)
    logger.info("Downloaded image to %s", save_path)

    await safe_reply(message, "Processing the screenshot. This can take a moment...")

    try:
        logger.info("Parsing screenshot")
        t_parse_start = time.perf_counter()
        data = await asyncio.wait_for(asyncio.to_thread(parse_screenshot, save_path), timeout=120)
        t_parse = time.perf_counter() - t_parse_start
        parsed_text = format_result(data)

        if "error" in data:
            max_len = 3500
            for i in range(0, len(parsed_text), max_len):
                await safe_reply(message, parsed_text[i : i + max_len])
            return

        await safe_reply(message, "Screenshot parsed successfully.")

        output_text = f"{parsed_text}\n\nTiming: parse={t_parse:.2f}s"

        if ENABLE_RAG:
            await safe_reply(message, "Running RAG...")
            global query_engine
            if query_engine is None:
                logger.info("Initializing RAG query engine")
                query_engine = await asyncio.wait_for(asyncio.to_thread(get_query_engine), timeout=90)

            question = data.get("question", "").strip()
            options = data.get("options", [])

            prompt_lines = [
                "Analyze the driver's theory test question (Spanish Permiso B / DGT).",
                "Select the correct option and briefly explain why.",
                "",
                f"Question: {question}",
            ]
            if options:
                prompt_lines.append("")
                prompt_lines.extend(options)

            query = "\n".join(prompt_lines).strip()
            logger.info("Querying RAG")
            t_rag_start = time.perf_counter()
            response = await asyncio.wait_for(asyncio.to_thread(query_engine.query, query), timeout=120)
            t_rag = time.perf_counter() - t_rag_start
            output_text += f"\n\n---\n\nRAG Answer:\n{response}\n\nTiming: rag={t_rag:.2f}s"

        max_len = 3500
        for i in range(0, len(output_text), max_len):
            await safe_reply(message, output_text[i : i + max_len])
    except asyncio.TimeoutError:
        logger.exception("Timed out while processing")
        await safe_reply(message, "Timed out while processing. Try again in a minute.")
    except Exception as e:
        logger.exception("Failed to handle image")
        await safe_reply(message, f"Error while processing: {e}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error", exc_info=context.error)


async def safe_reply(message, text: str, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            await message.reply_text(text)
            return
        except RetryAfter as e:
            await asyncio.sleep(int(e.retry_after) + 1)
        except Exception:
            logger.exception("Failed to reply (attempt %s)", attempt + 1)
            await asyncio.sleep(1)
    # last try without swallowing exceptions
    await message.reply_text(text)


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_error_handler(error_handler)

    if BOT_MODE == "webhook":
        if not WEBHOOK_URL or not WEBHOOK_URL.startswith("https://"):
            raise RuntimeError("WEBHOOK_URL must be set to your https Cloud Run URL.")
        webhook_path = WEBHOOK_PATH.lstrip("/")
        webhook_url = f"{WEBHOOK_URL.rstrip('/')}/{webhook_path}"
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=webhook_path,
            webhook_url=webhook_url,
            drop_pending_updates=True,
        )
    else:
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
