"""
agent_logic.py
──────────────
All backend / AI logic for the Personal AI Chef & Pantry Agent.

SDK: google-genai (new, replaces deprecated google-generativeai)
      https://github.com/googleapis/python-genai
Model: gemini-2.0-flash (fast, cheap, current)
"""

import os
import logging
import tempfile
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── New Google GenAI SDK (google-genai package) ───────────────────────────────
from google import genai
from google.genai import types as genai_types

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BARK_KEY       = os.getenv("BARK_KEY", "")

# Chat: fast model — streaming, everyday cooking Q&A
CHAT_MODEL   = "gemini-3-flash-preview"
# Settlement & shopping: reasoning model — careful inference from conversation
SETTLE_MODEL = "gemini-3.1-pro-preview"

# ── State file ────────────────────────────────────────────────────────────────
STATE_FILE = Path("state.md")

DEFAULT_STATE = """\
# 核心状态 (State)

## 👤 成员与营养需求


## 🍳 厨具与餐具
- 炒锅、烤箱、微波炉、蒸锅 instant pot 砂锅

## 🧊 冰箱库存 & ⏳ 剩菜


## 👅 动态口味偏好 (AI 自动更新区)

"""

# ── Prompts ───────────────────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """\
你是一个私人主厨智能体（Personal AI Chef & Pantry Agent）。
你的长期记忆保存在下方的「当前状态」中——这是真实的冰箱库存、口味偏好和厨具信息。
结合用户的话语和对话历史，给出具体的做菜建议、回答问题、或记录零散指示。

回复规则：
- 简短、直接、实用
- 使用 Markdown 格式
- 支持中英混杂（如 "加一点 dark soy sauce"）
- 如果用户提到了食材的消耗/新增或口味偏好，友好提醒：
  「✅ 已记录，点击【结束本次备餐】后我会更新到长期记忆。」
"""

SETTLE_PROMPT_TEMPLATE = """\
以下是当前的长期记忆（state.md）内容：

{state}

以下是用户本次备餐的完整对话记录：

{history}

---
任务：
基于对话历史，推断用户消耗了哪些食材、冰箱里还剩什么、是否有新的口味偏好。

请输出**全新且完整的 Markdown 文本**，直接替换原有 state 格式。
要求：
1. 保持所有二级标题（##）结构
2. 更新「冰箱库存 & 剩菜」和「动态口味偏好」两节
3. 对话中未涉及的章节保持不变
4. **只输出 Markdown 文本本身，不要有任何解释或代码块标记**
"""

SHOPPING_PROMPT_TEMPLATE = """\
以下是当前冰箱和食材库存状态：

{state}

---
任务：分析当前库存。考虑用户的半马训练需求（高蛋白、高碳水）。
如果发现关键食材（鸡蛋、肉类、绿叶菜、主食类碳水）严重不足，生成一份简洁购物清单：
- 去 Bloomington 的 Kroger 或 Fresh Thyme 买的生鲜品
- 去 Weee! 买的亚洲特色食材

如果库存充足，仅输出「无需采购」四个字，不要任何其他内容。
"""


# ════════════════════════════════════════════════════════════════════════════
# State I/O
# ════════════════════════════════════════════════════════════════════════════

def init_state() -> None:
    if not STATE_FILE.exists():
        STATE_FILE.write_text(DEFAULT_STATE, encoding="utf-8")
        logger.info("state.md initialised with default content.")


def get_state() -> str:
    init_state()
    return STATE_FILE.read_text(encoding="utf-8")


def save_state(text: str) -> None:
    STATE_FILE.write_text(text, encoding="utf-8")
    logger.info("state.md updated.")


# ════════════════════════════════════════════════════════════════════════════
# Gemini helpers (new google-genai SDK)
# ════════════════════════════════════════════════════════════════════════════

def _gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def _history_to_genai(history: list[dict]) -> list:
    """
    Convert Streamlit history  [{"role": "user"|"assistant", "content": "..."}]
    into google-genai Content  [types.Content(role=..., parts=[...])]
    """
    result = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        result.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=msg["content"])],
            )
        )
    return result


# ════════════════════════════════════════════════════════════════════════════
# Core AI functions
# ════════════════════════════════════════════════════════════════════════════

def chat_with_gemini(history: list[dict], user_msg: str):
    """
    Streaming chat with Gemini Flash — yields text chunks as they arrive.
    Use with st.write_stream() in the UI.
    Falls back to yielding an error string on failure.
    """
    try:
        client = _gemini_client()
        state = get_state()
        system_instruction = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            f"【当前长期记忆 (state.md)】\n\n{state}"
        )

        chat_session = client.chats.create(
            model=CHAT_MODEL,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            ),
            history=_history_to_genai(history),
        )

        # ── Streaming: yield chunks as they arrive ────────────────────────
        full_text = ""
        for chunk in chat_session.send_message_stream(user_msg):
            if chunk.text:
                full_text += chunk.text
                yield chunk.text

        # Log the full response for debugging
        logger.info(f"Chat response ({len(full_text)} chars)")

    except EnvironmentError as e:
        logger.error(f"Config error: {e}")
        yield f"⚠️ 配置错误：{e}"
    except Exception as e:
        logger.error(f"Gemini chat error: {e}")
        yield f"⚠️ AI 接口出错，请稍后重试。\n\n`{e}`"


def settle_state(history: list[dict]) -> str:
    """
    Post-cooking settlement: ask Gemini to rewrite state.md based on conversation.
    Falls back to original state on failure.
    """
    if not history:
        return get_state()

    try:
        client = _gemini_client()
        state = get_state()
        history_text = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in history
        )
        prompt = SETTLE_PROMPT_TEMPLATE.format(state=state, history=history_text)

        response = client.models.generate_content(
            model=SETTLE_MODEL,
            contents=prompt,
        )
        new_state = response.text.strip()

        # Sanity check: must look like Markdown
        if not new_state.startswith("#"):
            logger.warning("Gemini returned unexpected format; keeping original state.")
            return state

        return new_state

    except Exception as e:
        logger.error(f"Gemini settle_state error: {e}")
        return get_state()


def generate_shopping_list() -> str:
    """
    Analyse current inventory. Returns shopping list text or '无需采购'.
    Called by APScheduler daily job.
    """
    try:
        client = _gemini_client()
        state = get_state()
        prompt = SHOPPING_PROMPT_TEMPLATE.format(state=state)

        response = client.models.generate_content(
            model=CHAT_MODEL,   # shopping list doesn't need heavy reasoning
            contents=prompt,
        )
        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini shopping list error: {e}")
        return ""


# ════════════════════════════════════════════════════════════════════════════
# Bark (iPhone push notification)
# ════════════════════════════════════════════════════════════════════════════

def send_bark_notification(title: str, body: str) -> None:
    if not BARK_KEY:
        logger.warning("BARK_KEY not set — push notification skipped.")
        return

    encoded_title = urllib.parse.quote(title, safe="")
    encoded_body  = urllib.parse.quote(body,  safe="")
    url = f"https://api.day.app/{BARK_KEY}/{encoded_title}/{encoded_body}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        logger.info("Bark notification sent successfully.")
    except requests.exceptions.Timeout:
        logger.error("Bark notification timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Bark notification failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# APScheduler — daily 16:00 inventory check
# ════════════════════════════════════════════════════════════════════════════

_scheduler = None


def run_daily_check() -> None:
    logger.info("⏰ Daily inventory check triggered.")
    result = generate_shopping_list()
    if result and result.strip() != "无需采购":
        send_bark_notification("🛒 食材库存告警", result)
        logger.info("Shopping list pushed to iPhone via Bark.")
    else:
        logger.info("Inventory sufficient — no notification sent.")


def start_scheduler() -> None:
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    import atexit

    if _scheduler is not None and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        run_daily_check,
        trigger=CronTrigger(hour=16, minute=0),
        id="daily_shopping_check",
        replace_existing=True,
        name="Daily shopping list check",
    )
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False))
    logger.info("✅ APScheduler started — daily check at 16:00.")


# ════════════════════════════════════════════════════════════════════════════
# OpenAI Whisper — voice transcription
# NO retries on quota/auth errors to avoid burning through rate limits.
# ════════════════════════════════════════════════════════════════════════════

def _detect_audio_ext(audio_bytes: bytes) -> str:
    """
    Detect the real audio format from magic bytes.
    Browsers (Safari/Chrome) record WebM or OGG, NOT WAV.
    Whisper uses the file extension to pick a decoder, so the wrong
    extension (e.g. .wav for a WebM file) causes data corruption
    and word drops — especially past the first ~10 s.
    """
    if len(audio_bytes) >= 4:
        if audio_bytes[:4] == b'RIFF':
            return '.wav'
        if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':   # EBML / WebM / MKV
            return '.webm'
        if audio_bytes[:4] == b'OggS':
            return '.ogg'
        if len(audio_bytes) >= 8 and audio_bytes[4:8] == b'ftyp':
            return '.mp4'                              # M4A / MP4
        if audio_bytes[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
            return '.mp3'
    return '.webm'   # safe default for modern browser recordings


# Whisper prompt: context about domain and code-switching
# This dramatically improves accuracy for Chinese-English mixed speech
_WHISPER_PROMPT = (
    "以下是关于做饭、冰箱食材、菜谱的对话，可能混杂英文词汇，"
    "例如 soy sauce、dark soy sauce、instant pot、sesame oil、stir fry 等。"
)


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Transcribe audio bytes via OpenAI Whisper.
    Returns:
      str   — transcribed text on success
      None  — on recoverable/unknown error
      ""    — on permanent quota/key error (show billing link, don't retry)

    Key improvements:
      - Detects real audio format from magic bytes (fixes >10 s word-drop bug)
      - Domain prompt guides Chinese-English code-switch recognition
      - max_retries=0 stops the SDK retry loop on 429 quota errors
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — transcription skipped.")
        return None

    from openai import OpenAI, RateLimitError, AuthenticationError

    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)

    ext = _detect_audio_ext(audio_bytes)
    logger.info(f"Detected audio format: {ext} ({len(audio_bytes)} bytes)")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="zh",          # Mandarin primary; still handles English well
                prompt=_WHISPER_PROMPT, # domain context boosts accuracy
            )
        text = transcript.text.strip()
        logger.info(f"Whisper transcript ({len(text)} chars): {text!r}")
        return text if text else None

    except RateLimitError as e:
        body = str(e)
        if any(code in body for code in _NO_RETRY_CODES):
            logger.error(f"Whisper permanent quota error: {e}")
            return ""
        logger.warning(f"Whisper temporary rate limit: {e}")
        return None

    except AuthenticationError as e:
        logger.error(f"Whisper auth error: {e}")
        return ""

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return None

    finally:
        tmp_path.unlink(missing_ok=True)

