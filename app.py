"""
app.py
──────
Streamlit front-end for the Personal AI Chef & Pantry Agent.

Tabs:
  🧑‍🍳 厨房助理  — chat + voice input + settlement
  🧠 大脑记忆   — state.md editor
"""

import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="ByteChef · 私人主厨",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS: premium dark design ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Remove default Streamlit padding ── */
.main .block-container { padding: 0.5rem 1rem 4rem; max-width: 720px; }

/* ── Header gradient banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0a00 0%, #2d1500 40%, #1a0a00 100%);
    border: 1px solid rgba(255,107,53,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.hero-emoji { font-size: 2.5rem; }
.hero-title { font-size: 1.6rem; font-weight: 700; color: #ff8c5a; margin: 0; }
.hero-sub   { font-size: 0.8rem; color: #888; margin: 0; letter-spacing: 0.05em; }

/* ── Tab style override ── */
button[data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 10px 10px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff8c5a !important;
    border-bottom: 3px solid #ff6b35 !important;
}

/* ── Chat message bubbles ── */
div[data-testid="stChatMessage"] {
    border-radius: 14px;
    margin-bottom: 0.5rem;
    padding: 0.3rem 0.5rem;
}
/* User bubble: right-tinted */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(255, 107, 53, 0.07);
    border: 1px solid rgba(255, 107, 53, 0.18);
}
/* Assistant bubble: subtle glow */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255,255,255, 0.03);
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── Settlement button ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #ff6b35, #f7931e) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    box-shadow: 0 0 18px rgba(255,107,53,0.35) !important;
    transition: box-shadow 0.2s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(255,107,53,0.6) !important;
}

/* ── Secondary buttons ── */
div[data-testid="stButton"] button[kind="secondary"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.05) !important;
    color: #ccc !important;
    width: 100% !important;
}

/* ── Voice input widget ── */
div[data-testid="stAudioInput"] {
    background: linear-gradient(135deg, rgba(255,107,53,0.08), rgba(247,147,30,0.05));
    border: 1px solid rgba(255,107,53,0.25);
    border-radius: 14px;
    padding: 0.8rem 1rem;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255,107,53,0.3) !important;
    background: rgba(255,255,255,0.04) !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.2) !important;
}

/* ── Text area (memory editor) ── */
textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 10px !important;
}

/* ── Info / success toasts ── */
div[data-testid="stAlert"] { border-radius: 10px; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; margin: 0.8rem 0 !important; }

/* ── Voice section label ── */
.voice-label {
    font-size: 0.78rem;
    color: #888;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Backend imports (after page config) ─────────────────────────────────────
from agent_logic import (
    init_state,
    get_state,
    save_state,
    chat_with_gemini,
    settle_state,
    transcribe_audio,
    start_scheduler,
    DEFAULT_STATE,
)

# ════════════════════════════════════════════════════════════════════════════
# One-time server initialisation (runs once per process, not per rerun)
# ════════════════════════════════════════════════════════════════════════════
if "app_initialized" not in st.session_state:
    init_state()
    start_scheduler()
    st.session_state["app_initialized"] = True

# ── Short-term memory ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# Track last processed audio to prevent reprocessing on every rerun
if "last_audio_id" not in st.session_state:
    st.session_state["last_audio_id"] = None

# ════════════════════════════════════════════════════════════════════════════
# Hero banner
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-emoji">🍳</div>
  <div>
    <p class="hero-title">ByteChef</p>
    <p class="hero-sub">PERSONAL AI CHEF &amp; PANTRY AGENT</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════
tab_chat, tab_memory = st.tabs(["🧑‍🍳 厨房助理", "🧠 大脑记忆"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Kitchen Assistant
# ────────────────────────────────────────────────────────────────────────────
with tab_chat:

    # ── Chat history ─────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            "<p style='color:#555; text-align:center; padding:2rem 0; font-size:0.9rem;'>"
            "👋 跟我说说今天想吃什么，或者冰箱里有什么食材？"
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    st.divider()

    # ── Voice input (st.audio_input — Streamlit built-in, no extra package) ──
    st.markdown('<p class="voice-label">🎙 语音输入 &nbsp;·&nbsp; 支持中英混杂</p>',
                unsafe_allow_html=True)

    audio_value = st.audio_input(
        label="按住录音，松开发送",
        label_visibility="collapsed",
        key="voice_input",
    )

    # Process audio only when a NEW recording is detected.
    # Use hash(bytes) — NOT id(object) — because Streamlit creates a new
    # Python object on every rerun even for the same audio blob.
    transcribed_text = ""
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hash(audio_bytes)
        if audio_hash != st.session_state["last_audio_id"]:
            st.session_state["last_audio_id"] = audio_hash
            with st.spinner("🎙️ 正在识别语音…"):
                result = transcribe_audio(audio_bytes)

            if result is None:
                st.warning("⚠️ 识别失败，请重试或直接文字输入。")
            elif result == "":
                st.error(
                    "**OpenAI 余额不足，语音识别已停用。**\n\n"
                    "请前往 [platform.openai.com/billing](https://platform.openai.com/billing) "
                    "充值后刷新。\n\n*文字聊天完全正常，不受影响。*"
                )
            else:
                transcribed_text = result
                st.success(f"📝 **识别结果：** {transcribed_text}")


    # ── Text input ───────────────────────────────────────────────────────
    user_input = st.chat_input(
        placeholder="用文字提问，或录音后自动填入…"
    )

    # Determine final message: transcribed voice takes priority
    final_input = transcribed_text if transcribed_text else user_input

    if final_input and final_input.strip():
        with st.chat_message("user"):
            st.markdown(final_input)
        st.session_state.messages.append({"role": "user", "content": final_input})

        with st.chat_message("assistant"):
            # stream=True: tokens appear as they arrive (no more spinner wait)
            reply_chunks = chat_with_gemini(
                st.session_state.messages[:-1],
                final_input,
            )
            reply = st.write_stream(reply_chunks)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    st.divider()

    # ── Settlement button ─────────────────────────────────────────────────
    if st.button("🍳 结束本次备餐 · 更新记忆", type="primary", key="btn_settle"):
        if not st.session_state.messages:
            st.toast("本次没有对话记录，无需结算。", icon="ℹ️")
        else:
            with st.spinner("🧠 AI 正在整理食材消耗与口味记录…"):
                new_state = settle_state(st.session_state.messages)
                save_state(new_state)
                st.session_state.messages = []
                st.session_state["last_audio_id"] = None
            st.success("✅ 长期记忆已更新！开启新的备餐。")
            st.rerun()

    # ── Session stats ─────────────────────────────────────────────────────
    if st.session_state.messages:
        n = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.caption(f"本次对话 {n} 条消息 · 点击上方按钮结算并清空")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Memory Editor
# ────────────────────────────────────────────────────────────────────────────
with tab_memory:
    st.markdown("### 🧠 长期记忆编辑器")
    st.caption("直接编辑 `state.md`。AI 下次对话时自动读取最新版本。")

    current_state = get_state()
    edited = st.text_area(
        label="state_editor",
        value=current_state,
        height=520,
        label_visibility="collapsed",
        key="memory_editor",
    )

    col_save, col_reset = st.columns([3, 1])
    with col_save:
        if st.button("💾 保存修改", key="btn_save", use_container_width=True):
            if edited.strip():
                save_state(edited)
                st.success("✅ state.md 已保存！")
                st.rerun()
            else:
                st.error("内容不能为空，操作取消。")

    with col_reset:
        if st.button("🔄 重置", key="btn_reset", use_container_width=True):
            save_state(DEFAULT_STATE)
            st.success("已重置")
            st.rerun()

    with st.expander("📖 预览渲染效果"):
        st.markdown(current_state)
