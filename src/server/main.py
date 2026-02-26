"""
Voice Control Server

WebSocket server that handles:
- Audio input from browser
- Speech-to-Text via Whisper
- AI backend communication
- Text-to-Speech via ElevenLabs
- Audio streaming back to browser
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Optional

# Load .env into os.environ BEFORE anything reads env vars
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import clean_for_speech
from .audio_processing import preprocess as preprocess_audio


class Settings(BaseSettings):
    """Server configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Auth
    require_auth: bool = False  # Set True for production
    master_key: Optional[str] = None  # Admin key for full access
    
    # STT
    stt_model: str = "base"  # tiny, base, small, medium, large-v3-turbo
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    
    # TTS
    tts_model: str = "chatterbox"
    tts_voice: Optional[str] = None  # Path to voice sample for cloning
    elevenlabs_api_key: Optional[str] = None
    
    # AI Backend
    backend_type: str = "openai"  # openai, openclaw, custom
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # OpenClaw Gateway (auto-detected from OPENCLAW_GATEWAY_URL + TOKEN)
    gateway_url: Optional[str] = None
    gateway_token: Optional[str] = None
    
    # Audio
    sample_rate: int = 16000
    
    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"


settings = Settings()
app = FastAPI(title="Voice Control", version="0.1.0")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, tts, backend, vad
    
    logger.info("Initializing Voice Control server...")
    
    # Load API keys
    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")
    
    # Initialize STT
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
    )
    
    # Initialize TTS
    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = ChatterboxTTS(
        voice_sample=settings.tts_voice,
    )
    
    # Initialize AI backend
    # Auto-detect OpenClaw gateway
    gateway_url = settings.gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    
    if gateway_url and gateway_token:
        # Use OpenClaw gateway (connects to Aria!)
        # Route to voice agent — Sonnet, shared workspace, orchestration tools only
        voice_agent = os.getenv("OPENCLAW_VOICE_AGENT", "voice")
        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url} (agent: {voice_agent})")
        # Load voice system prompt from file (keeps personalization out of source code)
        # Check: env var path → ./voice-prompt.txt → generic fallback
        voice_prompt_path = os.getenv("OPENCLAW_VOICE_PROMPT_FILE", "voice-prompt.txt")
        if os.path.exists(voice_prompt_path):
            with open(voice_prompt_path) as f:
                voice_system_prompt = f.read().strip()
            logger.info(f"Loaded voice prompt from {voice_prompt_path}")
        else:
            logger.warning("No voice-prompt.txt found — using generic default. Create voice-prompt.txt to customize.")
            voice_system_prompt = (
                "This conversation is happening via real-time voice chat. "
                "Keep responses concise and conversational — 2-5 sentences for most replies. "
                "You are speaking out loud, so use plain speech only. "
                "No markdown, bullet points, code blocks, or visual formatting. "
                "You can use tools when needed. Summarize results conversationally. "
                "Be direct and helpful."
            )
        
        backend = AIBackend(
            backend_type="openai",  # Gateway speaks OpenAI API
            url=f"{gateway_url}/v1",
            model=f"openclaw:{voice_agent}",
            api_key=gateway_token,
            max_tokens=600,
            system_prompt=voice_system_prompt,
        )
    else:
        # Fallback to direct OpenAI
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    # Initialize VAD
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()
    
    logger.info("✅ Voice Control server ready!")


@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse("src/client/index.html")


@app.get("/audio-processor.js")
@app.get("/voice/audio-processor.js")
async def audio_worklet():
    """Serve the AudioWorklet processor (must be same-origin)."""
    return FileResponse(
        "src/client/audio-processor.js",
        media_type="application/javascript",
    )


@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
):
    """
    Create a new API key (requires master key).
    
    curl -X POST "http://localhost:8765/api/keys?name=myapp&tier=pro" \
         -H "x-master-key: YOUR_MASTER_KEY"
    """
    # Verify master key
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}
        
        provided_key = master_key or ""
        if provided_key != settings.master_key:
            # Also check if it's a valid master-tier key
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}
    
    from .auth import PRICING_TIERS
    
    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    
    tier_config = PRICING_TIERS[tier]
    
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    
    return {
        "api_key": plaintext_key,  # Only shown once!
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@app.get("/api/usage")
async def get_usage(api_key: str):
    """
    Get usage stats for an API key.
    
    curl "http://localhost:8765/api/usage?api_key=ocv_xxx"
    """
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    
    return token_manager.get_usage(key)


@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    # Check for API key in query params or headers
    api_key_str = websocket.query_params.get("api_key") or \
                  websocket.headers.get("x-api-key")
    
    api_key: Optional[APIKey] = None
    
    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        
        if not token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        # Dev mode - allow all
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")
    
    await websocket.accept()
    
    audio_buffer = []
    is_listening = False
    session_start = None
    
    async def stream_ai_response(user_text: str):
        """Stream AI response with progressive TTS for a given user message."""
        logger.debug("Streaming AI response...")
        
        full_response = ""
        sentence_buffer = ""
        
        # Stream response and synthesize sentences as they complete
        async for chunk in backend.chat_stream(user_text):
            full_response += chunk
            sentence_buffer += chunk
            
            # Send text chunk for progressive display
            await websocket.send_json({
                "type": "response_chunk",
                "text": chunk,
            })
            
            # Check for sentence boundaries
            while any(sep in sentence_buffer for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                # Find first sentence boundary
                earliest_idx = len(sentence_buffer)
                for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    idx = sentence_buffer.find(sep)
                    if idx != -1 and idx < earliest_idx:
                        earliest_idx = idx + len(sep)
                
                if earliest_idx < len(sentence_buffer):
                    sentence = sentence_buffer[:earliest_idx].strip()
                    sentence_buffer = sentence_buffer[earliest_idx:]
                    
                    if sentence:
                        # Clean and synthesize this sentence
                        speech_text = clean_for_speech(sentence)
                        if speech_text:
                            logger.debug(f"Synthesizing: {speech_text[:50]}...")
                            async for audio_chunk in tts.synthesize_stream(speech_text):
                                audio_b64 = base64.b64encode(audio_chunk).decode()
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "data": audio_b64,
                                    "format": "mp3",
                                })
                else:
                    break
        
        # Handle any remaining text
        if sentence_buffer.strip():
            speech_text = clean_for_speech(sentence_buffer.strip())
            if speech_text:
                async for audio_chunk in tts.synthesize_stream(speech_text):
                    audio_b64 = base64.b64encode(audio_chunk).decode()
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "format": "mp3",
                    })
        
        # Signal end of response
        await websocket.send_json({
            "type": "response_complete",
            "text": full_response,
        })
        logger.info(f"Response complete: {full_response[:100]}...")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client: {data[:100]}")
                continue
            
            msg_type = msg.get("type")
            if not msg_type:
                logger.warning(f"Message missing 'type': {str(msg)[:100]}")
                continue
            
            try:
                if msg_type == "start_listening":
                    is_listening = True
                    audio_buffer = []
                    await websocket.send_json({"type": "listening_started"})
                    logger.debug("Started listening")
                    
                elif msg_type == "stop_listening":
                    is_listening = False
                    
                    if audio_buffer:
                        # Combine audio chunks
                        audio_data = np.concatenate(audio_buffer)
                        
                        # Preprocess: high-pass filter → noise reduction → trim → normalize
                        loop = asyncio.get_event_loop()
                        audio_data = await loop.run_in_executor(None, preprocess_audio, audio_data)
                        
                        # Transcribe
                        logger.debug("Transcribing audio...")
                        transcript = await stt.transcribe(audio_data)
                        
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript,
                            "final": True,
                        })
                        logger.info(f"Transcript: {transcript}")
                        
                        if transcript.strip() and len(transcript.strip()) >= 2:
                            await stream_ai_response(transcript)
                    
                    audio_buffer = []
                    await websocket.send_json({"type": "listening_stopped"})
                    logger.debug("Stopped listening")
                
                elif msg_type == "text_message":
                    # Text input from the UI — skip STT, go straight to AI
                    user_text = (msg.get("text") or "").strip()
                    if user_text:
                        logger.info(f"Text message: {user_text}")
                        await stream_ai_response(user_text)
                    
                elif msg_type == "new_session":
                    # Clear conversation history for a fresh start
                    if backend:
                        backend.clear_history()
                    logger.info("Session reset by client")
                
                elif msg_type == "context_restore":
                    # Client is sending conversation history for context recovery
                    # (e.g. after server restart or reconnect)
                    messages = msg.get("messages", [])
                    if backend and messages:
                        backend.clear_history()
                        restored = 0
                        for m in messages[-20:]:  # Cap at last 20 turns
                            role = m.get("role")
                            text = m.get("text", "").strip()
                            if role in ("user", "assistant") and text:
                                backend.conversation_history.append({
                                    "role": role,
                                    "content": text,
                                })
                                restored += 1
                        logger.info(f"Context restored: {restored} messages from client")
                        await websocket.send_json({
                            "type": "context_restored",
                            "count": restored,
                        })
                    
                elif msg_type == "cancel_listening":
                    # Client detected noise/too-short — discard without processing
                    is_listening = False
                    audio_buffer = []
                    logger.debug("Listening cancelled (noise/too short)")
                    
                elif msg_type == "audio" and is_listening:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(msg.get("data", ""))
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_buffer.append(audio_np)
                    
                    # VAD check - notify client if speech detected
                    if vad and len(audio_np) > 0:
                        has_speech = vad.is_speech(audio_np)
                        await websocket.send_json({
                            "type": "vad_status",
                            "speech_detected": has_speech,
                        })
                    
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
                else:
                    logger.debug(f"Unknown message type: {msg_type}")
                    
            except Exception as e:
                logger.error(f"Error handling '{msg_type}': {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Server error processing {msg_type}",
                })
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# Serve static files for client
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
