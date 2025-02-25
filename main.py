import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.websockets import WebSocketDisconnect

from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client
from dotenv import load_dotenv

# 1) Charger les variables d'environnement
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
PORT = int(os.getenv("PORT", 5050))

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("Missing Twilio credentials")
if not TWILIO_FROM_NUMBER:
    raise ValueError("Missing TWILIO_FROM_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Prompt "system" pour modeler la personnalité GPT
SYSTEM_MESSAGE = (
    "You are a helpful, friendly AI assistant. "
    "Respond to the user in a pleasant tone, with short, clear, spoken phrases."
)

# On utilise la voix "alloy" côté Realtime
VOICE = "alloy"

# Événements Realtime qu’on log pour debug
LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
]

app = FastAPI()

@app.get("/")
def health():
    return {"message": "Outbound call API is running!"}

# 2) Endpoint: POST /outbound-call
@app.post("/outbound-call")
async def outbound_call(request: Request):
    """
    Reçoit { "to": "+33..." }, appelle Twilio (outbound).
    Twilio appelle /outbound-call-answered quand le destinataire décroche.
    """
    body = await request.json()
    to_number = body.get("to")
    if not to_number:
        return JSONResponse({"error": "Missing 'to' phone number."}, status_code=400)

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=f"https://{request.url.hostname}/outbound-call-answered"
    )

    return JSONResponse({
        "message": "Outbound call initiated.",
        "call_sid": call.sid
    })

# 3) Endpoint: /outbound-call-answered => Twilio arrive ici au décroché
@app.post("/outbound-call-answered")
def outbound_call_answered(request: Request):
    """
    Renvoie un TwiML <Connect><Stream track="both">, sans 'Say'.
    L'accueil vocal sera prononcé par l'IA via /media-stream
    """
    response = VoiceResponse()
    connect = Connect()

    # track="both" => audio bidirectionnel
    host = request.url.hostname
    connect.stream(url=f"wss://{host}/media-stream", track="both")
    response.append(connect)

    return HTMLResponse(str(response), media_type="application/xml")

# 4) WebSocket /media-stream => Twilio <-> OpenAI Realtime
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio envoie l'audio G711 inbound => On forward vers OpenAI
    On reçoit TTS => On renvoie à Twilio
    """
    print("[/media-stream] Twilio connected.")
    await websocket.accept()

    try:
        # Connexion WS à OpenAI Realtime
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            # On initialise la session
            await initialize_session(openai_ws)

            # Variables de contexte
            stream_sid = None
            latest_media_timestamp = 0
            last_assistant_item = None
            mark_queue = []
            response_start_timestamp = None

            async def receive_from_twilio():
                """
                Gère les events inbound (start, media, mark).
                Sur 'start', on envoie un message d'accueil via l'IA.
                """
                nonlocal stream_sid, latest_media_timestamp

                try:
                    async for msg in websocket.iter_text():
                        data = json.loads(msg)
                        event_type = data.get("event")

                        if event_type == "start":
                            stream_sid = data["start"]["streamSid"]
                            print(f"Call started, streamSid={stream_sid}")

                            # => Envoyer le message d'accueil à l'IA
                            await send_welcome_message(openai_ws)

                        elif event_type == "media":
                            latest_media_timestamp = int(data["media"]["timestamp"])
                            audio_payload = data["media"]["payload"]
                            # forward audio to OpenAI
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_payload
                            }))

                        elif event_type == "mark":
                            if mark_queue:
                                mark_queue.pop(0)

                except WebSocketDisconnect:
                    print("Twilio WebSocket disconnected.")
                    if openai_ws.open:
                        await openai_ws.close()

            async def send_to_twilio():
                """
                Reçoit events d'OpenAI Realtime
                => envoie TTS à Twilio
                => gère l'interruption si l'utilisateur reparle
                """
                nonlocal response_start_timestamp, last_assistant_item
                try:
                    async for msg in openai_ws:
                        resp = json.loads(msg)

                        # Log
                        if resp.get("type") in LOG_EVENT_TYPES:
                            print("OpenAI event:", resp["type"], resp)

                        # 1) TTS
                        if resp.get("type") == "response.audio.delta" and "delta" in resp:
                            # On ré-encode en base64 G711
                            tts_base64 = base64.b64encode(
                                base64.b64decode(resp["delta"])
                            ).decode("utf-8")

                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": tts_base64}
                            })

                            if response_start_timestamp is None:
                                response_start_timestamp = latest_media_timestamp

                            if resp.get("item_id"):
                                last_assistant_item = resp["item_id"]

                            await send_mark(websocket, stream_sid)

                        # 2) L'utilisateur reparle => interruption TTS
                        if resp.get("type") == "input_audio_buffer.speech_started":
                            print("Speech started => interrupt TTS.")
                            await handle_user_interrupt()

                except Exception as e:
                    print("Error in send_to_twilio:", e)

            async def handle_user_interrupt():
                """
                Couper la lecture TTS quand l'utilisateur prend la parole
                """
                nonlocal response_start_timestamp, last_assistant_item
                if mark_queue and response_start_timestamp is not None:
                    elapsed = latest_media_timestamp - response_start_timestamp
                    if last_assistant_item:
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.truncate",
                            "item_id": last_assistant_item,
                            "content_index": 0,
                            "audio_end_ms": elapsed
                        }))

                    await websocket.send_json({
                        "event": "clear",
                        "streamSid": stream_sid
                    })

                    mark_queue.clear()
                    last_assistant_item = None
                    response_start_timestamp = None

            async def send_mark(ws_conn, sid):
                """
                Envoie un marqueur TTS à Twilio
                """
                if sid:
                    mark_event = {
                        "event": "mark",
                        "streamSid": sid,
                        "mark": {"name": "tts-chunk"}
                    }
                    await ws_conn.send_json(mark_event)
                    mark_queue.append("tts-chunk")

            # Lancer les deux coroutines en parallèle
            await asyncio.gather(receive_from_twilio(), send_to_twilio())

    except Exception as e:
        print("Error in /media-stream main block:", e)


async def initialize_session(openai_ws):
    """
    Configure la session Realtime
    (audio g711, param. GPT, etc.)
    """
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8
        }
    }
    await openai_ws.send(json.dumps(session_update))


async def send_welcome_message(openai_ws):
    """
    Envoie un message initial (IA TTS) quand l'appel a démarré côté Twilio.
    (ex: "Bonjour, je suis votre assistant virtuel…")
    """
    welcome_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Bonjour, je suis votre assistant IA. N'hésite pas à parler, je t'écoute !"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(welcome_item))
    # Demande à l'IA de générer une réponse
    await openai_ws.send(json.dumps({"type": "response.create"}))


# 5) Lancement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
