import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from twilio.rest import Client
from dotenv import load_dotenv

# 1. Chargement des variables d'environnement
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')  # Votre numéro Twilio
PORT = int(os.getenv('PORT', 5050))

# Prompt "system" qui initialise le comportement GPT
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant ... "
    "always stay positive, add jokes when appropriate..."
)

# Voix configurée pour la restitution TTS par l’API Realtime
VOICE = "alloy"

# Log d'événements Realtime qu'on veut voir passer
LOG_EVENT_TYPES = [
    'error',
    'response.content.done',
    'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created'
]

# Création de l'app FastAPI
app = FastAPI()

# On vérifie qu'on a bien toutes les clés nécessaires
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("Missing Twilio credentials.")
if not TWILIO_FROM_NUMBER:
    raise ValueError("Missing TWILIO_FROM_NUMBER environment variable.")

# Instancier le client Twilio
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.get("/")
async def health_check():
    """
    Juste pour vérifier que l'app tourne.
    """
    return {"message": "Outbound calls are available."}

@app.post("/outbound-call")
async def handle_outbound_call(request: Request):
    """
    Endpoint pour initier un appel sortant. 
    JSON attendu: { "to": "+3312345678" } par ex.
    """
    data = await request.json()
    to_number = data.get("to")
    if not to_number:
        return JSONResponse({"error": "Missing 'to' phone number."}, status_code=400)

    # Création de l'appel Twilio
    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        # Twilio appellera cette URL quand le destinataire décroche
        url=f"https://{request.url.hostname}/outbound-call-answered"
    )
    return JSONResponse({"message": "Outbound call initiated.", "call_sid": call.sid})

@app.post("/outbound-call-answered")
async def handle_outbound_call_answered(request: Request):
    """
    Twilio POST ici quand la personne décroche.
    On renvoie un TwiML <Connect><Stream> pour relier l'audio au /media-stream.
    """
    response = VoiceResponse()
    response.say("Hello, you're connected to our A.I. assistant. Please talk now!")

    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)

    return HTMLResponse(str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    WebSocket Twilio <-> OpenAI Realtime.
    Transmet l'audio G711 (ulaw) vers OpenAI, et réinjecte l'audio TTS en sens inverse.
    """
    print("Twilio has connected to /media-stream (Outbound).")
    await websocket.accept()

    try:
        async with websockets.connect(
            # L’endpoint Realtime GPT (exemple GPT-4o):
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            # Initialisation session Realtime
            await initialize_session(openai_ws)

            stream_sid = None
            latest_media_timestamp = 0
            last_assistant_item = None
            mark_queue = []
            response_start_timestamp = None

            # Coroutine 1: Ecoute Twilio et forward vers OpenAI
            async def receive_from_twilio():
                nonlocal stream_sid, latest_media_timestamp
                try:
                    async for msg in websocket.iter_text():
                        data = json.loads(msg)

                        if data.get("event") == "start":
                            stream_sid = data["start"]["streamSid"]
                            print(f"Outbound call started, streamSid={stream_sid}")

                        elif data.get("event") == "media":
                            # Audio G711 en base64
                            latest_media_timestamp = int(data["media"]["timestamp"])
                            audio_payload = data["media"]["payload"]
                            # On envoie à openai_ws
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_payload
                            }))

                        elif data.get("event") == "mark":
                            # Twilio nous notifie qu'un mark a été joué
                            if mark_queue:
                                mark_queue.pop(0)

                except WebSocketDisconnect:
                    print("Twilio disconnected.")
                    if openai_ws.open:
                        await openai_ws.close()

            # Coroutine 2: Ecoute OpenAI => forward audio TTS => Twilio
            async def send_to_twilio():
                nonlocal response_start_timestamp, last_assistant_item
                try:
                    async for message in openai_ws:
                        resp = json.loads(message)

                        # Log certains types d'événements
                        if resp.get("type") in LOG_EVENT_TYPES:
                            print("OpenAI event:", resp["type"], resp)

                        # On reçoit de l'audio du bot
                        if resp.get("type") == "response.audio.delta" and "delta" in resp:
                            # C'est un block audio TTS, base64 G711
                            audio_base64 = base64.b64encode(
                                base64.b64decode(resp["delta"])
                            ).decode("utf-8")

                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": { "payload": audio_base64 }
                            })

                            if response_start_timestamp is None:
                                response_start_timestamp = latest_media_timestamp

                            if resp.get("item_id"):
                                last_assistant_item = resp["item_id"]

                            # On envoie un 'mark' à Twilio pour séparer segments TTS
                            await send_mark(websocket, stream_sid)

                        # Event: user se remet à parler => interruption TTS
                        if resp.get("type") == "input_audio_buffer.speech_started":
                            print("User speech started => interrupt TTS.")
                            await handle_user_interrupt()

                except Exception as e:
                    print("Error send_to_twilio:", e)

            async def handle_user_interrupt():
                """
                Couper la lecture TTS si l'utilisateur reparle.
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

                    # Dire à Twilio de nettoyer le buffer TTS
                    await websocket.send_json({
                        "event": "clear",
                        "streamSid": stream_sid
                    })

                    mark_queue.clear()
                    last_assistant_item = None
                    response_start_timestamp = None

            async def send_mark(ws_conn, sid):
                if sid:
                    mark_event = {
                        "event": "mark",
                        "streamSid": sid,
                        "mark": { "name": "response-part" }
                    }
                    await ws_conn.send_json(mark_event)
                    mark_queue.append("response-part")

            # Lancement des 2 coroutines en parallèle
            await asyncio.gather(receive_from_twilio(), send_to_twilio())

    except Exception as e:
        print("Error in /media-stream main block:", e)

async def initialize_session(openai_ws):
    """
    Configure la session OpenAI Realtime, format audio, prompt system, etc.
    """
    session_msg = {
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
    print("Initializing session =>", session_msg)
    await openai_ws.send(json.dumps(session_msg))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
