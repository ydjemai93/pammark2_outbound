"""
main_outbound.py - pam_markII (version Outbound)

Ce script Python gère un assistant vocal en temps réel avec Twilio (Media Streams) et l'API Realtime d'OpenAI (GPT-4o), 
mais cette fois pour APPELS SORTANTS.

Workflow:
  1) On appelle /outbound-call avec {"to": "+33123456789"} => Twilio compose un appel vers 'to'.
  2) Quand la personne décroche, Twilio fait un POST sur /outbound-call-answered.
  3) /outbound-call-answered renvoie un TwiML <Connect><Stream track="both"> vers wss://<host>/media-stream
  4) Twilio ouvre la WebSocket -> /media-stream
  5) /media-stream relie l'audio en temps réel à l'API OpenAI Realtime => STT -> GPT -> TTS
"""

import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect

from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client
from dotenv import load_dotenv

# 1) Chargement des variables d'environnement
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')
PORT = int(os.getenv('PORT', 5050))

# Vérifications
if not OPENAI_API_KEY:
    raise ValueError('Missing OPENAI_API_KEY.')
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError('Missing Twilio credentials (SID / Auth Token).')
if not TWILIO_FROM_NUMBER:
    raise ValueError('Missing TWILIO_FROM_NUMBER.')

# Instanciation Twilio pour lancer l'appel
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Message système (prompt "system") pour initier la personnalité de l'IA
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
)

# Nom de la voix pour l'API Realtime
VOICE = 'alloy'

# Liste d'événements Realtime qu'on veut logger
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

# Flag éventuel d'affichage du timing
SHOW_TIMING_MATH = False

# Création de l'application FastAPI
app = FastAPI()

@app.get("/", response_class=JSONResponse)
async def index_page():
    """
    Endpoint GET pour vérifier que pam_markII tourne bien.
    """
    return {"message": "pam_markII (Outbound) is running!"}


@app.post("/outbound-call")
async def handle_outbound_call(request: Request):
    """
    1) On reçoit un JSON {"to": "+33xxxxxx"}
    2) On déclenche un appel Twilio => to
    3) Twilio appelle /outbound-call-answered quand la personne décroche
    """
    data = await request.json()
    to_number = data.get("to")
    if not to_number:
        return JSONResponse({"error": "Missing 'to' phone number."}, status_code=400)

    # Création de l'appel via Twilio
    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=f"https://{request.url.hostname}/outbound-call-answered"
    )

    return JSONResponse({
        "message": "Outbound call initiated.",
        "call_sid": call.sid
    })


@app.api_route("/outbound-call-answered", methods=["GET", "POST"])
async def handle_outbound_call_answered(request: Request):
    """
    Twilio POST ici quand l'appel sortant est décroché.
    On renvoie un TwiML <Connect><Stream track="both"> => wss://<host>/media-stream
    """
    host = request.url.hostname

    # Construction du TwiML
    response = VoiceResponse()
    connect = Connect()
    # track="both" => audio bidirectionnel
    connect.stream(url=f"wss://{host}/media-stream", track="both")
    response.append(connect)

    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """
    WebSocket /media-stream : Twilio envoie l'audio G711 ulaw (base64).
    On relaye l'audio vers l'API OpenAI Realtime (GPT-4o).
    Le pipeline:
      Twilio -> pam_markII (Outbound) -> OpenAI Realtime -> pam_markII -> Twilio
    """
    print("Client connected (Twilio side) - pam_markII outbound media-stream")

    # Accepte la connexion WS Twilio
    await websocket.accept()

    # Connexion simultanée avec OpenAI Realtime
    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        # Initialise la session (instructions, voix, etc.)
        await initialize_session(openai_ws)

        # Variables de contexte
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None

        async def receive_from_twilio():
            """
            Reçoit l'audio & events Twilio (start, media, mark)
            => envoie l'audio inbound à OpenAI Realtime
            """
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    event = data.get('event')

                    if event == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Outgoing stream started: {stream_sid}")
                        latest_media_timestamp = 0
                        response_start_timestamp_twilio = None
                        last_assistant_item = None

                    elif event == 'media':
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        # Forward l'audio Twilio => OpenAI
                        if openai_ws.open:
                            await openai_ws.send(json.dumps(audio_append))

                    elif event == 'mark':
                        # Twilio a joué un 'mark'
                        if mark_queue:
                            mark_queue.pop(0)

            except WebSocketDisconnect:
                print("Client disconnected (Twilio).")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """
            Reçoit les événements depuis OpenAI Realtime (e.g. TTS),
            => on renvoie l'audio TTS vers Twilio, base64 G711.
            """
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    # Log éventuels
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    # Cas: audio TTS
                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        # On re-base64 G711
                        audio_payload = base64.b64encode(
                            base64.b64decode(response['delta'])
                        ).decode('utf-8')

                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_payload}
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp

                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Interruption si user parle
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started from user => interrupt TTS.")
                        if last_assistant_item:
                            await handle_speech_started_event()

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_started_event():
            """
            Envoie un event conversation.item.truncate si l'utilisateur reparle 
            => coupe le TTS en cours
            """
            nonlocal response_start_timestamp_twilio, last_assistant_item
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Truncation at: {elapsed_time}ms")

                if last_assistant_item:
                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                # Demande à Twilio de flush le buffer TTS
                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None

        async def send_mark(connection, sid):
            """
            Envoie un 'mark' à Twilio pour séparer les blocs TTS.
            """
            if sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        # Exécution simultanée
        await asyncio.gather(receive_from_twilio(), send_to_twilio())


async def initialize_session(openai_ws):
    """
    Initialise la session Realtime (voix, instructions, etc.)
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
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Pour faire parler l'IA en premier, décommentez l'appel:
    # await send_initial_conversation_item(openai_ws)


async def send_initial_conversation_item(openai_ws):
    """
    Permet à l'IA de parler immédiatement. Appelé dans initialize_session() si besoin.
    """
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Hello, you are connected to an AI voice assistant. Ask me anything!"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    # On déclenche la réponse:
    await openai_ws.send(json.dumps({"type": "response.create"}))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
