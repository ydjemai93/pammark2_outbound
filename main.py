import os
import json
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from dotenv import load_dotenv

# 1. Chargement des variables d'environnement
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')
PORT = int(os.getenv('PORT', 5050))

# Configuration système
SYSTEM_MESSAGE = "You are a helpful and bubbly AI assistant..."
VOICE = "alloy"
LOG_EVENT_TYPES = ['error', 'response.done', 'session.created']
OPENAI_MODEL = "gpt-4o"  # Modèle réellement disponible

app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Helpers
async def initialize_session(openai_ws):
    """Initialise la session OpenAI Realtime"""
    await openai_ws.send(json.dumps({
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
    }))

async def send_mark(websocket, stream_sid):
    """Envoie un mark pour synchroniser Twilio"""
    await websocket.send_json({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "response-part"}
    })

# Endpoints
@app.get("/")
async def health_check():
    return {"status": "OK"}

@app.post("/outbound-call")
async def handle_outbound_call(request: Request):
    data = await request.json()
    
    # Validation du numéro (doit être formaté E.164 par Xano)
    to_number = data.get("to")
    if not to_number or not to_number.startswith("+"):
        return JSONResponse(
            {"error": "Numéro invalide. Format E.164 requis (ex: +33123456789)"}, 
            status_code=400
        )

    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_FROM_NUMBER,
            url=f"https://{request.url.hostname}/outbound-call-answered"
        )
        return {"status": "Call initiated", "call_sid": call.sid}
    
    except Exception as e:
        return JSONResponse(
            {"error": f"Twilio error: {str(e)}"}, 
            status_code=500
        )

@app.post("/outbound-call-answered")
async def handle_outbound_call_answered(request: Request):
    """Génère le TwiML de connexion au WebSocket"""
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{request.url.hostname}/media-stream")
    response.append(connect)
    return HTMLResponse(str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    
    try:
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            ping_timeout=30,
            close_timeout=15
        ) as openai_ws:
            
            await initialize_session(openai_ws)

            async def handle_twilio_messages():
                nonlocal stream_sid
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    
                    if data.get("event") == "start":
                        stream_sid = data["start"]["streamSid"]
                    
                    elif data.get("event") == "media":
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"]  # Payload direct
                        }))

            async def handle_openai_messages():
                async for message in openai_ws:
                    resp = json.loads(message)
                    
                    if resp.get("type") == "response.audio.delta" and "delta" in resp:
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": resp["delta"]}  # Pas de re-encodage
                        })
                        
                        if stream_sid:
                            await send_mark(websocket, stream_sid)

            await asyncio.gather(
                handle_twilio_messages(),
                handle_openai_messages()
            )

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connexion fermée: code={e.code}, raison={e.reason}")
    except Exception as e:
        print(f"Erreur critique: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
