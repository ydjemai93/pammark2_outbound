import os
import json
import asyncio
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from dotenv import load_dotenv

# Configuration initiale
load_dotenv()

# Variables d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
PORT = int(os.getenv("PORT", 5050))

# Validation des variables
if not all([OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
    raise ValueError("Variables d'environnement manquantes")

# Configuration OpenAI (selon documentation)
OPENAI_MODEL = "gpt-4o-realtime-preview-2024-12-17"  # Modèle exact spécifié dans la doc
SYSTEM_INSTRUCTIONS = "Vous êtes un assistant vocal professionnel. Répondez en français de manière concise."

app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.get("/")
def health_check():
    return {"status": "active"}

@app.post("/outbound-call")
async def initiate_call(request: Request):
    data = await request.json()
    to_number = data.get("to")
    
    if not to_number or not to_number.startswith("+"):
        return JSONResponse({"error": "Format E.164 requis"}, status_code=400)

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=f"https://{request.url.hostname}/outbound-call-answered"
    )
    
    return {"status": "call_initiated", "call_sid": call.sid}

@app.post("/outbound-call-answered")
def handle_call_answer(request: Request):
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{request.url.hostname}/media-stream")
    response.append(connect)
    return HTMLResponse(str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Connexion à OpenAI avec les paramètres de la doc
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            # Configuration de session initiale
            await openai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "voice": "alloy",
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "modalities": ["text", "audio"],
                    "temperature": 0.7
                }
            }))

            stream_sid = None
            
            async def handle_twilio():
                nonlocal stream_sid
                async for msg in websocket.iter_text():
                    data = json.loads(msg)
                    
                    if data.get("event") == "start":
                        stream_sid = data["start"]["streamSid"]
                        # Déclencher la réponse initiale
                        await openai_ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "modalities": ["audio"],
                                "instructions": "Dire 'Bonjour, comment puis-je vous aider ?'"
                            }
                        }))
                    
                    elif data.get("event") == "media":
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"]
                        }))

            async def handle_openai():
                async for msg in openai_ws:
                    data = json.loads(msg)
                    
                    if data.get("type") == "response.audio.delta":
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": data["delta"]}
                        })

            await asyncio.gather(handle_twilio(), handle_openai())

    except Exception as e:
        print(f"Erreur: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
