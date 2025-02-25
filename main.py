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

# 1) Configuration initiale
load_dotenv()

# Variables d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
PORT = int(os.getenv("PORT", 5050))

# Validation des variables
if not all([OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
    raise ValueError("Missing required environment variables")

# Configuration OpenAI
SYSTEM_MESSAGE = "Vous êtes un assistant vocal professionnel. Répondez de manière concise en français."
VOICE = "alloy"
OPENAI_MODEL = "gpt-4o"  # Modèle validé

# Initialisation Twilio
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# 2) Endpoints principaux
@app.get("/")
def health_check():
    return {"status": "active", "service": "Outbound Call Gateway"}

@app.post("/outbound-call")
async def initiate_call(request: Request):
    """Déclenche un appel sortant via Twilio"""
    try:
        data = await request.json()
        to_number = data.get("to")
        
        if not to_number or not to_number.startswith("+"):
            return JSONResponse(
                {"error": "Numéro invalide. Format E.164 requis (ex: +33123456789)"}, 
                status_code=400
            )

        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_FROM_NUMBER,
            url=f"https://{request.url.hostname}/outbound-call-answered"
        )
        
        return {"status": "call_initiated", "call_sid": call.sid}

    except Exception as e:
        return JSONResponse(
            {"error": f"Erreur Twilio: {str(e)}"},
            status_code=500
        )

@app.post("/outbound-call-answered")
def handle_call_answer(request: Request):
    """Génère le TwiML de connexion au flux média"""
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{request.url.hostname}/media-stream")
    response.append(connect)
    return HTMLResponse(str(response), media_type="application/xml")

# 3) Gestion du WebSocket
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("\n[WebSocket] Connexion Twilio établie")

    try:
        # Connexion à OpenAI
        print(f"[OpenAI] Connexion à {OPENAI_MODEL}...")
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            ping_timeout=15,
            close_timeout=10
        ) as openai_ws:
            print("[OpenAI] Connexion réussie!")

            # Initialisation session
            await openai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "voice": VOICE,
                    "instructions": SYSTEM_MESSAGE,
                    "modalities": ["text", "audio"],
                    "temperature": 0.7
                }
            }))
            print("[OpenAI] Session configurée")

            # Variables d'état
            stream_sid = None
            last_audio_timestamp = 0

            async def handle_twilio_messages():
                """Traite les messages entrants de Twilio"""
                nonlocal stream_sid, last_audio_timestamp
                
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        
                        if data.get("event") == "start":
                            stream_sid = data["start"]["streamSid"]
                            print(f"[Twilio] Stream démarré: {stream_sid}")
                            
                            # Envoi message d'accueil
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": "Bonjour ! Comment puis-je vous aider ?"
                                    }]
                                }
                            }))
                            await openai_ws.send(json.dumps({"type": "response.create"}))
                        
                        elif data.get("event") == "media":
                            last_audio_timestamp = int(data["media"]["timestamp"])
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": data["media"]["payload"]  # Payload direct
                            }))

                except WebSocketDisconnect:
                    print("[Twilio] Déconnexion inattendue")

            async def handle_openai_messages():
                """Traite les messages entrants d'OpenAI"""
                try:
                    async for message in openai_ws:
                        data = json.loads(message)
                        
                        if data.get("type") == "response.audio.delta" and "delta" in data:
                            # Envoi audio directement sans re-encodage
                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": data["delta"]}
                            })
                            
                            # Synchronisation
                            await websocket.send_json({
                                "event": "mark",
                                "streamSid": stream_sid,
                                "mark": {"name": "audio_chunk"}
                            })

                except Exception as e:
                    print(f"[OpenAI] Erreur: {str(e)}")

            # Exécution parallèle
            await asyncio.gather(
                handle_twilio_messages(),
                handle_openai_messages()
            )

    except Exception as e:
        print(f"[ERREUR CRITIQUE] {str(e)}")
    finally:
        await websocket.close()
        print("[WebSocket] Connexion fermée")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
