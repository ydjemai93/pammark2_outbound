# pam_markII

This application demonstrates how to use Python, [Twilio Voice](https://www.twilio.com/docs/voice) and [Media Streams](https://www.twilio.com/docs/voice/media-streams), and [OpenAI's Realtime API](https://platform.openai.com/docs/) to make a phone call to speak with an AI Assistant. 

The application opens websockets with the OpenAI Realtime API and Twilio, and sends voice audio from one to the other to enable a two-way conversation.

See [the official Twilio docs](https://www.twilio.com/docs/voice) for more details on Voice & Media Streams, and the [OpenAI Realtime docs](https://platform.openai.com/docs) for GPT-4o.

## Prerequisites

- **Python 3.9+** 
- A Twilio account (with a **Voice** number).
- An OpenAI account with **Realtime API** access (GPT-4o).
- `pip install -r requirements.txt`

## Local Setup

1. **ngrok** or another tunneling solution to expose local port 5050.
2. `python main.py`
3. Twilio -> "A call comes in" -> `https://<ngrok>/incoming-call`
4. Call your Twilio number, speak, and watch the AI respond in real time.

## Outbound calls ?

Not covered in this MVP, but you can adapt `/incoming-call` or use Twilio's API for outbound, passing the same TwiML. 

## Special features

- **Interrupt**: when user speech is detected, we truncate the AI TTS in progress (`conversation.item.truncate`).
- **Initial greet**: un-comment `await send_initial_conversation_item(openai_ws)` to have the AI speak first.

Enjoy building with **pam_markII**!
