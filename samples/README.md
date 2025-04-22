# Twilio Inbound Web Socket Streams:

Sends Three Events:
- Start => On the start of call
- Media => During Call
- Stop => When Call ends

```python
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse

import json


app = FastAPI()

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Whisper + Ollama server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Respond with TwiML to route call to WebSocket."""
    from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream

    response = VoiceResponse()
    response.say("Please wait while we connect your call to the AI voice assistant.")
    response.pause(length=1)
    response.say("Okay, you can start talking now.")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")
    await websocket.accept()

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data["event"] == "start":
                print("-"*200)
                print("Event [start]")
                print("-"*200)
                print("Payload: ", data)
                print("-"*200)

            elif data["event"] == "media":
                print("-"*200)
                print("Event [Media]")
                print("Payload: ", data)
                print("-"*200)
            
            elif data["event"] == "stop":
                print("-"*200)
                print("Event [Stop]")
                print("Payload: ", data)
                print("-"*200)
                
    except Exception as e:
        print(f"Exception occured::: {e.__class__} :: e")
        websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
```

Events Payload Looks like below:

```text
Recieving events from twilio phone calls

Start Event
---------------------------------------------------------------------------
{
    'event': 'start', 
    'sequenceNumber': '1', 
    'start': {
        'accountSid': 'AC3425a5d9b91c5a512447d30365ef2fa0', 
        'streamSid': 'MZdc5361ff2f79c53f738cd553b7357c82', 
        'callSid': 'CA13a60bfd63a9f320259ace4c17a3d82e', 
        'tracks': ['inbound'], 
        'mediaFormat': {
            'encoding': 'audio/x-mulaw', 
            'sampleRate': 8000, 
            'channels': 1
        }, 
        'customParameters': {}
    }, 
    'streamSid': 'MZdc5361ff2f79c53f738cd553b7357c82'
}
---------------------------------------------------------------------------

Media Event
---------------------------------------------------------------------------
{
    'event': 'media', 
    'sequenceNumber': '2', 
    'media': {
        'track': 'inbound', 
        'chunk': '1', 
        'timestamp': '9', 
        'payload': '...'
    }, 
    'streamSid': 'MZdc5361ff2f79c53f738cd553b7357c82'
}
---------------------------------------------------------------------------

Stop Event
---------------------------------------------------------------------------
{
    'event': 'stop', 
    'sequenceNumber': '99', 
    'streamSid': 'MZdc5361ff2f79c53f738cd553b7357c82', 
    'stop': {
        'accountSid': 'AC3425a5d9b91c5a512447d30365ef2fa0', 
        'callSid': 'CA13a60bfd63a9f320259ace4c17a3d82e'
    }
}
---------------------------------------------------------------------------
```