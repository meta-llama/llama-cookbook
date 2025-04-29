from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import send_message,llm_reply_to_text,handle_image_message,get_llm_response,send_audio_message,fetch_media,text_to_speech,llm_reply_to_text_v2,audio_conversion
import os
import requests
import httpx
from dotenv import load_dotenv
#from utils import handle_image_message

load_dotenv()
app = FastAPI()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
AGENT_URL = os.getenv("AGENT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
class WhatsAppMessage(BaseModel):
    object: str
    entry: list


# @app.get("/webhook")
# async def verify_webhook(request: Request):
#     mode = request.query_params.get("hub.mode")
#     token = request.query_params.get("hub.verify_token")
#     challenge = request.query_params.get("hub.challenge")
#     print(mode)
#     print(token)
#     print(challenge)

#     # if mode and token and mode == "subscribe" and token == "1234":
#     #     return {"hub_verfiy_mode":mode,"hub_verify_token":token, "hub_verify_challange":challenge }
#     # return token

#     return int(challenge)
#     # return {"error": "Invalid verification token"}





@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    message_data = WhatsAppMessage(**data)
    
    change = message_data.entry[0]["changes"][0]["value"]
    print(change)
    if 'messages' in change:
        message = change["messages"][-1]
        user_phone = message["from"]
        print(message)
        if "text" in message:
            user_message = message["text"]["body"].lower()
            print(user_message)
            background_tasks.add_task(llm_reply_to_text_v2, user_message, user_phone,None,None)
        elif "image" in message:
            media_id = message["image"]["id"]
            print(media_id)
            caption = message["image"].get("caption", "")
            # background_tasks.add_task(handle_image_message, media_id, user_phone, caption)
            background_tasks.add_task(llm_reply_to_text_v2,caption,user_phone,media_id,'image')
        elif message.get("audio"):
            media_id = message["audio"]["id"]
            print(media_id)
            path = await audio_conversion("",media_id,'audio')
            # Send final audio reply
            print(user_phone)
            await send_audio_message(user_phone, path)
        return JSONResponse(content={"status": "ok"}), 200