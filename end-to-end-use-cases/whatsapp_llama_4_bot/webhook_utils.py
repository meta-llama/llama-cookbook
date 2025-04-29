import os
import base64
import asyncio
import requests
import httpx
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MEDIA_URL = "https://graph.facebook.com/v20.0/{media_id}"
BASE_URL = os.getenv("BASE_URL")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def send_message(to: str, text: str):
    if not text:
        print("Error: Message text is empty.")
        return

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        print("Message sent")
    else:
        print(f"Send failed: {response.text}")



async def send_message_async(user_phone: str, message: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, send_message, user_phone, message)



        
async def send_audio_message(to: str, file_path: str):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/media"
    with open(file_path, "rb") as f:
        files = { "file": ("reply.mp3", open(file_path, "rb"), "audio/mpeg")}
        params = {
            "messaging_product": "whatsapp",
            "type": "audio",
            "access_token": ACCESS_TOKEN
        }
        response = requests.post(url, params=params, files=files)

    if response.status_code == 200:
        media_id = response.json().get("id")
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "audio",
            "audio": {"id": media_id}
        }
        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        requests.post(WHATSAPP_API_URL, headers=headers, json=payload)
    else:
        print("Audio upload failed:", response.text)






async def llm_reply_to_text_v2(user_input: str, user_phone: str, media_id: str = None,kind: str = None):
    try:
        # print("inside this function")
        headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

        json_data = {
            'user_input': user_input,
            'media_id': media_id,
            'kind': kind
        }
        
        async with httpx.AsyncClient() as client:
          response = await client.post("https://df00-171-60-176-142.ngrok-free.app/llm-response", json=json_data, headers=headers,timeout=60)
          response_data = response.json()
          # print(response_data)
          if response.status_code == 200 and response_data['error'] == None:
              message_content = response_data['response']
              if message_content:
                  loop = asyncio.get_running_loop()
                  await loop.run_in_executor(None, send_message, user_phone, message_content)
              else:
                  print("Error: Empty message content from LLM API")
                  await send_message_async(user_phone, "Received empty response from LLM API.")
          else:
              print("Error: Invalid LLM API response", response_data)
              await send_message_async(user_phone, "Failed to process image due to an internal server error.")

    except Exception as e:
        print("LLM error:", e)
        await send_message_async(user_phone, "Sorry, something went wrong while generating a response.")