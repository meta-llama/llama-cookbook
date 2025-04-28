from together import Together
from openai import OpenAI 
import os
import base64
import asyncio
import requests
import httpx
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path
from groq import Groq
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
#LLAMA_API_URL = os.getenv("API_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL")

def text_to_speech(text: str, output_path: str = "reply.mp3") -> str:
    """
    Synthesizes a given text into an audio file using Groq's TTS service.

    Args:
        text (str): The text to be synthesized.
        output_path (str): The path where the output audio file will be saved. Defaults to "reply.mp3".

    Returns:
        str: The path to the output audio file, or None if the synthesis failed.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Aaliyah-PlayAI",
            response_format="mp3",
            input=text
        )
        
        # Convert string path to Path object and stream the response to a file
        path_obj = Path(output_path)
        response.write_to_file(path_obj)
        return str(path_obj)
    except Exception as e:
        print(f"TTS failed: {e}")
        return None


def speech_to_text(input_path: str) -> str:
    """
    Transcribe an audio file using Groq.

    Args:
        input_path (str): Path to the audio file to be transcribed.
        output_path (str, optional): Path to the output file where the transcription will be saved. Defaults to "transcription.txt".

    Returns:
        str: The transcribed text.
    """

    client = Groq(api_key=GROQ_API_KEY)
    with open(input_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            model="distil-whisper-large-v3-en",
            response_format="verbose_json",
            file=(input_path, file.read())
        )
        transcription.text

    return transcription.text
      




def get_llm_response(text_input: str, image_input : str = None) -> str:
    """
    Get the response from the Together AI LLM given a text input and an optional image input.

    Args:
        text_input (str): The text to be sent to the LLM.
        image_input (str, optional): The base64 encoded image to be sent to the LLM. Defaults to None.

    Returns:
        str: The response from the LLM.
    """
    messages = []
    # print(bool(image_input))
    if image_input:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_input}"}
        })
    messages.append({
        "type": "text",
        "text": text_input
    })
    try:
        #client = Together(api_key=TOGETHER_API_KEY)
        client = OpenAI(base_url= "https://api.llama.com/compat/v1/")
        completion = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": messages
                }
            ]
        )
        
        if completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            print("Empty response from Together API")
            return None
    except Exception as e:
        print(f"LLM error: {e}")
        return None







async def fetch_media(media_id: str) -> str:
    """
    Fetches the URL of a media given its ID.

    Args:
        media_id (str): The ID of the media to fetch.

    Returns:
        str: The URL of the media.
    """
    url = "https://graph.facebook.com/v22.0/{media_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url.format(media_id=media_id),
                headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
            )
            if response.status_code == 200:
                return response.json().get("url")
            else:
                print(f"Failed to fetch media: {response.text}")
        except Exception as e:
            print(f"Exception during media fetch: {e}")
    return None

async def handle_image_message(media_id: str) -> str:
    """
    Handle an image message by fetching the image media, converting it to base64,
    and returning the base64 string.

    Args:
        media_id (str): The ID of the image media to fetch.

    Returns:
        str: The base64 string of the image.
    """
    media_url = await fetch_media(media_id)
    # print(media_url)
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
        response = await client.get(media_url, headers=headers)
        response.raise_for_status()

        # Convert image to base64
        image = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # Save as JPEG
        # image.save("./test.jpeg", format="JPEG")  # Optional save
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return base64_image

async def handle_audio_message(media_id: str):
    """
    Handle an audio message by fetching the audio media, writing it to a temporary file,
    and then using Groq to transcribe the audio to text.

    Args:
        media_id (str): The ID of the audio media to fetch.

    Returns:
        str: The transcribed text.
    """
    media_url = await fetch_media(media_id)
    # print(media_url)
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
        response = await client.get(media_url, headers=headers)

        response.raise_for_status()
        audio_bytes = response.content
        temp_audio_path = "temp_audio.m4a"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        return speech_to_text(temp_audio_path)

async def send_audio_message(to: str, file_path: str):
    """
    Send an audio message to a WhatsApp user.

    Args:
        to (str): The phone number of the recipient.
        file_path (str): The path to the audio file to be sent.

    Returns:
        None

    Raises:
        None
    """
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/media"
    with open(file_path, "rb") as f:
        files = { "file": ("reply.mp3", open(file_path, "rb"), "audio/mpeg")}
        params = {
            "messaging_product": "whatsapp",
            "type": "audio",
            "access_token": META_ACCESS_TOKEN
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
            "Authorization": f"Bearer {META_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        requests.post(WHATSAPP_API_URL, headers=headers, json=payload)
    else:
        print("Audio upload failed:", response.text)