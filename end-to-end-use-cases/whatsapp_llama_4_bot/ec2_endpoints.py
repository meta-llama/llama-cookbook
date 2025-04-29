from fastapi import FastAPI, HTTPException 
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from service import text_to_speech, get_llm_response, handle_image_message,handle_audio_message,send_audio_message
from enum import Enum
app = FastAPI()

class TextToSpeechRequest(BaseModel):
    text: str
    output_path: Optional[str] = "reply.mp3"

class TextToSpeechResponse(BaseModel):
    file_path: Optional[str]
    error: Optional[str] = None

class KindEnum(str, Enum):
    audio = "audio"
    image = "image"

class LLMRequest(BaseModel):
    user_input: str
    media_id: Optional[str] = None
    kind: Optional[KindEnum] = None


class LLMResponse(BaseModel):
    response: Optional[str]
    error: Optional[str] = None

@app.post("/llm-response", response_model=LLMResponse)
async def api_llm_response(req: LLMRequest):
    text_message = req.user_input
    image_base64 = None
    if req.kind == KindEnum.image:
        image_base64 = await handle_image_message(req.media_id)
        result = get_llm_response(text_message, image_input=image_base64)
        # print(result)
    elif req.kind == KindEnum.audio:
        text_message = await handle_audio_message(req.media_id)
        result = get_llm_response(text_message)
        audio_path = text_to_speech(text=result, output_path="reply.mp3")
        return FileResponse(audio_path, media_type="audio/mpeg", filename="reply.mp3")
    else:
        result = get_llm_response(text_message)
    
    if result is None:
        return LLMResponse(response=None, error="LLM response generation failed.")
    return LLMResponse(response=result)