package com.example.llamaandroiddemo

import android.content.ContentResolver
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.IOException
import org.json.JSONObject
import java.io.File
import java.net.URLConnection
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import androidx.core.net.toUri

interface InferenceStreamingCallback {
    fun onStreamReceived(message: String)
    fun onStatStreamReceived(tps: Float)
}

class ExampleLlamaRemoteInference(remoteURL: String) {

    var remoteURL: String = ""

    init {
            this.remoteURL = remoteURL
    }

    fun inferenceStartWithoutAgent(modelName: String, temperature: Double, prompt: ArrayList<Message>, userProvidedSystemPrompt:String, ctx: Context): String {
        val future = CompletableFuture<String>()
        val thread = Thread {
            try {
                val response = inferenceCallWithoutAgent(modelName, temperature, prompt, userProvidedSystemPrompt, ctx, true);
                future.complete(response)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
        return future.get();
    }

    fun makeStreamingPostRequest(url: String, requestBody: RequestBody): Response {

        val client = OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS) // No timeout for streaming
            .build()

        if (AppUtils.API_KEY == "") {
            AppLogging.getInstance().log("API key not set, configure it in AppUtils")
        }

        val request = Request.Builder()
            .url(url)
            .addHeader("Authorization","Bearer " + AppUtils.API_KEY)
            .addHeader("Accept", "text/event-stream")
            .post(requestBody)
            .build()

        return client.newCall(request).execute()
    }

    private fun llamaChatCompletion(ctx: Context, modelName: String, conversationHistory: ArrayList<Message>, userProvidedSystemPrompt: String, temperature: Double){
        var msg = """
                        {
                          "role": "system",
                          "content": "$userProvidedSystemPrompt"
                        },
        """.trimIndent()

        msg += constructMessageForAPICall(conversationHistory,ctx)
        val thread = Thread {
            try {
                // Create request body
                val json = """
                {
                    "messages": [
                         $msg
                    ],
                    "model": "$modelName",
                    "repetition_penalty": 1,
                    "temperature": $temperature,
                    "top_p": 0.9,
                    "max_completion_tokens": 2048,
                    "stream": true
                }""".trimIndent()
                val requestBody = json.toRequestBody("application/json".toMediaType())
                // Make request
                val response = makeStreamingPostRequest("$remoteURL/v1/chat/completions", requestBody)
                val callback = ctx as InferenceStreamingCallback

                // Process streaming response
                response.use { res ->
                    if (!res.isSuccessful) throw IOException("Unexpected code $res")

                    res.body?.source()?.let { source ->
                        while (!source.exhausted()) {
                            val streamDelta = source.readUtf8Line()
                            if (streamDelta != null){
                                val jsonString = streamDelta.substringAfter("data: ")
                                if (jsonString != ""){
                                    val obj = JSONObject(jsonString)
                                    if (obj.has("choices")) {
                                        val choices = obj.getJSONArray("choices")
                                        if (choices.length() > 0) {
                                            val choice = choices.getJSONObject(0)
                                            if (choice.has("delta")) {
                                                val delta = choice.getJSONObject("delta")
                                                if (delta.has("content")) {
                                                    val result = delta.getString("content")
                                                    callback.onStreamReceived(result)
                                                }
                                            } else if (choice.has("text")) {
                                                val result = choice.getString("text")
                                                callback.onStreamReceived(result)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d("error",e.message.toString())
                e.printStackTrace()
            }
        }
        thread.start();
    }

    //Example running simple inference + tool calls without using agent's workflow
    private fun inferenceCallWithoutAgent(modelName: String, temperature: Double, conversationHistory: ArrayList<Message>, userProvidedSystemPrompt: String, ctx: Context, streaming: Boolean): String {

        llamaChatCompletion(ctx,modelName,conversationHistory, userProvidedSystemPrompt, temperature)
        return ""
    }

    private fun encodeImageToDataUrl(filePath: String): String {
        val mimeType = URLConnection.guessContentTypeFromName(filePath)
            ?: throw RuntimeException("Could not determine MIME type of the file")
        val imageFile = File(filePath)
        val encodedString = Base64.encodeToString(imageFile.readBytes(), Base64.NO_WRAP)
        return "data:image/jpeg;base64,$encodedString"
    }

    private fun getFilePathFromUri(contentResolver: ContentResolver, uri: Uri): String? {
        var filePath: String? = null
        val projection = arrayOf(MediaStore.Images.Media.DATA)
        val cursor: Cursor? = contentResolver.query(uri, projection, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val columnIndex = it.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
                filePath = it.getString(columnIndex)
            }
        }
        return filePath
    }

    private fun constructMessageForAPICall(
        conversationHistory: ArrayList<Message>,
        ctx: Context
    ):String {
        var prompt = ""
        var isPreviousChatImage = false
        val imagePromptList = ArrayList<String>()
        for ((index, chat) in conversationHistory.withIndex()) {
            if (chat.isSent) {
                // First image in the chat. Image must pair with a prompt
                if (chat.messageType == MessageType.IMAGE) {
                    val imageUri = chat.imagePath.toUri()
                    val contentResolver = ctx.contentResolver
                    val imageFilePath = getFilePathFromUri(contentResolver, imageUri)
                    val imageDataUrl = imageFilePath?.let { encodeImageToDataUrl(it) }
                    Log.d("imageDataURL",imageDataUrl.toString())
                    imagePromptList += """     
                        {
                              "type": "image_url",
                              "image_url": {
                                "url": "$imageDataUrl"
                              }
                        }
                    """.trimIndent()
                    isPreviousChatImage = true
                    continue
                }
                // Prompt right after the image
                else if (chat.messageType == MessageType.TEXT) {
                    if (isPreviousChatImage) {
                        var imagePrompts = ""
                        for ((idx, image) in imagePromptList.withIndex()) {
                            imagePrompts += image
                            if (idx < imagePromptList.lastIndex) {
                                imagePrompts += ","
                            }
                        }
                        prompt += """
                            {
                                "role": "user",
                                "content": [
                                    $imagePrompts,
                                    {
                                        "type": "text",
                                        "text": "${chat.text}"
                                    }
                                ]
                            }
                        """.trimIndent()
                        isPreviousChatImage = false
                    } else {
                        prompt += """
                            {
                              "role": "user",
                              "content": "${chat.text}"
                            }                                            
                        """.trimIndent()
                    }
                }
            } else {
                // assistant message/response
                // only text response
                prompt += """
                    {  "role": "assistant", 
                       "content": ${JSONObject.quote(chat.text)}
                    }                   
                """.trimIndent()
            }
            if (chat.messageType != MessageType.IMAGE && index != conversationHistory.lastIndex) {
                // This is NOT the last chat and not image
                prompt += ","
            }
        }
        Log.d("inference", "this is prompt: $prompt")
        return prompt
    }
}