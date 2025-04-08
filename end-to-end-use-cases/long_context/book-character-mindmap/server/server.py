from flask import Flask, jsonify, request
from flask_cors import CORS
from vllm import LLM, sampling_params, SamplingParams

# Flask setup
app = Flask(__name__)
CORS(app)

# Memory initialization
active_sessions = {}

SYSTEM_PROMPT = """You are tasked with performing a comprehensive analysis of a book excerpt to extract every character mentioned and all identifiable relationships between them. Your job is to produce a structured JSON object that includes the book’s title, a summary of its narrative, a full list of characters (nodes), and labeled relationships (links) among them.

Focus on capturing every character mentioned in the text—no matter how minor—and clearly define their relationships. Pay attention to implied, familial, professional, social, romantic, antagonistic, and historical relationships. Avoid missing any entity that might reasonably be considered a character in the context of the narrative.

Your output must be a single valid JSON object and must not include any explanatory text.

Input:

A text excerpt from a book.

Output:

A valid JSON object representing the characters and their relationships. Make sure only to include json format and no other text.

⸻

JSON Format Specification:

1. title: The title of the book.
2. summary: A brief summary of the book's content.
3. Nodes: Each character should be a node with the following properties:
	•	"id": A unique identifier for the character (e.g., "c1", "char_john").
	•	"name": The full name of the character as it appears in the text.
	•	"val": A numerical value for the character, increasing sequentially starting from 1.
4. Links: Each relationship should be a link with the following properties:
	•	"source": The id of the source character.
	•	"target": The id of the target character.
	•	"label":  A detailed, natural-language description of the relationship, including context, roles, emotional dynamics, or historical connections whenever possible. Avoid vague terms—be specific (e.g., "childhood best friend and traveling companion of", "rival general who betrayed during the siege", "secret lover and political adversary of").

  ## Example JSON Structure:
  JSON
{
  "title": "The Fellowship of the Ring",
  "summary": "In the first part of the epic trilogy, Frodo Baggins inherits a powerful ring that must be destroyed to stop the rise of evil. He sets out on a perilous journey with a group of companions to reach Mount Doom. Along the way, they face temptation, betrayal, and battles that test their unity and resolve.",
  "nodes": [
    { "id": "c1", "name": "Frodo Baggins", "val": 1 },
    { "id": "c2", "name": "Samwise Gamgee", "val": 2 },
    { "id": "c3", "name": "Gandalf", "val": 3 },
    { "id": "c4", "name": "Aragorn", "val": 4 }
  ],
  "links": [
    { "source": "c2", "target": "c1", "label": "childhood friend and fiercely loyal traveling companion of" },
    { "source": "c3", "target": "c1", "label": "wise mentor who guides Frodo through early parts of the journey and warns him about the Ring's power" },
    { "source": "c4", "target": "c3", "label": "trusted warrior and future king who follows Gandalf’s counsel during the quest" }
  ]
}
Note: Every character mentioned must be represented in nodes, and all relevant connections must be captured in links, with the "label" field providing as much contextual, emotional, or historical detail as possible. Treat relationship descriptions as mini-narratives that reflect their complexity in the story."""


# llm = LLM(
#         model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
#         enforce_eager=False,
#         tensor_parallel_size=8,
#         max_model_len=1100000,
#         override_generation_config= {
#             "attn_temperature_tuning": True,
#         }
#     )
# sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=16000)
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    enforce_eager=False,
    tensor_parallel_size=1,  # Reduce for smaller model
    max_model_len=2048,  # Smaller context window for efficiency
)
sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=4096)


@app.route("/inference", methods=["POST"])
def inference():
    """
    Handles inference requests from the frontend.
    """

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read file content directly from the uploaded file
        file_content = file.read().decode("utf-8")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": file_content},
        ]

        outputs = llm.chat(messages, sampling_params)

        # Clean the response to handle markdown code blocks
        response_text = outputs[0].outputs[0].text
        print("response_text: ", response_text)
        cleaned_response = (
            response_text.replace("```json", "").replace("```", "").strip()
        )

        # Handle potential leading/trailing backticks
        if cleaned_response.startswith("`"):
            cleaned_response = cleaned_response[1:]
        if cleaned_response.endswith("`"):
            cleaned_response = cleaned_response[:-1]
        # cleaned_response = read_file_to_string(filepath)

        # cleaned_response = "test"

        return jsonify({"response": cleaned_response}), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500
