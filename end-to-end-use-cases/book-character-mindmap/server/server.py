import json
import logging

from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoTokenizer
from vllm import LLM, sampling_params, SamplingParams

# Flask setup
app = Flask(__name__)
CORS(app)

CHARACTER_SYSTEM_PROMPT = """
You are a highly detailed literary analyst AI. Your sole mission is to meticulously extract comprehensive information about characters and the *nuances* of their relationships from the provided text segment. This data will be used later to build a relationship graph.

**Objective:** Identify EVERY character mentioned. For each pair of interacting characters, describe their relationship in detail, focusing on the context, roles, emotional dynamics, history, and key interactions *as presented or clearly implied* within this specific text segment.

**Instructions:**

1.  **Identify Characters:** List every unique character name mentioned in the text segment.
2.  **Identify Relationships & Interactions:** For each character, document their interactions and connections with *every other* character mentioned *within this segment*.
3.  **Describe Relationship Nuances:** Do not just state the type (e.g., "friend"). Describe the *quality and context* of the relationship based *only* on the text. Note:
    * **Roles:** (e.g., mentor-mentee, leader-follower, parent-child, rivals for power, allies in battle).
    * **Emotional Dynamics:** (e.g., loyalty, distrust, affection, resentment, fear, admiration).
    * **History:** (e.g., childhood friends, former enemies, long-lost siblings, recent acquaintances).
    * **Key Events/Context:** Mention specific events, shared goals, conflicts, or settings *within this segment* that define or illustrate the relationship (e.g., "fought side-by-side during the siege," "argued fiercely over the inheritance," "shared a secret confided in the garden").
4.  **Quote Evidence (Briefly):** If a short quote directly illuminates the nature of the relationship, include it as supporting evidence.
5.  **Be Exhaustive:** Capture every piece of relationship information present *in this specific text segment*.
6.  **Stick Strictly to the Text:** Base your analysis *only* on the provided text segment. Do not infer information not present, make assumptions, or bring in outside knowledge.
7.  **Output Format:** Present the findings as clear, descriptive text for each character, detailing their relationships. **DO NOT use JSON or graph formats (nodes/links) at this stage.** Focus purely on capturing rich, accurate, descriptive textual data about the relationships.

**Example Output Structure (Conceptual):**

* **Character:** [Character Name A]
    * **Relationship with [Character Name B]:** Described as close friends since childhood ('lifelong companions' mentioned). In this segment, Character A relies on B for emotional support during the journey planning. Character B shows fierce loyalty, vowing to protect A.
    * **Relationship with [Character Name C]:** Character C acts as a mentor, providing guidance about the ancient artifact. Character A shows respect but also some fear of C's power, as seen when A hesitates to ask a direct question.
    * **Relationship with [Character Name D]:** Openly antagonistic rivals. In this segment, they have a heated argument regarding leadership strategy, revealing deep-seated distrust. Character A believes D is reckless.

Process the provided text segment thoroughly based *only* on these instructions.
"""

RELATIONSHIP_SYSTEM_PROMPT = """
You are an expert data architect AI specializing in transforming literary analysis into structured graph data. Your task is to synthesize character and relationship information into a specific JSON format containing nodes and links, including a title and summary.

**Objective:** Convert the provided textual analysis of characters and relationships (extracted from a book) into the specified JSON graph format. Generate unique IDs, sequential values, and synthesize detailed relationship descriptions into link labels.
I'll give you a harsh punishment if you miss any character or relationship.

**Input:**
1.  **Character & Relationship Data:** Unstructured or semi-structured text detailing character names and rich descriptions of their relationships (context, roles, dynamics, history, key interactions). This data is compiled from the analysis of the entire book.
2.  **Book Title:** The full title of the book.
3.  **Book Summary:** A brief summary of the book's plot or content.

**Instructions:**

1.  **Identify Unique Characters:** From the input data, identify the list of all unique characters.
2.  **Generate Nodes:** Create a JSON list under the key `"nodes"`. For each unique character:
    * Assign a unique `"id"` string (e.g., "c1", "c2", "c3"...). Keep a mapping of character names to their assigned IDs.
    * Include the character's full `"name"` as found in the data.
    * Assign a sequential integer `"val"`, starting from 1.
3.  **Generate Links:** Create a JSON list under the key `"links"`. For each distinct relationship between two characters identified in the input data:
    * Determine the `source` character's ID and the `target` character's ID using the mapping created in step 2.
    * **Synthesize the Relationship Label:** Carefully analyze the *detailed description* of the relationship provided in the input data (including roles, dynamics, context, history). Create a concise yet descriptive **natural-language `"label"`** that captures the essence of this relationship.
        * **Focus on Specificity:** Avoid vague terms like "friend" or "related to". Use descriptive phrases like the examples provided (e.g., "childhood best friend and traveling companion of", "rival general who betrayed during the siege", "wise mentor guiding the protagonist", "secret lover and political adversary of").
        * The label should ideally describe the relationship *from* the source *to* the target, or be neutral if applicable (e.g., "siblings").
    * Ensure each significant relationship pair is represented by a link object. A single mutual relationship should typically be represented by one link, with the label reflecting the connection. If the relationship is distinctly different from each perspective, consider if two links are necessary.
4.  **Assemble Final JSON:** Construct the final JSON object with the following top-level keys:
    * `"title"`: Use the provided Book Title.
    * `"summary"`: Use the provided Book Summary.
    * `"nodes"`: The list of node objects created in step 2.
    * `"links"`: The list of link objects created in step 3.
5.  **Strict JSON Output:** Generate *only* the complete, valid JSON object adhering to the specified structure. Do not include any introductory text, explanations, comments, or markdown formatting outside the JSON structure itself. If you include one of them, I'll give you a punishment. You are gonna get a

**Target JSON Structure Example:**

```json
{
  "title": "The Fellowship of the Ring",
  "summary": "In the first part of the epic trilogy, Frodo Baggins inherits a powerful ring that must be destroyed to stop the rise of evil. He sets out on a perilous journey with a group of companions to reach Mount Doom. Along the way, they face temptation, betrayal, and battles that test their unity and resolve.",
  "nodes": [
    { "id": "c1", "name": "Frodo Baggins", "val": 1 },
    { "id": "c2", "name": "Samwise Gamgee", "val": 2 },
    { "id": "c3", "name": "Gandalf", "val": 3 },
    { "id": "c4", "name": "Aragorn", "val": 4 }
    // ... other characters
  ],
  "links": [
    { "source": "c2", "target": "c1", "label": "childhood friend and fiercely loyal traveling companion of" },
    { "source": "c3", "target": "c1", "label": "wise mentor who guides Frodo through early parts of the journey and warns him about the Ring's power" },
    { "source": "c4", "target": "c3", "label": "trusted warrior and future king who follows Gandalf’s counsel during the quest" }
    // ... other relationships
  ]
}
```
"""

JSON_SYSTEM_PROMPT = """
You are an extremely precise and strict JSON extractor.
Extract only the complete JSON object from the input. Get the last one if there are multiple.
Output must:
1. Start with opening brace {
2. End with closing brace }
3. Contain no text, markdown, or other characters outside the JSON
4. Be valid, parseable JSON
```
"""

SEARCH_SYSTEM_PROMPT = """
You are an expert search AI designed to help users find detailed information about character relationships from a book. Your task is to assist users in querying the relationship data extracted from the book.

**Objective:** Allow users to search for specific character relationships using natural language queries. Provide concise and accurate responses based on the relationship data.

**Instructions:**

1. **Understand the Query:** Analyze the user's query to identify the characters and the type of relationship information they are seeking.
2. **Search Relationship Data:** Use the relationship data extracted from the book to find relevant information. Focus on the characters and relationship details mentioned in the query.
3. **Provide Clear Responses:** Respond with clear and concise information about the relationship, including roles, dynamics, history, and key interactions as described in the data.
4. **Be Specific:** Avoid vague responses. Use specific details from the relationship data to answer the query.
5. **Maintain Context:** Ensure that the response is relevant to the query and provides a comprehensive understanding of the relationship.

**Example Query and Response:**

*Query:* "What is the relationship between Frodo Baggins and Samwise Gamgee?"

*Response:* "Samwise Gamgee is Frodo Baggins' childhood friend and fiercely loyal traveling companion. He provides emotional support and protection during their journey."

Use this format to assist users in finding the relationship information they need.
"""

HEAVY_MODEL = True

if HEAVY_MODEL:
    # LLM_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    LLM_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    llm = LLM(
        model=LLM_MODEL,
        enforce_eager=False,
        tensor_parallel_size=8,
        max_model_len=500000,
        override_generation_config={
            "attn_temperature_tuning": True,
        },
    )
    sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=10000)
else:
    LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    llm = LLM(
        model=LLM_MODEL,
        enforce_eager=False,
        tensor_parallel_size=1,  # Reduce for smaller model
        max_model_len=4096,  # Smaller context window for efficiency
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

        with open("book.txt", "w") as f:
            f.write(file_content)

        # Calculate the number of input tokens
        num_input_tokens = calculate_input_tokens(file_content)

        # Step 1: Character extraction
        messages = [
            {"role": "system", "content": CHARACTER_SYSTEM_PROMPT},
            {"role": "user", "content": file_content},
        ]
        character_outputs = llm.chat(messages, sampling_params)
        character_response_text = character_outputs[0].outputs[0].text
        print("character_response_text: ", character_response_text)

        # Step 2: Relationship extraction
        messages = [
            {"role": "system", "content": RELATIONSHIP_SYSTEM_PROMPT},
            {"role": "user", "content": f"Book content:\n{file_content}"},
            {"role": "assistant", "content": character_response_text},
            {
                "role": "user",
                "content": "Generate the JSON graph with title, summary, nodes, and links.",
            },
        ]
        relationship_outputs = llm.chat(messages, sampling_params)
        relationship_response_text = relationship_outputs[0].outputs[0].text
        print("relationship_response_text: ", relationship_response_text)

        graph_data = ""
        try:
            graph_data = jsonify_graph_response(relationship_response_text)
            logging.info("Graph data generated:", json.dumps(graph_data, indent=2))
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing graph response from : {e}")
            try:
                # Try to parse the response as a JSON object
                json_response = llm_json_output(relationship_response_text)
                print("json_response: ", json_response)
                graph_data = jsonify_graph_response(json_response)
                logging.info("Graph data generated:", json.dumps(graph_data, indent=2))
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing graph response from json result: {e}")

        return (
            jsonify(
                {
                    "graph_data": graph_data,
                    "character_response_text": character_response_text,
                    "num_input_tokens": num_input_tokens,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles search requests from the frontend.
    """
    try:
        data = request.json
        search_query = data.get("query")
        relationship_data = data.get("relationship_data")
        chat_history_data = data.get("chat_history_data")

        with open("book.txt", "r") as f:
            file_content = f.read()

        if not search_query or not relationship_data:
            return (
                jsonify({"error": "search_query and relationship_data are required"}),
                400,
            )
        messages = [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "assistant", "content": file_content},
            {"role": "assistant", "content": relationship_data},
        ]

        # Format chat history for the model
        formatted_history = []
        for msg in chat_history_data:
            formatted_history.append({"role": msg["sender"], "content": msg["text"]})

        # Add chat history
        messages.extend(formatted_history)

        # Add the current user message
        messages.append({"role": "user", "content": search_query})

        search_outputs = llm.chat(messages, sampling_params)
        search_response_text = search_outputs[0].outputs[0].text
        print("search_response_text: ", search_response_text)
        return jsonify({"response": search_response_text}), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


def llm_json_output(response):
    messages = [
        {"role": "system", "content": JSON_SYSTEM_PROMPT},
        {"role": "user", "content": response},
    ]

    outputs = llm.chat(messages, sampling_params)

    response_text = outputs[0].outputs[0].text
    print("response_text: ", response_text)
    return response_text


def calculate_input_tokens(input_text):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    input_tokens = tokenized_input["input_ids"].size(1)
    return input_tokens


def jsonify_graph_response(content):
    """Extract and parse JSON content from graph response."""
    try:
        # Find indices of first { and last }
        start_idx = content.find("{")
        end_idx = content.rfind("}")

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid JSON object found in response")

        # Extract JSON string
        json_str = content[start_idx : end_idx + 1]

        # Parse JSON
        return json.loads(json_str)

    except Exception as e:
        logging.error(f"Error parsing graph response: {e}")
        return None


if __name__ == "__main__":
    app.run(debug=False, port=5001)
