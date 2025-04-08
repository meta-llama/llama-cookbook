import React, { useState } from "react";
import { FaSearch, FaBook, FaHome } from "react-icons/fa";
import { Link } from "react-router-dom";
import AISearch from "./components/AISearch";
import CharacterGraph from "./components/CharacterGraph";
import Together from "together-ai";

const SYSTEM_PROMPT = `You are tasked with performing a comprehensive analysis of a book excerpt to extract every character mentioned and all identifiable relationships between them. Your job is to produce a structured JSON object that includes the book’s title, a summary of its narrative, a full list of characters (nodes), and labeled relationships (links) among them.

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
Note: Every character mentioned must be represented in nodes, and all relevant connections must be captured in links, with the "label" field providing as much contextual, emotional, or historical detail as possible. Treat relationship descriptions as mini-narratives that reflect their complexity in the story.`;

const together = new Together({
  apiKey: process.env.REACT_APP_TOGETHER_API_KEY
});

export default function BookPage() {
  const [filePath, setFilePath] = useState("");
  const [fileObject, setFileObject] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [bookData, setBookData] = useState(null);
  const [searchComplete, setSearchComplete] = useState(false);

  const readFileContent = async (file) => {
    try {
      const reader = new FileReader();
      return new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject("Error reading file");
        reader.readAsText(file);
      });
    } catch (error) {
      console.error("Error reading file content:", error);
      return "Failed to read file content.";
    }
  };

  const submitQuery = async (query) => {
    try {
      const response = await together.chat.completions.create({
        // model: "Llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        model: "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages: [{
          role: "system",
          content: SYSTEM_PROMPT
        },
        {
          role: "user",
          content: query
        }]
      });
      console.log("Response:", response.choices[0].message.content);
      return response.choices[0].message.content;
    } catch (error) {
      console.error("Error submitting query:", error);
      return "Sorry, I couldn't generate a response.";
    }
  };

  const verificationGraphData = (graphData) => {
    try {
      // Create Set of valid node IDs
      const nodeIds = new Set(graphData.nodes.map((node) => node.id));

      // Filter links to only include valid node references
      const validLinks = graphData.links.filter(
        (link) => nodeIds.has(link.source) && nodeIds.has(link.target)
      );

      return {
        nodes: graphData.nodes,
        links: validLinks,
      };
    } catch (error) {
      console.error("Error validating graph data:", error);
      return graphData; // Return original data if validation fails
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!filePath.trim()) {
      return;
    }

    setIsLoading(true);
    try {
      // Initialize memory and fetch book data in parallel
      // const [memoryResponse, bookInfo] = await Promise.all([
      //   initializeMemory(filePath),
      //   fetchBookData(filePath),
      // ]);
      const fileContent = await readFileContent(fileObject);
      console.log("fileContent", fileContent)
      // const queryResult = await submitQuery(fileContent);
      const queryResultString = await submitQuery(fileContent);
      const queryResult = JSON.parse(queryResultString.replace(/^```json|```$/g, ""));
      // const queryResult = JSON.parse(`{
      //   "title": "The Reunion",
      //   "summary": "A group of friends reunite in their hometown five years after high school, rekindling their old relationships and navigating changes in their lives.",
      //   "nodes": [
      //     { "id": "c1", "name": "Alex Chen", "val": 10 },
      //     { "id": "c2", "name": "Emily Patel", "val": 10 },
      //     { "id": "c3", "name": "Jake Lee", "val": 10 },
      //     { "id": "c4", "name": "Sarah Kim", "val": 10 }
      //   ],
      //   "links": [
      //     { "source": "c1", "target": "c2", "label": "best friends with" },
      //     { "source": "c1", "target": "c3", "label": "friends with" },
      //     { "source": "c1", "target": "c4", "label": "friends with" },
      //     { "source": "c2", "target": "c3", "label": "friends with" },
      //     { "source": "c2", "target": "c4", "label": "friends with" },
      //     { "source": "c3", "target": "c4", "label": "dating" },
      //     { "source": "c3", "target": "c1", "label": "friends with" },
      //     { "source": "c4", "target": "c1", "label": "friends with" },
      //     { "source": "c4", "target": "c2", "label": "friends with" }
      //   ]
      // }`);

      // setBookData({
      //   title: bookInfo.title,
      //   subtitle: bookInfo.summary,
      //   posterUrl: bookInfo.coverUrl,
      //   author: bookInfo.author,
      //   publishedDate: bookInfo.publishedDate,
      //   pageCount: bookInfo.pageCount,
      // });

      setBookData({
        title: queryResult.title,
        subtitle: queryResult.summary,
        posterUrl: "",
        author: "",
        publishedDate: "",
        pageCount: "",
      });

      const graphData = verificationGraphData({
        nodes: queryResult.nodes,
        links: queryResult.links,
      })

      setGraphData(graphData);
      setSearchComplete(true);
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      setIsLoading(false);
    }
  };
  // Add new function to fetch book cover
  // const fetchBookData = async (bookTitle) => {
  //   try {
  //     const response = await axios.get(
  //       `https://www.googleapis.com/books/v1/volumes`,
  //       {
  //         params: {
  //           q: bookTitle,
  //           key: process.env.REACT_APP_GOOGLE_BOOKS_API_KEY,
  //         },
  //       }
  //     );

  //     if (response.data.items && response.data.items[0]) {
  //       const volumeInfo = response.data.items[0].volumeInfo;
  //       const imageLinks = volumeInfo.imageLinks || {};

  //       return {
  //         coverUrl:
  //           imageLinks.extraLarge ||
  //           imageLinks.large ||
  //           imageLinks.medium ||
  //           imageLinks.thumbnail ||
  //           "/placeholder.jpg",
  //         summary: volumeInfo.description || "No summary available",
  //         title: volumeInfo.title,
  //         author: volumeInfo.authors?.[0] || "Unknown Author",
  //         publishedDate: volumeInfo.publishedDate,
  //         pageCount: volumeInfo.pageCount,
  //       };
  //     }

  //     return {
  //       coverUrl: "/placeholder.jpg",
  //       summary: "No summary available",
  //       title: bookTitle,
  //       author: "Unknown Author",
  //       publishedDate: "",
  //       pageCount: 0,
  //     };
  //   } catch (error) {
  //     console.error("Error fetching book data:", error);
  //     return {
  //       coverUrl: "/placeholder.jpg",
  //       summary: "Failed to load book information",
  //       title: bookTitle,
  //       author: "Unknown Author",
  //       publishedDate: "",
  //       pageCount: 0,
  //     };
  //   }
  // };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Home Button */}
        <div className="flex justify-center mb-4">
          <Link
            to="/"
            className="flex items-center text-blue-600 hover:text-blue-800 transition-colors"
          >
            <FaHome className="mr-2" />
            Home
          </Link>
        </div>

        {/* Search Section */}
        <div className="max-w-md mx-auto mb-16">
          <h1 className="text-4xl font-extrabold text-center mb-8 text-gray-800 tracking-tight">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
              Character Mind Map
            </span>
          </h1>
          <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-8 space-y-6 transform transition-all duration-300 hover:scale-[1.02]">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <div className="w-full px-5 py-3 rounded-lg border-2 border-dashed border-gray-200
                    focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-200
                    transition-all duration-200 bg-white/90 text-center">
                  <label className="flex flex-col items-center justify-center cursor-pointer">
                    {filePath ? (
                      <>
                        <div className="flex items-center text-blue-500">
                          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                          </svg>
                          <span className="truncate max-w-xs">{filePath.split('/').pop()}</span>
                        </div>
                        <span className="text-xs text-gray-500 mt-1">Click to change file</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                        </svg>
                        <span className="text-gray-500">Upload book file (.txt, .pdf, .doc, .docx)</span>
                      </>
                    )}
                    <input
                      type="file"
                      className="hidden"
                      accept=".txt,.pdf,.doc,.docx"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          setFilePath(file.name);
                          setFileObject(file);
                        }
                      }}
                      disabled={isLoading}
                    />
                  </label>
                </div>
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-blue-500 to-indigo-600
                       text-white font-semibold py-3 px-6 rounded-lg
                       transform transition-all duration-200
                       hover:from-blue-600 hover:to-indigo-700
                       focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
                       disabled:opacity-50 disabled:cursor-not-allowed
                       flex items-center justify-center space-x-2"
              >
                <FaSearch className={`${isLoading ? "animate-spin" : ""}`} />
                <span>{isLoading ? "Generating..." : "Visualize"}</span>
              </button>
            </form>
          </div>
          <p className="mt-4 text-center text-sm text-gray-600">
            Search for any book or movie to explore character relationships
          </p>
        </div>

        {/* Info Section - Only show when search is complete */}
        {searchComplete && bookData && (
          <div className="space-y-8">
            <div className="bg-white shadow-xl rounded-xl overflow-hidden">
              <div className="md:flex">
                <div className="md:flex-shrink-0">
                  <img
                    src={bookData.posterUrl}
                    alt={bookData.title}
                    className="h-48 w-full object-cover md:h-full md:w-48"
                  />
                </div>
                <div className="p-8">
                  <div className="flex items-center">
                    <FaBook className="text-blue-500 mr-2" />
                    <h1 className="text-3xl font-bold text-gray-800">
                      {bookData.title}
                    </h1>
                  </div>
                  <p className="mt-2 text-gray-600">{bookData.subtitle}</p>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-6">
                <AISearch bookTitle={bookData.title} />
              </div>
              <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-6">
                <CharacterGraph graphData={graphData} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
