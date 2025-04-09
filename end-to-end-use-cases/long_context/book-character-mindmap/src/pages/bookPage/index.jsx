import React, { useState } from "react";
import { FaSearch, FaBook, FaHome } from "react-icons/fa";
import { Link } from "react-router-dom";
import CharacterGraph from "./components/CharacterGraph";
import ChatInterface from "./components/ChatInterface";
import axios from "axios";

export default function BookPage() {
  const [filePath, setFilePath] = useState("");
  const [fileObject, setFileObject] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [bookData, setBookData] = useState(null);
  const [searchComplete, setSearchComplete] = useState(false);
  const [tokenUsage, setTokenUsage] = useState(0);
  const [relationshipData, setRelationshipData] = useState(null);
  const debug = false;

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
      // Create FormData to send the file
      const formData = new FormData();
      formData.append('file', fileObject);

      const response = await axios.post("http://localhost:5001/inference", formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setTokenUsage(response.data.num_input_tokens || 0);
      setRelationshipData(response.data.character_response_text);
      return response.data.graph_data;
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
      const fileContent = await readFileContent(fileObject);
      const queryResultString = await submitQuery(fileContent);
      let cleanedResponse = queryResultString.replace(/```(?:json)?|```/g, "").trim();

      // Handle potential leading/trailing backticks that might remain
      if (cleanedResponse.startsWith('`')) {
        cleanedResponse = cleanedResponse.substring(1);
      }
      if (cleanedResponse.endsWith('`')) {
        cleanedResponse = cleanedResponse.substring(0, cleanedResponse.length - 1);
      }

      const queryResult = JSON.parse(cleanedResponse);

      // Result of Maverick
      if (debug) {
        const queryResult = JSON.parse(`{
          "title": "Romeo and Juliet",
          "summary": "The tragic love story of Romeo and Juliet, two young lovers from feuding families in Verona, who ultimately sacrifice everything for their love.",
          "nodes": [
            { "id": "c1", "name": "Romeo Montague", "val": 1 },
            { "id": "c2", "name": "Juliet Capulet", "val": 2 },
            { "id": "c3", "name": "Friar Laurence", "val": 3 },
            { "id": "c4", "name": "Tybalt", "val": 4 },
            { "id": "c5", "name": "Mercutio", "val": 5 },
            { "id": "c6", "name": "Benvolio", "val": 6 },
            { "id": "c7", "name": "Lord Capulet", "val": 7 },
            { "id": "c8", "name": "Lady Capulet", "val": 8 },
            { "id": "c9", "name": "Lord Montague", "val": 9 },
            { "id": "c10", "name": "Lady Montague", "val": 10 },
            { "id": "c11", "name": "Paris", "val": 11 },
            { "id": "c12", "name": "Nurse", "val": 12 },
            { "id": "c13", "name": "Prince Escalus", "val": 13 },
            { "id": "c14", "name": "Sampson", "val": 14 },
            { "id": "c15", "name": "Gregory", "val": 15 },
            { "id": "c16", "name": "Abram", "val": 16 },
            { "id": "c17", "name": "Balthasar", "val": 17 },
            { "id": "c18", "name": "Peter", "val": 18 },
            { "id": "c19", "name": "Apothecary", "val": 19 },
            { "id": "c20", "name": "Chorus", "val": 20 },
            { "id": "c21", "name": "Friar John", "val": 21 },
            { "id": "c22", "name": "County Paris's Page", "val": 22 }
          ],
          "links": [
            { "source": "c1", "target": "c2", "label": "secretly married to" },
            { "source": "c1", "target": "c5", "label": "close friend of" },
            { "source": "c1", "target": "c6", "label": "cousin and friend of" },
            { "source": "c1", "target": "c4", "label": "sworn enemy of" },
            { "source": "c2", "target": "c7", "label": "daughter of" },
            { "source": "c2", "target": "c8", "label": "daughter of" },
            { "source": "c2", "target": "c12", "label": "nursed by" },
            { "source": "c3", "target": "c1", "label": "married Romeo and Juliet" },
            { "source": "c4", "target": "c8", "label": "nephew of" },
            { "source": "c5", "target": "c13", "label": "kinsman of" },
            { "source": "c5", "target": "c1", "label": "friend of" },
            { "source": "c6", "target": "c9", "label": "nephew of" },
            { "source": "c6", "target": "c1", "label": "cousin and friend of" },
            { "source": "c7", "target": "c2", "label": "father of" },
            { "source": "c7", "target": "c8", "label": "husband of" },
            { "source": "c8", "target": "c2", "label": "mother of" },
            { "source": "c8", "target": "c7", "label": "wife of" },
            { "source": "c8", "target": "c4", "label": "aunt of" },
            { "source": "c9", "target": "c1", "label": "father of" },
            { "source": "c9", "target": "c10", "label": "husband of" },
            { "source": "c10", "target": "c9", "label": "wife of" },
            { "source": "c11", "target": "c2", "label": "suitor of" },
            { "source": "c12", "target": "c2", "label": "nurse of" },
            { "source": "c14", "target": "c7", "label": "servant of" },
            { "source": "c15", "target": "c7", "label": "servant of" },
            { "source": "c16", "target": "c9", "label": "servant of" },
            { "source": "c17", "target": "c1", "label": "servant of" },
            { "source": "c18", "target": "c12", "label": "servant of" },
            { "source": "c1", "target": "c11", "label": "killed by" },
            { "source": "c4", "target": "c5", "label": "killed by" },
            { "source": "c1", "target": "c4", "label": "killed" },
            { "source": "c2", "target": "c1", "label": "loved" },
            { "source": "c2", "target": "c11", "label": "betrothed to" },
            { "source": "c1", "target": "c2", "label": "loved" }
          ]
        }`);
      }

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
        {/* Token Usage Section */}
        {searchComplete && (
          <div className="mt-4 text-center text-lg font-bold text-gray-600">
            <p>Input Tokens: {tokenUsage}</p>
          </div>
        )}
        </div>

        {/* Info Section - Only show when search is complete */}
        {searchComplete && bookData && (
          <div className="space-y-8">
            <div className="bg-white shadow-xl rounded-xl overflow-hidden">
              <div className="md:flex">
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
            <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-6">
                <CharacterGraph graphData={graphData} />
            </div>
          </div>
        )}

        {/* Chat Section - Only show when search is complete */}
        {graphData && relationshipData && (
          <div className="mt-12">
            <ChatInterface relationshipData={relationshipData} />
          </div>
        )}
      </div>
    </div>
  );
}
