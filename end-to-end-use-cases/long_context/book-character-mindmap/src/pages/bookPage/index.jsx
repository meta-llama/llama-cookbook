import React, { useEffect, useState } from "react";
import { FaBook, FaHome, FaMagic } from "react-icons/fa";
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

  useEffect(() => {
    if (debug) {
      setRelationshipData(`**Character:** Alex Chen
      * **Relationship with Emily Patel:** Described as a deep and unwavering bond. They share a history of trust and mutual support, evident from their ability to fall back into their old rhythms without hesitation. The text states, "they all knew that things had changed" but also "their connection remained strong," indicating that their relationship has held up over time. This connection is built on a foundation of trust, as evident from Emily being Alex's "best friend and confidant." However, the text does suggest that this friendship has become "more complicated," implying that there might be underlying tensions or challenges in their relationship.
      * **Relationship with Jake Lee:** No direct information about Alex and Jake's relationship is provided, except that they seem to share a comfortable banter and camaraderie, as seen in their "laughing and joking like they used to." This suggests a friendly and familiar rapport, but no deeper emotional nuances are described.
      * **Relationship with Sarah Kim:** The text does not explicitly describe Alex's relationship with Sarah, only mentioning that their connection remains strong despite the fact that they've grown apart. This implies a deep affection, but the text does not provide specific details about their interaction or the nature of their bond.
      * **Relationship with Jake Lee (secondary context):** Not explicitly mentioned, but the text does state that Jake was in a relationship with Sarah, creating a shared connection with Alex through his friend. This adds a layer of interconnectedness among the group, but does not directly describe Alex-Jake's relationship.

  **Character:** Emily Patel
      * **Relationship with Alex Chen:** See above. Emily is described as Alex's best friend and confidant, and their bond has become "more complicated," suggesting underlying tensions or challenges. She seems to have a deep understanding of Alex and is able to offer emotional support, as seen in the mention of Alex's drive for success and his cost of ambition.
      * **Relationship with Jake Lee:** The text does not provide direct information about Emily-Jake's relationship. However, Emily's affectionate interaction with Alex and the comfortable banter between them suggest a familiar, close relationship. Emily's connection to Jake is also emotional, as she is actively engaged with his partner Sarah.
      * **Relationship with Sarah Kim:** No explicit information is provided about Emily's relationship with Sarah. However, her support for Alex in his endeavors and her presence in his life suggest a caring and nurturing nature, potentially suggesting a good rapport with Sarah as well.
      * **Relationship with Jake Lee (secondary context):** As mentioned earlier, Jake and Sarah's relationship is secondary to the main relationships described, but it does not provide direct information about Emily-Jake's relationship. However, given the group's dynamics, it can be inferred that Emily's emotional involvement with Jake might imply some level of emotional investment or familiarity in his life.

  **Character:** Jake Lee
      * **Relationship with Alex Chen:** The text does not provide explicit information about their relationship. However, their comfortable banter and familiar interaction suggest a friendly rapport, indicating a level of ease and comfort in each other's company.
      * **Relationship with Emily Patel:** As mentioned earlier, the text describes a "deep bond" between Emily and Alex, and it can be inferred that Jake is also included in this group, allowing for a level of emotional proximity. However, direct information about Jake-Emily's relationship is limited.
      * **Relationship with Sarah Kim:** Jake and Sarah are partners in life, which implies a romantic relationship and potentially deep emotional connection. The text does not explicitly describe their relationship, but it suggests a level of intimacy and trust.
      * **Relationship with Alex Chen (secondary context):** As mentioned earlier, Jake's relationship with Sarah is not directly connected to Alex, but through his partner, Jake maintains a level of friendship with Alex, suggesting he is part of the group's social fabric.

  **Character:** Sarah Kim
      * **Relationship with Alex Chen:** No explicit information about Alex-Sarah's relationship is provided. However, her emotional involvement with Alex and her close bond with Emily suggest a caring nature that could apply to her connection with Alex.
      * **Relationship with Emily Patel:** As mentioned earlier, Emily is described as Sarah's confidant and possibly friend, indicating a level of emotional closeness. The text does not explicitly describe their relationship, but it suggests that they share a connection.
      * **Relationship with Jake Lee:** As mentioned earlier, Sarah and Jake are romantic partners, and this information hints at a deep emotional connection. Sarah's interaction with Emily also implies a level of familiarity and care.
      * **Relationship with Alex Chen (secondary context):** As mentioned earlier, Sarah is mentioned as a part of Alex's life through her relationship with Jake, suggesting she is a valued member of the social circle that includes Alex.`)
    setGraphData({"nodes": [
        { "id": "n1", "name": "Alex Chen", "val": 1 },
        { "id": "n2", "name": "Emily Patel", "val": 2 },
        { "id": "n3", "name": "Jake Lee", "val": 3 },
        { "id": "n4", "name": "Sarah Kim", "val": 4 }
      ],
      "links": [
        { "source": "n1", "target": "n2", "label": "best friend and confidant" },
        { "source": "n2", "target": "n1", "label": "childhood friend and confidant, bond became more complicated" },
        { "source": "n2", "target": "n3", "label": "possible secondary connection, not directly described" },
        { "source": "n2", "target": "n4", "label": "possible secondary connection, not directly described" },
        { "source": "n3", "target": "n1", "label": "friendly rapport, comfortable banter" },
        { "source": "n3", "target": "n2", "label": "friendly rapport, comfortable banter" },
        { "source": "n3", "target": "n4", "label": "possibly familiar through her relationship with Sarah" },
        { "source": "n4", "target": "n1", "label": "caring and nurturing, possible secondary connection" },
        { "source": "n4", "target": "n2", "label": "caring and nurturing, possible secondary connection" }
      ]})
    };
  }, [debug]);

  const submitBookQuery = async (query) => {
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
      const queryResult = await submitBookQuery();

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
                <FaMagic className={`${isLoading ? "animate-spin" : ""}`} />
                <span>{isLoading ? "Generating..." : "Visualize"}</span>
              </button>
            </form>
          </div>
          <p className="mt-4 text-center text-sm text-gray-600">
            Visualize any book or movie to explore character relationships
          </p>
        {/* Token Usage Section */}
        {searchComplete && (
          <div className="mt-4 text-center text-lg font-bold text-gray-600">
            <p>Input Tokens: {tokenUsage}</p>
          </div>
        )}
        </div>

        {/* Graph Section - Only show when search is complete */}
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
