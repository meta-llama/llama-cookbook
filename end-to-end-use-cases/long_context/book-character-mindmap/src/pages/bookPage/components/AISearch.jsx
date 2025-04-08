import { useState } from "react";
import { FaSearch, FaSpinner } from "react-icons/fa";
import axios from "axios";
import Together from "together-ai";

export default function AISearch({ bookTitle, onQueryResponse }) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:5001/query", {
        query: `About ${bookTitle}: ${query}`,
      });

      setResponse(response.data.response);
      onQueryResponse?.(response.data);
    } catch (error) {
      console.error("Error in AI search:", error);
      setError("Failed to get response. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-lg p-6 space-y-4">
      <h2 className="text-xl font-semibold text-gray-800">AI-Powered Search</h2>
      <div className="space-y-4">
        <input
          type="text"
          placeholder={`Ask about ${bookTitle}'s character relationships...`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full px-4 py-2 rounded-md border-gray-300
                   focus:border-blue-500 focus:ring focus:ring-blue-200
                   transition duration-200"
          disabled={isLoading}
        />
        <button
          onClick={handleSearch}
          disabled={isLoading || !query.trim()}
          className="w-full bg-blue-500 hover:bg-blue-600
                   text-white font-semibold py-2 px-4 rounded-md
                   transition duration-200 flex items-center justify-center
                   disabled:bg-blue-300 space-x-2"
        >
          {isLoading ? <FaSpinner className="animate-spin" /> : <FaSearch />}
          <span>{isLoading ? "Searching..." : "Search"}</span>
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 rounded-md text-red-600">{error}</div>
      )}

      {response && !error && (
        <div className="mt-4 p-4 bg-gray-50 rounded-md">
          <h3 className="font-semibold text-gray-800 mb-2">AI Response:</h3>
          <p className="text-gray-600 whitespace-pre-wrap">{response}</p>
        </div>
      )}
    </div>
  );
}
