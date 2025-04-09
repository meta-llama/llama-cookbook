// ChatInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane } from 'react-icons/fa';
import axios from 'axios';

const ChatInterface = ({ relationshipData }) => {
  const [messages, setMessages] = useState([
    { text: "Hello! I can answer questions about this book. What would you like to know?", sender: "assistant" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { text: userMessage, sender: "user" }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5001/chat', {
        query: userMessage,
        relationship_data: relationshipData,
        chat_history_data: messages
      });

      setMessages(prev => [...prev, { text: response.data.response, sender: "assistant" }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        text: "Sorry, I couldn't process your question. Please try again.",
        sender: "assistant"
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white/80 backdrop-blur-sm shadow-xl rounded-xl p-8 space-y-6">
      <div className="p-4 bg-blue-600 text-white rounded-xl font-medium">
        Book Chat Assistant
      </div>

      {/* Messages container */}
      <div className="h-96 overflow-y-auto p-4 bg-gray-50">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`mb-4 flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3/4 rounded-lg px-4 py-2 ${
                msg.sender === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none'
                  : 'bg-gray-200 text-gray-800 rounded-bl-none'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />

        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 text-gray-800 rounded-lg px-4 py-2 rounded-bl-none flex items-center">
              <div className="dot-typing"></div>
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      <form onSubmit={handleSend} className="border-t border-gray-200 p-4 flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the book..."
          className="flex-grow px-4 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-blue-600 text-white px-4 py-2 rounded-r-lg hover:bg-blue-700 transition-colors disabled:bg-blue-300 flex items-center justify-center"
        >
          <FaPaperPlane />
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
