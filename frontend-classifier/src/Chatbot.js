import React, { useState, useEffect } from "react";
import axios from "axios";

const ChatBot = ({ isOpen, toggleChat }) => {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Set an initial message when the chatbot opens
  useEffect(() => {
    if (isOpen) {
      setChatHistory([
        { role: "assistant", content: "Hello, I’m your farmer friend. How can I assist you with plant diseases today?" }
      ]);
    }
  }, [isOpen]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Add user's message to chat history
    setChatHistory((prevHistory) => [...prevHistory, { role: "user", content: message }]);
    
    try {
      setIsLoading(true);

      // Make API request to OpenAI
      const response = await axios.post(
        "https://api.openai.com/v1/chat/completions",
        {
          model: "gpt-4", // Latest GPT model
          messages: [
            { role: "system", content: "You are a helpful assistant." },
            ...chatHistory,
            { role: "user", content: message }
          ]
        },
        {
          headers: {
            "Authorization": `Bearer ${process.env.REACT_APP_OPENAI_API_KEY}`,
            "Content-Type": "application/json"
          }
        }
      );

      const botResponse = response.data.choices[0].message.content;

      // Add OpenAI's response to the chat history
      setChatHistory((prevHistory) => [...prevHistory, { role: "assistant", content: botResponse }]);
      setMessage(""); // Clear input field
    } catch (error) {
      console.error("Error during OpenAI request:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="chatbot-container">
      <div className="chat-history">
        {chatHistory.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong> {msg.content}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask me about plant diseases..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
      {isLoading && <div>Loading...</div>}
    </div>
  );
};

export default ChatBot;
