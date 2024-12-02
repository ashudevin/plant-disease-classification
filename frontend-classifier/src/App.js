import React, { useState } from "react";
import Home from "./home";
import axios from "axios";
import ChatBot from "./Chatbot"; // Import the ChatBot component

function App() {
  const [diseaseData, setDiseaseData] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false); // State to control chatbot visibility

  const handlePredict = async (imageFile) => {
    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await axios.post("http://89.116.20.44:9000/predict/", formData);
      setDiseaseData(response.data);
    } catch (error) {
      console.error("Error during prediction:", error);
    }
  };

  const toggleChat = () => {
    setIsChatOpen((prev) => !prev); // Toggle chat visibility
  };

  return (
    <div className="App">
      <Home diseaseData={diseaseData} onPredict={handlePredict} />

      {/* Button to toggle the chatbot */}
      <button className="chatbot-toggle" onClick={toggleChat}>
        {isChatOpen ? "Close Chat" : "Open Chat"}
      </button>

      {/* The ChatBot component is conditionally rendered based on isChatOpen */}
      <ChatBot isOpen={isChatOpen} toggleChat={toggleChat} />
    </div>
  );
}

export default App;
