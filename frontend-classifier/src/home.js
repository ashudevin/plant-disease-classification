import React, { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import "./App.css";

const Home = () => {
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [symptoms, setSymptoms] = useState(null);
  const [causes, setCauses] = useState(null);
  const [prevention, setPrevention] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedPlant, setSelectedPlant] = useState("");

  const plants = [
    "Apple",
    "Grape",
    "Cherry",
    "Peach",
    "Strawberry",
    "Tomato",
    "Potato",
  ];

  const onDrop = async (acceptedFiles) => {
    if (!selectedPlant) {
      alert("Please select a plant name.");
      return;
    }

    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("plant_name", selectedPlant);

    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);

    setIsLoading(true);
    try {
      const response = await axios.post("https://plant-disease-classification-8alj.onrender.com/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.class);
      setConfidence((response.data.confidence * 100).toFixed(2));
      setSymptoms(response.data.symptoms);
      setCauses(response.data.causes);
      setPrevention(response.data.prevention);
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: "image/*",
    multiple: false,
  });

  return (
    <div className="home">
      <div className="header-container">
        <h1 className="header">Plant Disease Classifier</h1>
      </div>

      {/* Plant Name Dropdown with Label */}
      <div className="select-container">
        <select
          id="plantName"
          className="plant-select"
          value={selectedPlant}
          onChange={(e) => setSelectedPlant(e.target.value)}
        >
          <option value="">--Select Plant Type--</option>
          {plants.map((plant) => (
            <option key={plant} value={plant}>
              {plant}
            </option>
          ))}
        </select>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "active" : ""}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the image here...</p>
        ) : (
          <p>Drag & drop an image here, or click to select one</p>
        )}
      </div>

      {/* Image Preview */}
      {imagePreview && (
        <div className="image-preview">
          <img src={imagePreview} alt="Preview" className="preview-img" />
        </div>
      )}

      {/* Loading spinner or result */}
      {isLoading ? (
        <div className="loading-spinner"></div>
      ) : (
        prediction && (
          <div className="result-container">
            <div className="result-card">
              <p>
                Prediction: <strong>{prediction}</strong>
              </p>
              <p>
                Confidence: <strong>{confidence}%</strong>
              </p>
            </div>

            {/* Display Symptoms, Causes, Prevention */}
            <div className="info-cards">
              {symptoms && (
                <div className="card">
                  <h3>Symptoms</h3>
                  <ul>
                    {symptoms.map((symptom, index) => (
                      <li key={index}>{symptom}</li>
                    ))}
                  </ul>
                </div>
              )}

              {causes && (
                <div className="card">
                  <h3>Causes</h3>
                  <ul>
                    {causes.map((cause, index) => (
                      <li key={index}>{cause}</li>
                    ))}
                  </ul>
                </div>
              )}

              {prevention && (
                <div className="card">
                  <h3>Prevention</h3>
                  <ul>
                    {prevention.map((prevent, index) => (
                      <li key={index}>{prevent}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )
      )}
    </div>
  );
};

export default Home;
