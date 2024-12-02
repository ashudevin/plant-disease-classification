import React from "react";

const DiseaseInfo = ({ data }) => {
  return (
    <div className="info-cards">
      {data.symptoms && (
        <div className="card">
          <h3>Symptoms</h3>
          <ul>
            {data.symptoms.map((symptom, index) => (
              <li key={index}>{symptom}</li>
            ))}
          </ul>
        </div>
      )}
      {data.causes && (
        <div className="card">
          <h3>Causes</h3>
          <ul>
            {data.causes.map((cause, index) => (
              <li key={index}>{cause}</li>
            ))}
          </ul>
        </div>
      )}
      {data.prevention && (
        <div className="card">
          <h3>Prevention</h3>
          <ul>
            {data.prevention.map((prevention, index) => (
              <li key={index}>{prevention}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DiseaseInfo;
