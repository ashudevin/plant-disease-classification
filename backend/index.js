const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const path = require('path');
const cors = require('cors');


const app = express();
const port = 4000;
app.use(cors());

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/', // Files will be stored in the "uploads" folder
});

// Route to handle file upload and prediction
app.post('/predict', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const plantName = req.body.plant_name || 'Unknown';
    const filePath = req.file.path;

    // Prepare the data for the external prediction API
    const data = new FormData();
    data.append('plant_name', plantName);
    data.append('file', fs.createReadStream(filePath));

    // Configure the request
    const config = {
      method: 'post',
      maxBodyLength: Infinity,
      url: 'http://89.116.20.44:9000/predict/',
      headers: {
        ...data.getHeaders(),
      },
      data,
    };

    // Make the request to the external API
    const response = await axios.request(config);

    // Clean up the uploaded file
    fs.unlinkSync(filePath);

    // Send the prediction result back to the frontend
    res.status(200).json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error processing the request' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
