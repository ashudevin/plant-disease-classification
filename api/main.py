from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Define allowed origins for CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and define class names
MODEL = tf.keras.models.load_model("./plant_disease_classification_model.h5")

FULL_CLASS_NAMES = [
    'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy'
]

# Mapping plant names to the relevant indices in the model's output
PLANT_CLASS_INDICES = {
    "apple": [0, 1],
    "cherry": [2, 3],
    "grape": [4, 5, 6, 7],
    "peach": [8, 9],
    "potato": [10, 11, 12],
    "strawberry": [13, 14],
    "tomato": [15, 16, 17]
}

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Function to preprocess the image file
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure it's RGB format
    image = image.resize((256, 256))  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize
    return image

@app.post("/predict")
async def predict(
    plant_name: str = Form(...),  # Retrieve the plant name from the form data
    file: UploadFile = File(...)  # Retrieve the image file
):
    plant_name = plant_name.lower()  # Normalize plant name for matching
    # Check if the plant name exists in PLANT_CLASS_INDICES
    if plant_name not in PLANT_CLASS_INDICES:
        return {"error": f"Plant '{plant_name}' not recognized. Please check the plant name."}
    
    # Load and preprocess the image
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # Add batch dimension
    except Exception as e:
        return {"error": f"Error processing the image: {e}"}
    
    # Perform prediction
    try:
        predictions = MODEL.predict(img_batch)[0]  # Model's output for one image
        indices = PLANT_CLASS_INDICES[plant_name]  # Get indices for the specified plant
        plant_predictions = [predictions[i] for i in indices]  # Filter relevant predictions
        
        # Get the predicted class within the plant-specific subset
        predicted_index = np.argmax(plant_predictions)
        predicted_class = FULL_CLASS_NAMES[indices[predicted_index]]
        confidence = plant_predictions[predicted_index]
    except Exception as e:
        return {"error": f"Error during prediction: {e}"}
    
    return {
        'plant': plant_name,
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
    