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

PLANT_CLASS_INDICES = {
    "apple": [0, 1],
    "cherry": [2, 3],
    "grape": [4, 5, 6, 7],
    "peach": [8, 9],
    "potato": [10, 11, 12],
    "strawberry": [13, 14],
    "tomato": [15, 16, 17]
}

# Disease information mapping
DISEASE_INFO = {
    "Apple___Cedar_apple_rust": {
        "Symptoms": [
            "Yellow-orange spots on leaves",
            "Swelling on twigs and branches",
            "Defoliation in severe cases"
        ],
        "Causes": [
            "Fungal infection by Gymnosporangium juniperi-virginianae",
            "Spread by wind from juniper trees",
            "Prolonged wet conditions"
        ],
        "Prevention": [
            "Prune infected branches on junipers",
            "Apply fungicides during early spring",
            "Plant resistant apple varieties"
        ]
    },
    "Apple___healthy": {
        "Symptoms": [
            "Bright green leaves",
            "No signs of disease or pests",
            "Uniformly developed fruits"
        ],
        "Causes": [
            "Optimal growing conditions",
            "Balanced fertilization and irrigation",
            "Absence of pathogens"
        ],
        "Prevention": [
            "Maintain regular monitoring",
            "Use preventive sprays if necessary",
            "Ensure proper air circulation around trees"
        ]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "Symptoms": [
            "White powdery coating on leaves",
            "Distorted or stunted leaf growth",
            "Premature leaf drop"
        ],
        "Causes": [
            "Fungal infection by Podosphaera clandestina",
            "High humidity and poor air circulation",
            "Overcrowded plantings"
        ],
        "Prevention": [
            "Prune trees for better air circulation",
            "Apply sulfur or fungicides",
            "Avoid overwatering"
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "Symptoms": [
            "Dark green healthy leaves",
            "Bright red fruit without blemishes",
            "Vigorous plant growth"
        ],
        "Causes": [
            "Proper sunlight and soil nutrients",
            "Absence of fungal or pest attacks",
            "Regular pruning and care"
        ],
        "Prevention": [
            "Follow a balanced fertilization routine",
            "Inspect for early signs of disease",
            "Maintain proper spacing between trees"
        ]
    },
    "Grape___Black_rot": {
        "Symptoms": [
            "Brown circular spots on leaves",
            "Black, shriveled berries",
            "Lesions on shoots and tendrils"
        ],
        "Causes": [
            "Fungal infection by Guignardia bidwellii",
            "Prolonged wet conditions",
            "Infected plant debris"
        ],
        "Prevention": [
            "Remove and destroy infected plant parts",
            "Apply fungicides before and after rainfall",
            "Practice crop rotation"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "Symptoms": [
            "Interveinal leaf discoloration",
            "Dark streaks on wood",
            "Berries shrivel and dry"
        ],
        "Causes": [
            "Complex fungal pathogens (e.g., Phaeoacremonium spp.)",
            "Infection through pruning wounds",
            "Poor vineyard sanitation"
        ],
        "Prevention": [
            "Seal pruning wounds",
            "Use disease-free planting materials",
            "Remove and burn infected vines"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "Symptoms": [
            "Brown necrotic spots on leaves",
            "Leaf curling and wilting",
            "Reduced photosynthesis"
        ],
        "Causes": [
            "Fungal infection by Isariopsis clavispora",
            "High humidity and wet conditions",
            "Overcrowded vines"
        ],
        "Prevention": [
            "Ensure proper spacing between plants",
            "Apply protective fungicides",
            "Improve vineyard drainage"
        ]
    },
    "Grape___healthy": {
        "Symptoms": [
            "Green, fully expanded leaves",
            "Healthy clusters of berries",
            "No visible signs of stress or infection"
        ],
        "Causes": [
            "Proper pruning and care",
            "Adequate nutrient supply",
            "Disease-free environment"
        ],
        "Prevention": [
            "Maintain regular vine management",
            "Monitor for early signs of disease",
            "Use preventive sprays when necessary"
        ]
    },
    "Peach___Bacterial_spot": {
        "Symptoms": [
            "Dark, water-soaked spots on leaves",
            "Fruit with pitted lesions",
            "Yellowing and leaf drop"
        ],
        "Causes": [
            "Bacterial infection by Xanthomonas campestris",
            "Wet and humid conditions",
            "Infected pruning tools or debris"
        ],
        "Prevention": [
            "Use resistant peach varieties",
            "Sanitize pruning tools",
            "Apply copper-based bactericides"
        ]
    },
    "Peach___healthy": {
        "Symptoms": [
            "Lush green leaves",
            "Firm, unblemished fruits",
            "Consistent growth"
        ],
        "Causes": [
            "Good soil health",
            "Timely watering and fertilization",
            "Absence of pests and diseases"
        ],
        "Prevention": [
            "Monitor tree health regularly",
            "Follow proper irrigation schedules",
            "Ensure balanced nutrient application"
        ]
    },
    "Potato___Early_blight": {
        "Symptoms": [
            "Dark spots with concentric rings on leaves",
            "Leaf yellowing and premature drop",
            "Brown sunken lesions on tubers"
        ],
        "Causes": [
            "Fungal infection by Alternaria solani",
            "High humidity and warm temperatures",
            "Infected plant debris"
        ],
        "Prevention": [
            "Remove infected plant debris",
            "Apply fungicides early in the season",
            "Practice crop rotation"
        ]
    },
    "Potato___Late_blight": {
        "Symptoms": [
            "Water-soaked lesions on leaves",
            "Rapid browning and rotting of foliage",
            "Tubers develop firm brown lesions"
        ],
        "Causes": [
            "Fungal-like organism Phytophthora infestans",
            "Cool and wet weather",
            "Spores spread by wind or water"
        ],
        "Prevention": [
            "Plant resistant varieties",
            "Use fungicides preventively",
            "Avoid overhead irrigation"
        ]
    },
    "Potato___healthy": {
        "Symptoms": [
            "Green, healthy foliage",
            "Uniformly developed tubers",
            "No signs of pest or disease damage"
        ],
        "Causes": [
            "Proper soil preparation",
            "Balanced watering and fertilization",
            "Regular pest and disease monitoring"
        ],
        "Prevention": [
            "Maintain good soil health",
            "Inspect regularly for early issues",
            "Avoid planting in infected fields"
        ]
    },
    "Strawberry___Leaf_scorch": {
        "Symptoms": [
            "Irregular brown spots on leaves",
            "Leaf curling and drying",
            "Reduced plant vigor"
        ],
        "Causes": [
            "Fungal infection by Diplocarpon earlianum",
            "Overhead watering and high humidity",
            "Infected plant residues"
        ],
        "Prevention": [
            "Remove and destroy infected leaves",
            "Apply fungicides as needed",
            "Avoid overhead watering"
        ]
    },
    "Strawberry___healthy": {
        "Symptoms": [
            "Vibrant green leaves",
            "Bright red fruit with no blemishes",
            "Strong, vigorous growth"
        ],
        "Causes": [
            "Optimal care and environment",
            "Regular pest control",
            "Adequate sunlight and nutrients"
        ],
        "Prevention": [
            "Inspect plants frequently",
            "Maintain proper irrigation schedules",
            "Provide balanced fertilizers"
        ]
    },
    "Tomato___Early_blight": {
        "Symptoms": [
            "Brown lesions with concentric rings on leaves",
            "Premature leaf drop",
            "Dark lesions on stems"
        ],
        "Causes": [
            "Fungal infection by Alternaria solani",
            "Wet and warm weather",
            "Contaminated seeds or soil"
        ],
        "Prevention": [
            "Use certified disease-free seeds",
            "Practice crop rotation",
            "Apply fungicides early"
        ]
    },
    "Tomato___Late_blight": {
        "Symptoms": [
            "Water-soaked spots on leaves",
            "Blackened stems",
            "Rotting fruit with firm dark patches"
        ],
        "Causes": [
            "Fungal-like organism Phytophthora infestans",
            "Cool, wet conditions",
            "Infected plant debris"
        ],
        "Prevention": [
            "Apply preventive fungicides",
            "Plant resistant tomato varieties",
            "Ensure good drainage"
        ]
    },
    "Tomato___healthy": {
        "Symptoms": [
            "Bright green foliage",
            "Healthy, ripe fruits",
            "No visible signs of disease or pests"
        ],
        "Causes": [
            "Proper watering and fertilization",
            "Disease-free environment",
            "Regular monitoring and care"
        ],
        "Prevention": [
            "Maintain consistent care routine",
            "Inspect plants regularly",
            "Avoid overwatering"
        ]
    }
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
    
    # Fetch disease information
    disease_info = DISEASE_INFO.get(predicted_class, {
        "Symptoms": ["No data available"],
        "Causes": ["No data available"],
        "Prevention": ["No data available"]
    })
    
    return {
        'plant': plant_name,
        'class': predicted_class,
        'confidence': float(confidence),
        'symptoms': disease_info["Symptoms"],
        'causes': disease_info["Causes"],
        'prevention': disease_info["Prevention"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
