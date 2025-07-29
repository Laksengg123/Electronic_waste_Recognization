import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# E-waste information database
e_waste_info = {
    "keyboard": {
        "type": "Non-biodegradable",
        "environmental_impact": "Plastic and electronic components can take decades to decompose.",
        "recycling_tips": "Recycle at certified e-waste centers or donate working units.",
        "recycling_location": "Local e-waste collection points or electronic stores.",
        "decomposition_time": "50-100 years",
        "recycling_cost": "₹500-₹800 depending on the location."
    },
    "mobile_phone": {
        "type": "Non-biodegradable",
        "environmental_impact": "Can release harmful chemicals into soil and water if not recycled properly.",
        "recycling_tips": "Donate old phones or recycle at certified e-waste centers.",
        "recycling_location": "Check local e-waste collection points or electronics stores.",
        "decomposition_time": "20-100 years",
        "recycling_cost": "₹1000 per unit depending on location."
    },
    "washing_machine": {
        "type": "Non-biodegradable",
        "environmental_impact": "Large appliances contribute to landfill overflow if not recycled.",
        "recycling_tips": "Recycle at large appliance recycling centers.",
        "recycling_location": "Authorized appliance recycling points.",
        "decomposition_time": "100-200 years",
        "recycling_cost": "₹2000-₹5000 depending on the model."
    },
    "screen": {
        "type": "Non-biodegradable",
        "environmental_impact": "Glass and electronic components can harm the environment.",
        "recycling_tips": "Recycle at e-waste centers or return to manufacturers.",
        "recycling_location": "E-waste collection points or manufacturer programs.",
        "decomposition_time": "50-100 years",
        "recycling_cost": "₹800-₹1500 depending on the size."
    },
    "pcb": {
        "type": "Non-biodegradable",
        "environmental_impact": "Contains heavy metals harmful to soil and water.",
        "recycling_tips": "Recycle through specialized e-waste recyclers.",
        "recycling_location": "Certified e-waste recyclers.",
        "decomposition_time": "100-1000 years",
        "recycling_cost": "₹1000-₹2000 depending on the complexity."
    },
    "printer": {
        "type": "Non-biodegradable",
        "environmental_impact": "Contributes to e-waste with plastic and electronic components.",
        "recycling_tips": "Recycle at e-waste centers or donate if functional.",
        "recycling_location": "Certified e-waste recycling centers.",
        "decomposition_time": "50-100 years",
        "recycling_cost": "₹1000-₹3000 depending on the model."
    },
    "battery": {
        "type": "Non-biodegradable",
        "environmental_impact": "Leaches toxic chemicals into the soil and water.",
        "recycling_tips": "Recycle at battery-specific recycling points.",
        "recycling_location": "Battery collection points or e-waste centers.",
        "decomposition_time": "500-1000 years",
        "recycling_cost": "₹50-₹200 per unit."
    },
    "mouse": {
        "type": "Non-biodegradable",
        "environmental_impact": "Plastic and small electronic components contribute to e-waste.",
        "recycling_tips": "Recycle at e-waste centers or donate if functional.",
        "recycling_location": "Local e-waste collection centers.",
        "decomposition_time": "50-100 years",
        "recycling_cost": "₹200-₹500 per unit."
    },
    "television": {
        "type": "Non-biodegradable",
        "environmental_impact": "Glass and electronic components are harmful if disposed of improperly.",
        "recycling_tips": "Recycle at certified e-waste centers.",
        "recycling_location": "Authorized e-waste collection points.",
        "decomposition_time": "50-100 years",
        "recycling_cost": "₹1500-₹5000 depending on the size."
    }
}

# Load MobileNetV2 model
def load_model():
    """
    Loads the MobileNetV2 pre-trained model.
    """
    try:
        model = MobileNetV2(weights='imagenet')
        print("MobileNetV2 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# Recognize the object in the image
def recognize_image(image_path, model):
    """
    Recognizes the object in the given image using MobileNetV2.
    """
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)
        label = decoded_predictions[0][0][1]

        # Correct misidentified labels
        correction_map = {
            "space_bar": "keyboard"
        }
        return correction_map.get(label, label)

    except Exception as e:
        return f"Error recognizing image: {e}"

# Display e-waste details
def display_details(label):
    """
    Formats and displays e-waste details in a structured manner.
    """
    details = e_waste_info.get(label.lower(), None)
    if not details:
        print(f"No e-waste information found for: {label}")
    else:
        print(f"\nThe image is recognized as: {label}")
        print("\nE-Waste Details:")
        print(f"Type: {details['type']}")
        print(f"Environmental Impact: {details['environmental_impact']}")
        print(f"Recycling Tips: {details['recycling_tips']}")
        print(f"Recycling Location: {details['recycling_location']}")
        print(f"Decomposition Time: {details['decomposition_time']}")
        print(f"Recycling Cost: {details['recycling_cost']}")

# Main script
if __name__ == "__main__":
    model = load_model()

    # Get user input for the image file path
    image_path = input("Please enter the file path of the image for recognition: ").strip()

    if not os.path.exists(image_path):
        print("Invalid file path. Exiting.")
        exit(1)

    # Recognize the object in the uploaded image
    label = recognize_image(image_path, model)

    if "Error" in label:
        print(f"Image recognition failed: {label}")
        exit(1)

    # Display the fetched details
    display_details(label)
