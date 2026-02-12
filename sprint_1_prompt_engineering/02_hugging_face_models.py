#!/usr/local/bin/python3
import warnings
import argparse
# Load model directly
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline


########################################
# Prepare environment
########################################

# Ignore warnings
warnings.filterwarnings("ignore")


# ======================================
# Hugging Face Models
# ======================================

# --------------------------------------
# Example: Image classification with a pre-trained model
# --------------------------------------

# Use a pipeline as a high-level helper
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
classifier(img)

# Load model directly
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]


########################################
# Examples
########################################

def run_example_1():
    print("--- Example 1: Image detection model (nsfw_image_detection) ---")
    
    # Load model directly
    print("Loading model: Falconsai/nsfw_image_detection")
    try:
        processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
        model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Note: You need to provide a path to an image
    image_path = "path_to_your_image.jpg" 
    print(f"Attempting to open image from: {image_path}")
    print("Please edit the script to point to a valid image file if you want to run inference.")

    try:
        img = Image.open(image_path)
        
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        print(f"Predicted label: {label}")
        
    except FileNotFoundError:
        print("Image file not found. Skipping inference.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

def run_exercise_1():
    print("--- Exercise 1: Text Classification model ---")
    print("This exercise is left for the user to implement based on the notebook.")

def run_exercise_2():
    print("--- Exercise 2: Text-to-Image or Image-to-Text models ---")
    print("This exercise is left for the user to implement based on the notebook.")

def run_exercise_3():
    print("--- Exercise 3: Multi-modal model ---")
    print("This exercise is left for the user to implement based on the notebook.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hugging Face model examples.")
    parser.add_argument("part", nargs="?", choices=["1", "ex1", "ex2", "ex3"], help="Run a specific part (1, ex1, ex2, or ex3)")
    args = parser.parse_args()

    if args.part == "1":
        run_example_1()
    elif args.part == "ex1":
        run_exercise_1()
    elif args.part == "ex2":
        run_exercise_2()
    elif args.part == "ex3":
        run_exercise_3()
    else:
        run_example_1()