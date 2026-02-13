#!/usr/local/bin/python3
import argparse
import warnings
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

warnings.filterwarnings("ignore")

MODEL_ID = "Falconsai/nsfw_image_detection"
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "cool-cat.jpg"


def classify_with_model(image_path: Path) -> str:
    """Mirror the notebook flow using AutoImageProcessor + AutoModelForImageClassification."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    img = Image.open(image_path)

    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


def classify_with_pipeline(image_path: Path) -> list[dict]:
    """Optional high-level helper shown in the notebook."""
    classifier = pipeline("image-classification", model=MODEL_ID)
    return classifier(str(image_path))


def run_example_1(image_path: Path, use_pipeline: bool) -> None:
    print("--- Example 1: Image detection model (nsfw_image_detection) ---")
    print(f"Model: {MODEL_ID}")
    print(f"Image: {image_path}")

    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return

    try:
        label = classify_with_model(image_path)
        print(f"Predicted label (direct model): {label}")

        if use_pipeline:
            result = classify_with_pipeline(image_path)
            print(f"Pipeline output: {result}")
    except Exception as exc:
        print(f"Failed to run inference: {exc}")


def run_exercise_1() -> None:
    print("--- Exercise 1: Text Classification model ---")
    print("No exercise implementation present in the reference notebook.")


def run_exercise_2() -> None:
    print("--- Exercise 2: Text-to-Image or Image-to-Text models ---")
    print("No exercise implementation present in the reference notebook.")


def run_exercise_3() -> None:
    print("--- Exercise 3: Multi-modal model ---")
    print("No exercise implementation present in the reference notebook.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hugging Face model examples.")
    parser.add_argument(
        "part",
        nargs="?",
        choices=["1", "ex1", "ex2", "ex3"],
        default="1",
        help="Run a specific part (1, ex1, ex2, or ex3)",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to input image. Defaults to cool-cat.jpg in this folder.",
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Also run Hugging Face pipeline() classification.",
    )
    args = parser.parse_args()

    if args.part == "1":
        run_example_1(Path(args.image), args.use_pipeline)
    elif args.part == "ex1":
        run_exercise_1()
    elif args.part == "ex2":
        run_exercise_2()
    elif args.part == "ex3":
        run_exercise_3()
