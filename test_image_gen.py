#!/usr/bin/env python3
"""
Test script to verify Gemini image generation capabilities.
Run this script to check if your API key has access to image generation.
"""

import os
import sys
import json
from io import BytesIO
from pathlib import Path
import time
import random

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

def check_models():
    """Check which Gemini models are available."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: Google API key not found in environment. Please check your .env file.")
            sys.exit(1)
            
        client = genai.Client(api_key=api_key)
        
        print("Querying available Gemini models...")
        models = client.models.list()
        
        # Filter for different types of models
        flash_models = []
        image_models = []
        thinking_models = []
        
        for model in models:
            model_name = model.name
            if "gemini-2.0-flash" in model_name:
                flash_models.append(model_name)
                
                # Check for image generation models
                if "image" in model_name.lower():
                    image_models.append(model_name)
                
                # Check for thinking models
                if "thinking" in model_name.lower():
                    thinking_models.append(model_name)
        
        # Report findings
        print("\n===== MODEL AVAILABILITY REPORT =====")
        print(f"Found {len(flash_models)} Gemini 2.0 Flash models")
        
        if flash_models:
            print("\nGemini 2.0 Flash models:")
            for model in flash_models:
                print(f"  - {model}")
        
        if image_models:
            print("\n✅ Image generation models available:")
            for model in image_models:
                print(f"  - {model}")
        else:
            print("\n❌ No image generation models found")
            print("Your API key may not have access to image generation capabilities")
        
        if thinking_models:
            print("\n✅ Thinking models available:")
            for model in thinking_models:
                print(f"  - {model}")
        
        # Return the image models for testing
        return {
            "flash_models": flash_models,
            "image_models": image_models,
            "thinking_models": thinking_models
        }
        
    except Exception as e:
        print(f"Error checking models: {e}")
        return {"flash_models": [], "image_models": [], "thinking_models": []}

def generate_test_image(model_name, prompt, output_file):
    """Try to generate an image using the specified model."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        
        print(f"\nAttempting to generate image with model: {model_name}")
        print(f"Prompt: {prompt}")
        
        # Show a thinking animation
        for i in range(5):
            time.sleep(0.5)
            dots = "." * (i % 3 + 1)
            spaces = " " * (2 - i % 3)
            print(f"Generating{dots}{spaces}", end="\r", flush=True)
        print("")
        
        # Call the API
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Image"],
                temperature=0.4,
            )
        )
        
        # Check for image in the response
        image_data = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                break
        
        if image_data:
            # Save the image
            image = Image.open(BytesIO(image_data))
            image.save(output_file)
            print(f"✅ Success! Image saved to {output_file}")
            return True
        else:
            print("❌ Error: API returned a response but no image data was found")
            
            # If there's text, print it as it might contain error information
            if hasattr(response.candidates[0], 'text'):
                print(f"Text response: {response.candidates[0].text}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        
        # Special handling for common errors
        error_str = str(e).lower()
        if "content" in error_str and "filter" in error_str:
            print("The content was filtered by safety systems. Try a different prompt.")
        elif "modalities" in error_str:
            print("This model doesn't support image generation.")
            print("Your API key might not have access to image generation models.")
        elif "rate" in error_str and "limit" in error_str:
            print("Rate limit exceeded. Try again later or reduce request frequency.")
        
        return False

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output/test_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check available models
    models = check_models()
    
    # If no image models are found, exit
    if not models["image_models"]:
        print("\nNo image generation models found. Cannot perform image generation test.")
        return False
    
    # Try to generate an image with each available image model
    test_successful = False
    
    # Create a few different prompts to try
    test_prompts = [
        "A vibrant landscape with mountains and a sunset sky.",
        "An abstract digital artwork with geometric shapes in blue and purple.",
        "A cute cartoon robot with big eyes on a simple white background.",
        "A futuristic cityscape at night with neon lights."
    ]
    
    # Try each image model
    for model_name in models["image_models"]:
        # Try with a few different prompts
        for i, prompt in enumerate(test_prompts):
            output_file = output_dir / f"test_image_{i+1}_{model_name.split('/')[-1]}.png"
            
            # If we succeed with any prompt, mark the test as successful
            if generate_test_image(model_name, prompt, output_file):
                test_successful = True
                break
        
        # If we got a successful generation, we can stop testing
        if test_successful:
            break
    
    if test_successful:
        print("\n✅ IMAGE GENERATION TEST SUCCESSFUL!")
        print("Your API key has access to image generation capabilities.")
        print(f"Check the {output_dir} directory to see your generated images.")
        return True
    else:
        print("\n❌ IMAGE GENERATION TEST FAILED")
        print("None of the image models were able to generate images.")
        print("This could be due to:")
        print("1. Your API key doesn't have access to image generation")
        print("2. The models are still in limited preview")
        print("3. The prompts were filtered by safety systems")
        return False

if __name__ == "__main__":
    # Run the test
    main()