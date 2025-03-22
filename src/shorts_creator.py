#!/usr/bin/env python3
"""
YouTube Shorts Creator - A command-line tool to generate short-form videos
using AI for script, image, and voice generation.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import random
import textwrap
from io import BytesIO
from pathlib import Path

import google.genai as genai
import openai
from dotenv import load_dotenv
from google.genai import types
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip
from PIL import Image

# Load environment variables
load_dotenv()

# Model definitions
THINKING_MODEL = "models/gemini-2.0-flash-thinking-exp"
STANDARD_MODEL = "models/gemini-2.0-flash"

# Initialize API clients
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not google_api_key or not openai_api_key:
        print("Error: API keys not found. Please check your .env file.")
        sys.exit(1)
        
    genai_client = genai.Client(api_key=google_api_key)
    openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
    
    # Check for available Gemini models
    def check_available_models():
        """Check which Gemini models are available, especially thinking models."""
        try:
            print("Checking available Gemini models...")
            models = genai_client.models.list()
            
            # Filter for Gemini 2.0 Flash models
            flash_models = []
            thinking_models = []
            
            for model in models:
                model_name = model.name
                if "gemini-2.0-flash" in model_name:
                    flash_models.append(model_name)
                    
                    # Check for thinking models
                    if "thinking" in model_name.lower():
                        thinking_models.append(model_name)
            
            print(f"Found {len(flash_models)} Gemini 2.0 Flash models")
            
            # Print flash models for reference
            if flash_models:
                print("Available flash models:")
                for model in flash_models:
                    print(f"  - {model}")
            
            # Update thinking model if available
            if thinking_models:
                print("✅ Thinking models available:")
                for model in thinking_models:
                    print(f"  - {model}")
                # Update the global THINKING_MODEL variable
                global THINKING_MODEL
                THINKING_MODEL = thinking_models[0]  # Use the first available thinking model
                print(f"Using thinking model: {THINKING_MODEL}")
            else:
                print("❌ No thinking models found")
                print("Will use standard model for ideation")
                global STANDARD_MODEL
                THINKING_MODEL = STANDARD_MODEL
                
            return flash_models
        except Exception as e:
            print(f"Error checking models: {e}")
            return []
    
    # Run the model check
    available_models = check_available_models()
    
except Exception as e:
    print(f"Error initializing API clients: {e}")
    sys.exit(1)

# Function to generate a readable folder name
def generate_folder_name(concept):
    """Generate a clean, readable folder name from the concept."""
    try:
        # Use a lightweight model for quick generation
        response = genai_client.models.generate_content(
            model="models/gemini-2.0-flash-lite",
            contents=f"""
            Convert this concept into a short, clean folder name (3-5 words max):
            Concept: {concept}
            
            Rules:
            - Use only lowercase letters, numbers, and underscores
            - No spaces or special characters
            - 3-5 words maximum
            - Should be descriptive but concise
            - DO NOT include any explanation, just return the folder name
            
            Examples:
            "The history of space exploration" -> space_exploration_history
            "How do black holes work?" -> black_hole_mechanics
            "Top 10 facts about dinosaurs" -> dinosaur_facts_top10
            
            Folder name:
            """
        )
        folder_name = response.text.strip().lower()
        
        # Sanitize the folder name (remove any remaining special chars)
        import re
        folder_name = re.sub(r'[^\w_]', '', folder_name)
        
        # Limit length
        if len(folder_name) > 50:
            folder_name = folder_name[:50]
            
        return folder_name
    except Exception as e:
        print(f"Error generating folder name: {e}")
        # Fallback to simplified concept
        import re
        simplified = re.sub(r'[^\w_]', '', concept.lower().replace(' ', '_'))
        return simplified[:30]  # Limit length

# Create a unique output directory for each run
timestamp = time.strftime("%Y%m%d_%H%M%S")
folder_name = generate_folder_name(sys.argv[1] if len(sys.argv) > 1 else "shorts_video")
output_dir = Path(f"output/{folder_name}_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Created output directory: {output_dir}")

def api_call_with_backoff(func, *args, max_retries=5, initial_delay=2.0, max_delay=60.0, **kwargs):
    """Make API call with exponential backoff for rate limiting and network issues."""
    delay = initial_delay
    retries = 0
    
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # If this is the last retry, re-raise the exception
            if retries == max_retries - 1:
                print(f"Failed after {retries+1} attempts: {error_type}: {error_str}")
                raise
            
            # Check if this is a rate limit or other retriable error
            is_retriable = any(indicator in error_str for indicator in 
                             ["rate limit", "quota", "429", "too many requests", 
                              "resource exhausted", "capacity", "connection", 
                              "timeout", "server error", "503", "502", "network"])
            
            if is_retriable:
                # Apply jitter to avoid thundering herd
                jitter = random.uniform(0.8, 1.2)
                sleep_time = min(delay * jitter, max_delay)
                
                print(f"API call failed (retry {retries+1}/{max_retries}): {error_type}")
                print(f"Waiting {sleep_time:.1f}s before retrying...")
                
                time.sleep(sleep_time)
                
                # Exponential backoff
                delay = min(delay * 2, max_delay)
                retries += 1
            else:
                # Not a retriable error, re-raise
                raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate YouTube Shorts videos from a concept.")
    
    # Create mutually exclusive group for concept vs from-script
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--concept", help="The video concept or topic")
    input_group.add_argument("--from-script", help="Path to a pre-generated script JSON file")
    
    parser.add_argument("--frames", type=int, default=12, 
                        help="Number of frames/segments to generate (default: 12)")
    parser.add_argument("--voice", default="nova", 
                        choices=["nova", "alloy", "echo", "fable", "onyx", "shimmer"],
                        help="Voice to use for narration (default: nova)")
    
    args = parser.parse_args()
    
    # Handle script-based approach
    if args.from_script:
        try:
            with open(args.from_script, 'r') as f:
                script_data = json.load(f)
                
            # If script doesn't specify concept, use title as concept
            args.concept = script_data.get("concept", script_data.get("title", "Educational Video"))
            
            # Use exact number of frames from the script file
            if "frames" in script_data:
                args.frames = len(script_data["frames"])
                
            # Store the pre-loaded script
            args.script_data = script_data
        except Exception as e:
            print(f"Error loading script file {args.from_script}: {e}")
            sys.exit(1)
    elif not args.concept:
        # For compatibility with older usage patterns where concept was positional
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            args.concept = sys.argv[1]
        else:
            parser.print_help()
            sys.exit(1)
    
    # Add fixed style and duration attributes
    args.style = "clear and direct"
    # Calculate duration based on number of frames (approx. 5 seconds per frame)
    args.duration = args.frames * 5  # Scale duration with number of frames
    return args

def generate_video_concept(concept, style, duration, frames):
    """Generate a creative concept for the video using the thinking model."""
    print("\n" + "="*80)
    print("🎬 GENERATING VIDEO CONCEPT")
    print("="*80)
    print(f"🔍 Topic: {concept}")
    print(f"🎨 Style: {style}")
    print(f"⏱️ Duration: {duration} seconds ({frames} frames)")
    print("-"*80)
    print("💭 Thinking about approach, style, and narrative structure...")
    
    # Animated thinking process
    for i in range(5):
        time.sleep(0.4)
        dots = "." * (i % 4 + 1)
        spaces = " " * (3 - i % 4)
        print(f"🧠 Thinking{dots}{spaces}", end="\r", flush=True)
    
    # Define duration range based on frames
    duration_min = int(duration * 0.8)  # Allow for some flexibility
    duration_max = int(duration * 1.2)

    prompt = f"""
    You are designing a YouTube Shorts video on the topic of "{concept}".
    
    Think deeply about how to present this topic in a compelling way. This will be a {duration_min}-{duration_max} second video
    with {frames} key scenes/frames. Use an {style} style. Consider the best approach for maximum viewer engagement.
    
    Please explore the following aspects in your thinking:
    1. What angle or perspective would be most interesting for this topic?
    2. What narrative structure would work best for a short-form video with {frames} distinct scenes?
    3. What visual style and mood would complement this topic?
    4. What key points must be included to make this informative and engaging?
    5. How can we make this topic immediately grab viewer attention in the first few seconds?
    
    Develop a cohesive concept that guides both the script and visual direction.
    Focus on making this short video memorable, informative, and engaging.
    
    Format your response as a conceptual brief with the following sections:
    - Overall Concept & Angle
    - Visual Style & Mood
    - Key Messaging Points
    - Engagement Strategy
    """
    
    try:
        def generate_with_thinking():
            response = genai_client.models.generate_content(
                model=THINKING_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2  # Lower temperature for more focused thinking
                )
            )
            return response.text
            
        concept_text = api_call_with_backoff(generate_with_thinking)
        
        print("\n" + "-"*80)
        print("✨ VIDEO CONCEPT GENERATED:")
        print("-"*80)
        
        # Format the concept nicely with wrapping
        concept_wrapped = textwrap.fill(concept_text, width=76)
        for line in concept_wrapped.split('\n'):
            print(f"  {line}")
        
        print("="*80 + "\n")
        
        return concept_text
    except Exception as e:
        print(f"Error generating video concept: {e}")
        # Continue with script generation even if concept generation fails
        return f"A compelling {style} video about {concept}"

def generate_script(concept, duration, frames, style, video_concept, pre_loaded_script=None):
    """Generate a video script using Gemini or load a pre-generated script."""
    # If a pre-loaded script is provided, use it
    if pre_loaded_script:
        print(f"Using pre-generated script: {pre_loaded_script.get('title', 'Untitled')}")
        
        # Handle segment vs frame naming
        if "segments" in pre_loaded_script and not "frames" in pre_loaded_script:
            # Convert segments format to frames format
            frames_data = []
            for segment in pre_loaded_script["segments"]:
                frames_data.append({
                    "frame_id": segment.get("segment_id", segment.get("id", len(frames_data) + 1)),
                    "narration": segment["narration"],
                    "image_description": segment["image_description"],
                    "duration_seconds": segment.get("duration_seconds", 5)
                })
            
            script = {
                "title": pre_loaded_script["title"],
                "description": pre_loaded_script.get("description", f"Educational video about {concept}"),
                "frames": frames_data
            }
        else:
            # Script is already in the right format
            script = pre_loaded_script
        
        # Ensure frames have duration_seconds
        for frame in script["frames"]:
            if "duration_seconds" not in frame:
                frame["duration_seconds"] = 5
        
        # Save script to file
        script_path = output_dir / "script.json"
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)
            
        print(f"Script loaded with title: {script['title']}")
        return script
    
    # Otherwise, generate a new script
    print("Generating script based on creative concept...")
    
    # Calculate values based on the desired duration and frames
    seconds_per_frame = max(3, min(10, duration / frames))  # Between 3-10 seconds per frame
    words_per_frame = int(seconds_per_frame * 3)  # Approximately 3 words per second
    
    prompt = f"""
    Create a YouTube Shorts script about "{concept}".
    
    VIDEO REQUIREMENTS:
    - Create exactly {frames} frames/scenes
    - Target approximately {words_per_frame} words per narration
    - Each frame should be around {seconds_per_frame:.1f} seconds
    - Total video length: approximately {duration} seconds
    - Use simple, clear language
    - The image descriptions should be specific and easy to visualize
    
    Format your response as a JSON object with the following structure:
    {{
      "title": "The video title",
      "description": "A short YouTube description",
      "frames": [
        {{
          "frame_id": 1,
          "narration": "Text to be narrated (about {words_per_frame} words)",
          "image_description": "Simple, clear description of what should appear in this frame",
          "duration_seconds": {seconds_per_frame:.1f}
        }},
        // Additional frames as needed (total: {frames})
      ]
    }}
    
    Return ONLY the JSON object.
    """
    
    try:
        def generate_script_content():
            response = genai_client.models.generate_content(
                model=STANDARD_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2  # Lower temperature for more predictable output
                )
            )
            return response.text
            
        script_text = api_call_with_backoff(generate_script_content)
        
        # Sometimes Gemini includes markdown code blocks, try to extract JSON
        if "```json" in script_text:
            script_text = script_text.split("```json")[1].split("```")[0].strip()
        elif "```" in script_text:
            script_text = script_text.split("```")[1].split("```")[0].strip()
            
        script = json.loads(script_text)
        
        # Save script to file
        script_path = output_dir / "script.json"
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)
            
        print(f"Script generated with title: {script['title']}")
        return script
        
    except Exception as e:
        print(f"Error generating script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def generate_image(frame, script_title):
    """Generate an image for a frame using Gemini."""
    frame_id = frame["frame_id"]
    image_path = output_dir / f"frame_{frame_id}.png"
    
    # Skip if image already exists
    if image_path.exists():
        print(f"Image for frame {frame_id} already exists, skipping generation")
        return str(image_path)
    
    print(f"Generating image for frame {frame_id}...")
    
    try:
        # First try standard flash-exp model with multimodal output
        prompt = f"""Create a high-quality image for a YouTube Shorts video titled '{script_title}'
        
        This image illustrates: {frame['narration']}
        
        Based on this description: {frame['image_description']}
        
        Make it vertical (9:16 aspect ratio) with vibrant colors, suitable for YouTube Shorts.
        The image should be exactly 1080×1920 pixels.
        Use bold, clear visuals that will be easily visible on a mobile phone screen."""
        
        # Make the API call similar to the example
        response = genai_client.models.generate_content(
            model="models/gemini-2.0-flash-exp",  # Use standard model with image capabilities
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image'],
                temperature=0.4
            )
        )
        
        # Process the response
        image_data = None
        response_text = None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text is not None:
                response_text = part.text
                print(f"Model response: {response_text[:100]}..." if len(response_text) > 100 else response_text)
            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = part.inline_data.data
                print("✅ Image generated successfully!")
        
        # Save the image if we got one
        if image_data:
            image = Image.open(BytesIO(image_data))
            
            # Ensure the image has the right dimensions
            if image.width != 1080 or image.height != 1920:
                print(f"Resizing image from {image.width}×{image.height} to 1080×1920")
                image = image.resize((1080, 1920), Image.Resampling.LANCZOS)
            
            image.save(image_path)
            return str(image_path)
        else:
            print("No image data in the response, falling back to text image")
            create_text_image(frame, script_title, image_path)
            return str(image_path)
            
    except Exception as e:
        print(f"Error generating image: {e}")
        print("Falling back to text-based image...")
        import traceback
        traceback.print_exc()
        create_text_image(frame, script_title, image_path)
        return str(image_path)

def create_text_image(frame, title, image_path):
    """Create a text-based image with the frame's content."""
    from PIL import ImageDraw, ImageFont
    import textwrap
    import random
    
    # Create a blank image with a gradient background
    width, height = 1080, 1920  # Vertical format for shorts
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Define a richer set of gradient color themes based on the frame content
    # These are more visually interesting than simple gradients
    mood_colors = {
        "informative": [
            [(25, 25, 112), (65, 105, 225), (176, 196, 222)],  # Blue tones
            [(0, 48, 73), (22, 92, 125), (58, 134, 255)]  # Deep blue to bright blue
        ],
        "exciting": [
            [(139, 0, 0), (255, 99, 71), (255, 165, 0)],  # Red to orange
            [(128, 0, 128), (221, 160, 221), (255, 192, 203)]  # Purple to pink
        ],
        "calm": [
            [(0, 100, 0), (34, 139, 34), (154, 205, 50)],  # Green tones
            [(25, 25, 112), (70, 130, 180), (173, 216, 230)]  # Blue to light blue
        ],
        "mysterious": [
            [(75, 0, 130), (138, 43, 226), (216, 191, 216)],  # Deep purple to lavender
            [(25, 25, 112), (0, 0, 139), (72, 61, 139)]  # Dark blue to indigo
        ],
        "technical": [
            [(47, 79, 79), (105, 105, 105), (192, 192, 192)],  # Slate to silver
            [(0, 0, 0), (47, 79, 79), (119, 136, 153)]  # Black to slate
        ]
    }
    
    # Analyze frame content to determine mood
    narration = frame["narration"].lower()
    image_desc = frame["image_description"].lower()
    content = narration + " " + image_desc
    
    # Simple keyword-based mood detection
    mood = "informative"  # default mood
    mood_keywords = {
        "exciting": ["amazing", "incredible", "exciting", "stunning", "breakthrough", 
                    "revolutionary", "mind-blowing", "spectacular", "awesome"],
        "calm": ["peaceful", "gentle", "serene", "relaxing", "soothing", "tranquil", 
                "quiet", "calm", "harmony"],
        "mysterious": ["secret", "mystery", "unknown", "hidden", "curious", "enigmatic", 
                      "strange", "unexplained", "mysterious"],
        "technical": ["technology", "system", "process", "function", "technical", "design", 
                     "engineering", "scientific", "code", "data"]
    }
    
    # Detect mood based on keywords
    for potential_mood, keywords in mood_keywords.items():
        if any(keyword in content for keyword in keywords):
            mood = potential_mood
            break
    
    # Select a color scheme based on detected mood
    color_options = mood_colors.get(mood, mood_colors["informative"])
    colors = random.choice(color_options)
    
    # Draw a more complex gradient (3-point gradient)
    for y in range(height):
        # Calculate the position ratio
        ratio = y / height
        
        # Determine which segment of the gradient we're in
        if ratio < 0.5:
            # Interpolate between first and second colors
            segment_ratio = ratio * 2  # Scale to 0-1 range
            r = int(colors[0][0] * (1 - segment_ratio) + colors[1][0] * segment_ratio)
            g = int(colors[0][1] * (1 - segment_ratio) + colors[1][1] * segment_ratio)
            b = int(colors[0][2] * (1 - segment_ratio) + colors[1][2] * segment_ratio)
        else:
            # Interpolate between second and third colors
            segment_ratio = (ratio - 0.5) * 2  # Scale to 0-1 range
            r = int(colors[1][0] * (1 - segment_ratio) + colors[2][0] * segment_ratio)
            g = int(colors[1][1] * (1 - segment_ratio) + colors[2][1] * segment_ratio)
            b = int(colors[1][2] * (1 - segment_ratio) + colors[2][2] * segment_ratio)
        
        # Draw a line with the calculated color
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add a subtle texture overlay for more visual interest
    for i in range(2000):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        # Draw semi-transparent white dots for texture
        draw.point((x, y), fill=(255, 255, 255, random.randint(5, 30)))
    
    # Try to load font, with multiple fallbacks
    title_font = None
    subtitle_font = None
    body_font = None
    
    # List of fonts to try in order of preference
    font_options = [
        "Arial", "Helvetica", "Verdana", "Tahoma", 
        "Times New Roman", "Georgia", "Courier New"
    ]
    
    # Try each font until we find one that works
    for font_name in font_options:
        try:
            title_font = ImageFont.truetype(font_name, 80)
            subtitle_font = ImageFont.truetype(font_name, 60)
            body_font = ImageFont.truetype(font_name, 50)
            break
        except:
            continue
    
    # If all failed, use default
    if title_font is None:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Draw title at the top with a subtle shadow effect
    title_text = title
    wrapped_title = textwrap.fill(title_text, width=20)
    
    # Get title dimensions to center it properly
    title_width, title_height = title_font.getbbox(wrapped_title)[2:4]
    
    # Draw shadow
    for offset in range(1, 4):
        draw.text((width//2 - title_width//2 + offset, 200 + offset), wrapped_title, 
                 fill=(0, 0, 0, 100), font=title_font, align="center")
    
    # Draw main title
    draw.text((width//2 - title_width//2, 200), wrapped_title, 
             fill="white", font=title_font, align="center")
    
    # Add decorative accent line under title
    line_y = 320
    line_width = min(700, len(title_text) * 30)
    draw.line([(width//2 - line_width//2, line_y), (width//2 + line_width//2, line_y)], 
             fill="white", width=3)
    
    # Draw the frame ID or a subtitle with a more engaging format
    if frame['frame_id'] == 1:
        subtitle = "START"
    elif frame['frame_id'] == 2:
        subtitle = "NEXT"
    else:
        subtitle = f"PART {frame['frame_id']}"
    
    # Get subtitle dimensions to center it properly
    subtitle_width, subtitle_height = subtitle_font.getbbox(subtitle)[2:4]
    
    # Draw subtitle with shadow
    for offset in range(1, 3):
        draw.text((width//2 - subtitle_width//2 + offset, 400 + offset), subtitle, 
                 fill=(0, 0, 0, 100), font=subtitle_font)
    
    draw.text((width//2 - subtitle_width//2, 400), subtitle, fill="white", font=subtitle_font)
    
    # Extract narration text and format it nicely
    narration = frame["narration"]
    
    # Clean up narration text - remove excess spaces, etc.
    narration = " ".join(narration.split())
    
    # For longer narration, split into smaller chunks for better readability
    if len(narration) > 60:
        wrapped_text = textwrap.fill(narration, width=25)
        
        # Calculate size to center text
        text_width, text_height = body_font.getbbox(wrapped_text)[2:4]
        
        # Use shadow effect for better readability
        for offset in range(1, 3):
            draw.text((width//2 - text_width//2 + offset, height//2 - text_height//2 + offset), wrapped_text, 
                     fill=(0, 0, 0, 150), font=body_font, align="center")
        
        draw.text((width//2 - text_width//2, height//2 - text_height//2), wrapped_text, 
                 fill="white", font=body_font, align="center")
    else:
        # For shorter text, make it more prominent
        text_width, text_height = subtitle_font.getbbox(narration)[2:4]
        
        # For shorter text, make it more prominent
        for offset in range(1, 3):
            draw.text((width//2 - text_width//2 + offset, height//2 - text_height//2 + offset), narration, 
                     fill=(0, 0, 0, 150), font=subtitle_font, align="center")
        
        draw.text((width//2 - text_width//2, height//2 - text_height//2), narration, 
                 fill="white", font=subtitle_font, align="center")
    
    # Get keywords from the image description for hashtags
    description = frame["image_description"]
    words = description.split()
    
    # Filter for more meaningful keywords (longer and alphabetic)
    keywords = [word for word in words if len(word) > 4 and word.isalpha() 
               and word.lower() not in ['about', 'these', 'those', 'their', 'there', 'would', 'should']]
    
    # Select a random subset of keywords if we have enough
    if len(keywords) > 5:
        keywords = random.sample(keywords, 5)
    
    # Add keywords as hashtags at the bottom
    if keywords:
        hashtags = " ".join([f"#{word.lower()}" for word in keywords])
        hashtags = textwrap.fill(hashtags, width=30)
        
        # Calculate size for centering
        tag_width, tag_height = body_font.getbbox(hashtags)[2:4]
        
        # Draw shadow
        for offset in range(1, 3):
            draw.text((width//2 - tag_width//2 + offset, height-150 - tag_height//2 + offset), hashtags, 
                     fill=(0, 0, 0, 100), font=body_font, align="center")
        
        draw.text((width//2 - tag_width//2, height-150 - tag_height//2), hashtags, 
                 fill="white", font=body_font, align="center")
    
    # Save the image
    img.save(image_path)
    print(f"Created enhanced text image for frame {frame['frame_id']}")

async def generate_audio(text, frame_id, voice="nova"):
    """Generate audio for narration using OpenAI's TTS."""
    audio_path = output_dir / f"audio_{frame_id}.mp3"
    
    # Skip if audio already exists
    if audio_path.exists():
        print(f"Audio for frame {frame_id} already exists, skipping generation")
        return str(audio_path)
    
    print(f"Generating audio for frame {frame_id}...")
    
    # Add a small animation to show progress
    for i in range(3):
        time.sleep(0.3)
        dots = "." * (i % 3 + 1)
        spaces = " " * (2 - i % 3)
        print(f"🔊 Converting text to speech{dots}{spaces}", end="\r", flush=True)
    print("")
    
    # Prepare the voice style instruction based on the narration length
    # Use different instructions for different narration lengths
    if len(text) < 30:
        # For very short narration, use a more energetic style
        instructions = """
        Voice Affect: Energetic and engaging, like a TikTok creator.
        Tone: Enthusiastic and attention-grabbing.
        Pacing: Quick but clear, with emphasis on key words.
        Emotion: Express excitement and interest to grab attention.
        """
    else:
        # For longer narration, use a more measured style
        instructions = """
        Voice Affect: Conversational and engaging, like a professional YouTuber.
        Tone: Clear and friendly, with natural emphasis on key points.
        Pacing: Dynamic - slower for important concepts, slightly faster for transitions.
        Emotion: Express genuine interest in the topic with subtle enthusiasm.
        """
    
    try:
        # Direct approach - simpler and more reliable
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.1,  # Slightly faster for Shorts format
            response_format="mp3",
            instructions=instructions
        )
        
        # Get the binary content
        response_bytes = await response.aread()
        
        # Save the audio
        with open(audio_path, "wb") as f:
            f.write(response_bytes)
        
        print(f"✓ Audio generated successfully for frame {frame_id}")
        return str(audio_path)
        
    except Exception as e:
        print(f"❌ Error generating audio for frame {frame_id}: {e}")
        # Print the full exception details for debugging
        import traceback
        traceback.print_exc()
        return None

def assemble_video(frames_data, title):
    """Assemble the final video from images and audio."""
    print("Assembling video...")
    
    clips = []
    total_duration = 0
    
    # Standard YouTube Shorts dimensions
    target_width = 1080
    target_height = 1920
    
    # Simplified approach - directly create clips from resized images
    for frame in frames_data:
        try:
            # Set up paths
            img_path = frame["image_path"]
            resized_path = output_dir / f"resized_frame_{frame['frame_id']}.png"
            
            # Resize using PIL directly to avoid MoviePy issues
            from PIL import Image
            img = Image.open(img_path)
            
            # Create new image with correct dimensions
            new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            
            # Resize original preserving aspect ratio
            img_width, img_height = img.size
            if img_width != target_width or img_height != target_height:
                # Calculate new dimensions preserving aspect ratio
                if img_width / img_height > target_width / target_height:
                    # Image is wider than target aspect ratio
                    new_width = target_width
                    new_height = int(img_height * (target_width / img_width))
                else:
                    # Image is taller than target aspect ratio
                    new_height = target_height
                    new_width = int(img_width * (target_height / img_height))
                
                # Use Lanczos resampling (equivalent to old ANTIALIAS)
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center the resized image
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                new_img.paste(resized, (paste_x, paste_y))
                
                # Save the properly sized image
                new_img.save(resized_path)
                frame_image_path = str(resized_path)
            else:
                # Image already has correct dimensions
                frame_image_path = img_path
            
            # Create clip from the image
            img_clip = ImageClip(frame_image_path, duration=frame.get("duration_seconds", 10))
            
            # Add audio if available
            if "audio_path" in frame and frame["audio_path"]:
                try:
                    audio_clip = AudioFileClip(frame["audio_path"])
                    audio_duration = audio_clip.duration
                    img_clip = img_clip.set_duration(audio_duration)
                    video_clip = img_clip.set_audio(audio_clip)
                    total_duration += audio_duration
                    print(f"Frame {frame['frame_id']}: {audio_duration:.1f} seconds")
                except Exception as audio_error:
                    print(f"Error adding audio for frame {frame['frame_id']}: {audio_error}")
                    # Continue without audio
                    video_clip = img_clip
                    total_duration += img_clip.duration
            else:
                video_clip = img_clip
                total_duration += img_clip.duration
                
            clips.append(video_clip)
            
        except Exception as e:
            print(f"Error adding frame {frame.get('frame_id', '?')}: {e}")
            import traceback
            traceback.print_exc()
    
    if not clips:
        print("No clips to assemble")
        return None
    
    try:
        # Concatenate the clips - no resize needed since all images are already correct size
        final_clip = concatenate_videoclips(clips)
        
        # Create a safe filename
        safe_title = ''.join(c if c.isalnum() or c in '_- ' else '_' for c in title)
        output_path = output_dir / f"{safe_title.replace(' ', '_')}.mp4"
        
        print("Rendering final video (this may take a moment)...")
        final_clip.write_videofile(
            str(output_path), 
            codec='libx264', 
            audio_codec='aac',
            fps=24
        )
        
        print(f"Video saved to {output_path}")
        print(f"Total duration: {final_clip.duration:.1f} seconds")
        return str(output_path)
        
    except Exception as e:
        print(f"Error assembling video: {e}")
        import traceback
        traceback.print_exc()
        return None

async def process_frame(frame, script_title, voice):
    """Process a single frame: generate image and audio."""
    image_path = await generate_image(frame, script_title)
    audio_path = await generate_audio(frame["narration"], frame["frame_id"], voice)
    
    return {
        **frame,
        "image_path": image_path,
        "audio_path": audio_path
    }

async def main():
    """Main function to orchestrate the video creation process."""
    args = parse_arguments()
    
    # For pre-loaded script flow
    if hasattr(args, 'script_data'):
        # Get script title for the output directory
        script_title = args.script_data.get('title', 'Educational Video')
        
        # Initialize the output directory with a name based on the script title
        safe_title = ''.join(c if c.isalnum() or c in '_- ' else '_' for c in script_title)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        global output_dir
        output_dir = Path(f"output/{safe_title.replace(' ', '_')}_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Skip concept generation, use the pre-loaded script
        script = generate_script(args.concept, args.duration, args.frames, args.style, 
                                None, args.script_data)
    else:
        # Standard flow - generate concept and script
        # 1. Generate creative concept with thinking model
        video_concept = generate_video_concept(args.concept, args.style, args.duration, args.frames)
        
        # 2. Generate detailed script based on concept
        script = generate_script(args.concept, args.duration, args.frames, args.style, video_concept)
    
    # 3. Process each frame (generate image and audio in parallel)
    print("Processing frames...")
    tasks = [process_frame(frame, script['title'], args.voice) for frame in script['frames']]
    frames_data = await asyncio.gather(*tasks)
    
    # 4. Assemble the video
    video_path = assemble_video(frames_data, script['title'])
    
    if video_path:
        print(f"\n✅ Process complete! YouTube Shorts video created successfully.")
        print(f"Title: {script['title']}")
        print(f"Output: {video_path}")
    else:
        print("\n❌ Video creation failed.")

if __name__ == "__main__":
    asyncio.run(main())