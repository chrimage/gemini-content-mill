#!/usr/bin/env python3
"""
Topic Generator - A command-line tool to generate hierarchical topic structures
and convert them into a series of YouTube Shorts video concepts.
"""

import argparse
import asyncio
import json
import os
import time
import random
import sys
from pathlib import Path

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API client
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        print("Error: Google API key not found. Please check your .env file.")
        sys.exit(1)
        
    genai_client = genai.Client(api_key=google_api_key)
    
except Exception as e:
    print(f"Error initializing API client: {e}")
    sys.exit(1)

# Model definitions  
THINKING_MODEL = "models/gemini-2.0-flash-thinking-exp"

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
    parser = argparse.ArgumentParser(description="Generate topic hierarchies and YouTube Shorts concepts.")
    parser.add_argument("topic", help="The main topic to explore")
    parser.add_argument("--depth", type=int, default=3, 
                        help="Depth of the topic hierarchy (default: 3)")
    parser.add_argument("--breadth", type=int, default=5, 
                        help="Number of subtopics per topic (default: 5)")
    parser.add_argument("--max-videos", type=int, default=10, 
                        help="Maximum number of videos to generate concepts for (default: 10)")
    parser.add_argument("--output", default="topic_hierarchy.json",
                        help="Output file for the topic hierarchy (default: topic_hierarchy.json)")
    args = parser.parse_args()
    return args

def generate_topic_hierarchy(main_topic, depth=3, breadth=5):
    """Generate a hierarchical topic structure using Gemini."""
    print("\n" + "="*80)
    print(f"🔍 GENERATING TOPIC HIERARCHY FOR: {main_topic}")
    print("="*80)
    print(f"📊 Depth: {depth}, Breadth: {breadth}")
    print("-"*80)
    print("💭 Thinking about knowledge structure and subtopics...")
    
    # Animated thinking process
    for i in range(5):
        time.sleep(0.4)
        dots = "." * (i % 4 + 1)
        spaces = " " * (3 - i % 4)
        print(f"🧠 Analyzing{dots}{spaces}", end="\r", flush=True)
    print()
    
    prompt = f"""
    You are a world-class expert in {main_topic} with decades of experience in teaching and research.
    
    Task: Create a hierarchical topic structure for {main_topic} that covers the field comprehensively 
    and would be suitable for a series of short educational videos.
    
    The hierarchy should have approximately {depth} levels deep and {breadth} topics wide at each level.
    
    For each topic, provide a brief description (1-2 sentences) explaining its importance.
    
    Rules:
    1. Topics should progress from foundational to advanced
    2. Each topic should be self-contained enough for a 60-second educational video
    3. Cover diverse aspects of the field
    4. Include both theoretical and practical topics
    5. Range from introductory to specialized topics
    
    Format your response as a valid JSON object with the following structure:
    {{
      "main_topic": "{{main topic name}}",
      "description": "{{brief description of the main topic}}",
      "subtopics": [
        {{
          "name": "{{subtopic 1 name}}",
          "description": "{{brief description}}",
          "subtopics": [
            {{
              "name": "{{sub-subtopic 1 name}}",
              "description": "{{brief description}}",
              "video_concepts": [
                {{
                  "title": "{{catchy video title}}",
                  "hook": "{{interesting hook/question}}",
                  "key_points": ["{{point 1}}", "{{point 2}}", "{{point 3}}"]
                }},
                // More video concepts...
              ]
            }},
            // More sub-subtopics...
          ]
        }},
        // More subtopics...
      ]
    }}
    
    Return ONLY the JSON object, without any explanations or conversation.
    The JSON must be valid and well-formed, with proper nesting.
    """
    
    try:
        def generate_hierarchy():
            response = genai_client.models.generate_content(
                model=THINKING_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            return response.text
            
        hierarchy_text = api_call_with_backoff(generate_hierarchy)
        
        # Try to extract JSON if it's wrapped in markdown code blocks
        if "```json" in hierarchy_text:
            hierarchy_text = hierarchy_text.split("```json")[1].split("```")[0].strip()
        elif "```" in hierarchy_text:
            hierarchy_text = hierarchy_text.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON
        try:
            hierarchy = json.loads(hierarchy_text)
            print("\n✅ Topic hierarchy generated successfully!")
            return hierarchy
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw response:")
            print(hierarchy_text[:500] + "..." if len(hierarchy_text) > 500 else hierarchy_text)
            raise
            
    except Exception as e:
        print(f"Error generating topic hierarchy: {e}")
        raise

def extract_video_concepts(hierarchy, max_videos=10):
    """Extract video concepts from the hierarchy and flatten into a list."""
    print("\n" + "="*80)
    print("🎬 EXTRACTING VIDEO CONCEPTS")
    print("="*80)
    
    concepts = []
    
    def traverse_hierarchy(node):
        if len(concepts) >= max_videos:
            return
            
        # Get video concepts at this level if they exist
        if "video_concepts" in node:
            for concept in node["video_concepts"]:
                if len(concepts) >= max_videos:
                    break
                concepts.append({
                    "topic": node["name"],
                    "title": concept["title"],
                    "hook": concept["hook"],
                    "key_points": concept["key_points"]
                })
        
        # Traverse subtopics
        if "subtopics" in node:
            for subtopic in node["subtopics"]:
                traverse_hierarchy(subtopic)
    
    # Start traversal from the main topic
    traverse_hierarchy(hierarchy)
    
    print(f"✅ Extracted {len(concepts)} video concepts")
    return concepts

def refine_video_concept(topic, concept):
    """Refine a video concept into a detailed script outline."""
    print(f"🔍 Refining concept: {concept['title']}")
    
    prompt = f"""
    You are creating a detailed outline for a 60-second educational video on "{concept['title']}" 
    which is part of the broader topic "{topic}".
    
    The video will have 12 segments/frames, each with specific narration and visual elements.
    
    Consider this hook: "{concept['hook']}"
    
    And these key points:
    {json.dumps(concept['key_points'], indent=2)}
    
    Create a detailed segment-by-segment outline with:
    1. A catchy, descriptive title (60 characters max)
    2. Exactly 12 segments
    3. Each segment containing:
       - Precise narration text (10-15 words)
       - Clear image description (what should be shown visually)
       - Educational value (what the viewer learns)
    
    Format your response as a valid JSON object with the following structure:
    {{
      "title": "Catchy Video Title",
      "description": "Brief YouTube description with #hashtags",
      "segments": [
        {{
          "segment_id": 1,
          "narration": "Specific narration text for this segment (10-15 words)",
          "image_description": "Detailed description of what should be shown visually",
          "educational_value": "What the viewer learns in this segment"
        }},
        // 11 more segments...
      ]
    }}
    
    Return ONLY valid JSON, without any explanations or conversation.
    """
    
    try:
        def generate_outline():
            response = genai_client.models.generate_content(
                model=THINKING_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            return response.text
            
        outline_text = api_call_with_backoff(generate_outline)
        
        # Try to extract JSON if it's wrapped in markdown code blocks
        if "```json" in outline_text:
            outline_text = outline_text.split("```json")[1].split("```")[0].strip()
        elif "```" in outline_text:
            outline_text = outline_text.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON
        try:
            outline = json.loads(outline_text)
            print(f"✅ Outline generated: {outline['title']}")
            return outline
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw response:")
            print(outline_text[:500] + "..." if len(outline_text) > 500 else outline_text)
            raise
            
    except Exception as e:
        print(f"Error generating video outline: {e}")
        raise

def convert_to_shorts_script(outline):
    """Convert a refined outline to a format compatible with the shorts_creator script."""
    print(f"🔄 Converting outline to shorts script format: {outline['title']}")
    
    shorts_script = {
        "title": outline["title"],
        "description": outline["description"],
        "frames": []
    }
    
    # Default duration for each segment
    duration_seconds = 5
    
    for segment in outline["segments"]:
        shorts_script["frames"].append({
            "frame_id": segment["segment_id"],
            "narration": segment["narration"],
            "image_description": segment["image_description"],
            "duration_seconds": duration_seconds
        })
    
    return shorts_script

def save_json(data, filename):
    """Save data as JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to {filename}")

def main():
    """Main function to orchestrate the process."""
    args = parse_arguments()
    
    # Generate topic hierarchy
    hierarchy = generate_topic_hierarchy(args.topic, args.depth, args.breadth)
    
    # Save the full hierarchy
    save_json(hierarchy, args.output)
    
    # Extract video concepts
    concepts = extract_video_concepts(hierarchy, args.max_videos)
    
    # Save the video concepts
    concepts_file = Path(args.output).stem + "_concepts.json"
    save_json(concepts, concepts_file)
    
    # Create directory for video scripts
    scripts_dir = Path("video_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Get base name of output file for related files
    output_base = Path(args.output).stem if args.output != "topic_hierarchy.json" else "topic_hierarchy"
    
    # Process each concept into a detailed script
    video_scripts = []
    for i, concept in enumerate(concepts):
        print(f"\nProcessing concept {i+1}/{len(concepts)}: {concept['title']}")
        
        # Generate detailed outline
        outline = refine_video_concept(args.topic, concept)
        
        # Convert to shorts script format
        shorts_script = convert_to_shorts_script(outline)
        
        # Save individual script
        safe_title = ''.join(c if c.isalnum() or c in '_- ' else '_' for c in concept['title'])
        script_file = scripts_dir / f"{safe_title.replace(' ', '_')}.json"
        save_json(shorts_script, script_file)
        
        # Add to list of scripts
        video_scripts.append({
            "title": concept["title"],
            "script_file": str(script_file)
        })
    
    # Save the list of scripts
    save_json(video_scripts, "video_scripts.json")
    
    print("\n" + "="*80)
    print(f"✅ TOPIC HIERARCHY AND VIDEO CONCEPTS GENERATED")
    print(f"✅ Generated {len(concepts)} video concepts")
    print(f"✅ Saved topic hierarchy to {args.output}")
    print(f"✅ Saved video concepts to {concepts_file}")
    print(f"✅ Saved video scripts to {scripts_dir}/")
    print(f"✅ Saved script list to video_scripts.json")
    print("="*80)
    
    print("\nTo create a video from a script, run:")
    print("./professor_mode.sh --from-script video_scripts/[script_file].json")

if __name__ == "__main__":
    main()