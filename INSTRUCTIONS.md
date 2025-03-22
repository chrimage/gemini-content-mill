# AI-Generated Content Mill Project Plan

## Project Overview

This document outlines the implementation plan for an AI-powered content mill that generates YouTube videos automatically using Gemini 2.0 Flash for script generation and image creation, and OpenAI's TTS for voice narration.

## Project Components

1. **Script Generation** - Using Gemini 2.0 Flash to create video scripts
2. **Image Generation** - Using Gemini 2.0 Flash to create images for each frame
3. **Voice Narration** - Using OpenAI's TTS to generate narration audio
4. **Video Assembly** - Using MoviePy to combine images and audio into final videos

## Implementation Timeline

| Phase | Description | Estimated Duration |
|-------|-------------|-------------------|
| 1 | Environment setup and API integration | 1-2 days |
| 2 | Script generation implementation | 2-3 days |
| 3 | Image generation pipeline | 2-3 days |
| 4 | Voice narration system | 1-2 days |
| 5 | Video assembly and export | 2-3 days |
| 6 | Testing and refinement | 3-5 days |

## Detailed Component Specifications

### 1. Script Generation Component

The script generator creates structured video scripts with image descriptions and narration text.

#### Example Script Generation Prompt:

```python
prompt = f"""
Create a detailed YouTube video script on a topic related to {topic_category}.
{f'The title should be: {title}' if title else 'Generate an engaging title.'}

The video should be approximately {duration_minutes} minutes long with a {style} style.

Format your response as a JSON object with the following structure:
{{
  "title": "The video title",
  "description": "A compelling YouTube description including relevant tags",
  "frames": [
    {{
      "frame_id": 1,
      "narration": "Text to be narrated for this frame (about 30-40 words)",
      "image_description": "Detailed description of what should appear in this frame's image",
      "duration_seconds": 10
    }},
    // Additional frames...
  ]
}}

Include approximately {num_frames} frames total. Make sure each image_description is detailed 
enough for an AI to generate a compelling image. The narration should be natural, engaging 
and flow well between frames.
"""
```

#### Example Script Output:

```json
{
  "title": "Beyond the Event Horizon: Unraveling Black Hole Mysteries",
  "description": "Journey into the most enigmatic objects in our universe - black holes. Discover how these cosmic phenomena bend space and time, what happens at the event horizon, and the latest discoveries from astronomers. #BlackHoles #Astrophysics #SpaceExploration #Science",
  "frames": [
    {
      "frame_id": 1,
      "narration": "In the vast expanse of our universe, few objects captivate our imagination like black holes - regions where gravity is so intense that nothing, not even light, can escape their grasp.",
      "image_description": "A stunning visualization of a black hole against the backdrop of colorful nebulae and distant stars, with a visible accretion disk glowing bright orange and blue around the central dark sphere. Show light bending around the black hole.",
      "duration_seconds": 10
    },
    {
      "frame_id": 2,
      "narration": "Black holes form when massive stars collapse under their own gravity. The more massive the star, the more powerful the resulting black hole.",
      "image_description": "A sequence showing a large, luminous star collapsing inward and then transforming into a black hole, with shock waves emanating outward. Use vibrant colors to show the energy release during collapse.",
      "duration_seconds": 12
    }
    // Additional frames...
  ]
}
```

### 2. Image Generation Component

The image generator creates visuals for each frame based on the image descriptions in the script.

#### Example Image Generation Code:

```python
# Generate image for a frame
prompt = f"Create an image for a YouTube video titled '{script_title}'. {frame['image_description']}"

response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=prompt,
    config=types.GenerateContentConfig(response_modalities=["Image"])
)

# Process and save the image
for part in response.candidates[0].content.parts:
    if part.inline_data:
        image_bytes = part.inline_data.data
        image = Image.open(BytesIO(image_bytes))
        image.save(f"frame_{frame['frame_id']}.png")
```

### 3. Voice Narration Component

The voice component converts script text to natural-sounding speech using OpenAI's TTS.

#### Example Voice Generation Code:

```python
# Generate audio for a frame using OpenAI TTS
async def generate_audio(text, output_path, voice="nova"):
    response = await openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions="""
        Voice Affect: Enthusiastic and engaging, like a professional YouTuber.
        Tone: Conversational and friendly, with natural emphasis on key points.
        Pacing: Dynamic - slow down for important concepts, speed up for transitions.
        Emotion: Express genuine interest in the topic with subtle excitement.
        """
    )
    
    response_bytes = await response.read()
    with open(output_path, "wb") as f:
        f.write(response_bytes)
```

#### Voice Style Options:

For educational content:
```
Voice Affect: Clear, authoritative but approachable, like a knowledgeable teacher.
Tone: Informative and encouraging, with enthusiasm for the subject matter.
Pacing: Measured and deliberate, allowing time for complex concepts to be understood.
Emotion: Convey genuine interest in the topic with occasional moments of wonder.
```

For entertaining content:
```
Voice Affect: Energetic, engaging and dynamic, like a charismatic host.
Tone: Conversational, humorous when appropriate, with natural expressiveness.
Pacing: Varied and lively, moving quickly through familiar concepts, slowing for punchlines.
Emotion: Express excitement, surprise, and curiosity to maintain viewer interest.
```

### 4. Video Assembly Component

The assembly component combines images and audio into a complete video using MoviePy.

#### Example Video Assembly Code:

```python
def assemble_video(frames, output_filename):
    clips = []
    
    for frame in frames:
        # Create image clip
        img_clip = ImageClip(frame["image_path"], duration=frame.get("duration_seconds", 10))
        
        # Add audio
        audio_clip = AudioFileClip(frame["audio_path"])
        video_clip = img_clip.set_audio(audio_clip)
        
        clips.append(video_clip)
    
    # Add title and end cards
    title_clip = TextClip("Video Title", fontsize=70, color='white', bg_color='black')
    title_clip = title_clip.set_duration(5)
    
    # Concatenate all clips
    final_clip = concatenate_videoclips([title_clip] + clips)
    
    # Export video
    final_clip.write_videofile(output_filename, codec='libx264', fps=24)
```

## Configuration Options

### Topic Categories
- Technology
- Science
- History
- Travel
- Cooking
- Finance
- Health
- Entertainment

### Style Options
- Educational
- Entertainment
- Documentary
- Tutorial
- News
- Review

### Voice Options
- nova (warm, engaging female voice)
- alloy (neutral versatile voice)
- echo (deep male voice)
- fable (British accent)
- onyx (authoritative voice)
- shimmer (youthful voice)

## Extension Ideas

1. **Topic Discovery**: Use Gemini to analyze trending topics on YouTube/Google and suggest video ideas

2. **Thumbnail Generation**: Create eye-catching thumbnails using AI

3. **SEO Optimization**: Generate tags, descriptions, and titles optimized for search

4. **Analytics Integration**: Track video performance and adjust generation strategies

5. **Batch Processing**: Generate multiple videos in parallel

6. **Foreign Language Support**: Generate videos in multiple languages

## Technical Requirements

- Python 3.8+
- Google API credentials for Gemini 2.0 Flash
- OpenAI API credentials for TTS
- MoviePy and related dependencies
- ~10GB disk space for project and generated content

## Budget Considerations

| Service | Estimated Cost |
|---------|---------------|
| Google Gemini API | $0.0007/1K characters (text), ~$0.05-0.10 per image |
| OpenAI TTS | $0.015/1K characters |
| Storage | Negligible |
| Processing | Local compute |
| **Estimated cost per 3-min video** | **$1.00-$2.00** |

## Next Steps

1. Set up development environment and install dependencies
2. Implement script generation with Gemini 2.0 Flash
3. Build image generation pipeline
4. Integrate OpenAI TTS for narration
5. Create video assembly module
6. Test full pipeline with various topics and styles
7. Implement batch processing for scaling
