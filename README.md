# YouTube Shorts Creator

An AI-powered tool to automatically generate YouTube Shorts videos from a simple concept prompt.

## Features

- Generate engaging video scripts using Gemini 2.0 Flash
- Create compelling images for each frame using Gemini's image generation
- Generate natural-sounding narration using OpenAI's TTS
- Automatically assemble the final video with MoviePy

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file from the template:
   ```
   cp .env-template .env
   ```
4. Add your API keys to the `.env` file:
   - Get a Google API key from [Google AI Studio](https://ai.google.dev/)
   - Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/)

## Usage

Create a YouTube Short from the command line:

```bash
./create_short.sh "Your video concept here"
```

### Options

- `--frames NUMBER`: Number of frames to generate (default: 2)
- `--voice VOICE`: Voice to use for narration (choices: nova, alloy, echo, fable, onyx, shimmer, default: nova)

The video duration is automatically calculated based on the number of frames (approximately 5 seconds per frame).

### Example

```bash
./create_short.sh "Amazing facts about space" --frames 10 --voice nova
```

## Output

Generated files are stored in the `output` directory:
- `script.json`: The generated script
- `frame_X.png`: Generated images for each frame
- `audio_X.mp3`: Generated audio narration for each frame
- `Video_Title.mp4`: The final assembled video

## Notes

- The script creates visually appealing text-based images for each frame
- If you have access to Gemini's image generation models, you can modify the code to use those
- Video generation can take several minutes depending on the number of frames
- The quality of results depends on the specificity of your concept prompt

## Future Enhancements

- Implement Gemini image generation when API access is available
- Add more customization options for text-based images
- Support for adding music or sound effects
- Generate video thumbnails automatically