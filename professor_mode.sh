#!/bin/bash

# Make scripts executable if not already
if [ ! -x "src/shorts_creator.py" ]; then
  chmod +x src/shorts_creator.py
fi

if [ ! -x "src/topic_generator.py" ]; then
  chmod +x src/topic_generator.py
fi

# Function to print help message
function print_help {
  echo "Professor Mode - Generate educational content hierarchies and shorts"
  echo
  echo "Usage:"
  echo "  ./professor_mode.sh [topic]                 - Generate topic hierarchy and video concepts"
  echo "  ./professor_mode.sh --from-script [file]    - Create video from existing script"
  echo "  ./professor_mode.sh --generate-all          - Create videos from all scripts in video_scripts.json"
  echo
  echo "Options:"
  echo "  --depth [NUMBER]     - Depth of topic hierarchy (default: 3)"
  echo "  --breadth [NUMBER]   - Number of subtopics per topic (default: 5)"
  echo "  --max-videos [NUMBER] - Maximum number of videos to generate concepts for (default: 10)"
  echo "  --voice [VOICE]      - Voice to use for narration (nova, alloy, echo, fable, onyx, shimmer)"
  echo "  --help               - Show this help message"
  echo
  echo "Examples:"
  echo "  ./professor_mode.sh \"Quantum Physics\" --depth 2 --breadth 3"
  echo "  ./professor_mode.sh --from-script video_scripts/quantum_entanglement.json --voice nova"
  echo "  ./professor_mode.sh --generate-all --voice alloy"
}

# Check if no arguments or help requested
if [ $# -eq 0 ] || [ "$1" == "--help" ]; then
  print_help
  exit 0
fi

# Process args for generate all mode
if [ "$1" == "--generate-all" ]; then
  echo "🎓 PROFESSOR MODE: Generating all videos from scripts"
  
  # Check if video_scripts.json exists
  if [ ! -f "video_scripts.json" ]; then
    echo "Error: video_scripts.json not found. Run ./professor_mode.sh [topic] first."
    exit 1
  fi
  
  # Extract voice option if provided
  VOICE_OPTION=""
  if [[ "$*" == *"--voice"* ]]; then
    VOICE=$(echo "$*" | grep -oP '(?<=--voice )[^ ]+')
    VOICE_OPTION="--voice $VOICE"
  fi
  
  # Loop through each script in the JSON file
  echo "Reading scripts from video_scripts.json..."
  SCRIPTS=$(cat video_scripts.json | jq -r '.[] | .script_file')
  
  COUNT=0
  TOTAL=$(echo "$SCRIPTS" | wc -l)
  
  for SCRIPT in $SCRIPTS; do
    COUNT=$((COUNT+1))
    echo
    echo "="*80
    echo "Processing video $COUNT/$TOTAL: $SCRIPT"
    echo "="*80
    
    # Generate the video from this script
    ./professor_mode.sh --from-script "$SCRIPT" $VOICE_OPTION
  done
  
  echo
  echo "✅ All $TOTAL videos generated successfully!"
  exit 0
fi

# Process args for from-script mode
if [ "$1" == "--from-script" ]; then
  if [ -z "$2" ]; then
    echo "Error: Script file not specified"
    echo "Usage: ./professor_mode.sh --from-script [file] [options]"
    exit 1
  fi
  
  SCRIPT="$2"
  
  # Check if script exists
  if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script file '$SCRIPT' not found"
    exit 1
  fi
  
  echo "🎓 PROFESSOR MODE: Creating video from script $SCRIPT"
  
  # Get title from the script
  TITLE=$(cat "$SCRIPT" | jq -r '.title')
  echo "📝 Video title: $TITLE"
  
  # Extract extra arguments for shorts_creator.py
  shift 2  # Remove the first two arguments
  EXTRA_ARGS="$@"
  
  # Default to 12 segments
  if [[ ! "$EXTRA_ARGS" == *"--frames"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --frames 12"
  fi
  
  # Run shorts_creator.py with the script
  echo "🎬 Creating video..."
  python3 src/shorts_creator.py --from-script "$SCRIPT" $EXTRA_ARGS
  
  exit 0
fi

# If we reach here, we're in topic generation mode
TOPIC="$1"
shift  # Remove the first argument

echo "🎓 PROFESSOR MODE: Generating topic hierarchy for '$TOPIC'"

# Default options
DEPTH=3
BREADTH=5
MAX_VIDEOS=10

# Process remaining arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --depth)
      DEPTH="$2"
      shift 2
      ;;
    --breadth)
      BREADTH="$2"
      shift 2
      ;;
    --max-videos)
      MAX_VIDEOS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

# Run the topic generator
python3 src/topic_generator.py "$TOPIC" --depth $DEPTH --breadth $BREADTH --max-videos $MAX_VIDEOS

echo
echo "To generate all videos, run:"
echo "./professor_mode.sh --generate-all [--voice VOICE]"