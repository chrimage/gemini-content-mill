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
  echo "  ./professor_mode.sh [topic]                 - Generate topic hierarchy and automatically create videos"
  echo "  ./professor_mode.sh --from-script [file]    - Create video from existing script"
  echo "  ./professor_mode.sh --generate-all          - Create videos from all scripts in video_scripts.json"
  echo
  echo "Options:"
  echo "  --depth [NUMBER]     - Depth of topic hierarchy (default: 3)"
  echo "  --breadth [NUMBER]   - Number of subtopics per topic (default: 5)"
  echo "  --max-videos [NUMBER] - Maximum number of videos to generate concepts for (default: 10)"
  echo "  --voice [VOICE]      - Voice to use for narration (nova, alloy, echo, fable, onyx, shimmer)"
  echo "  --no-auto-generate   - Only generate topic hierarchy without creating videos"
  echo "  --help               - Show this help message"
  echo
  echo "Examples:"
  echo "  ./professor_mode.sh \"Quantum Physics\" --depth 2 --breadth 3"
  echo "  ./professor_mode.sh \"Biology\" --max-videos 5 --voice echo"
  echo "  ./professor_mode.sh --from-script topic_output/quantum_physics/scripts/entanglement.json --voice nova"
}

# Check if no arguments or help requested
if [ $# -eq 0 ] || [ "$1" == "--help" ]; then
  print_help
  exit 0
fi

# Function to create a video from script
generate_video_from_script() {
  SCRIPT="$1"
  VOICE_OPT="$2"
  OUTPUT_DIR="$3"
  
  # Check if script exists
  if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script file '$SCRIPT' not found"
    return 1
  fi
  
  # Get title from the script
  TITLE=$(cat "$SCRIPT" | jq -r '.title')
  echo "📝 Video title: $TITLE"
  
  # Create output directory argument if provided
  OUTPUT_DIR_ARG=""
  if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_ARG="--output-dir $OUTPUT_DIR"
  fi
  
  # Run shorts_creator.py with the script
  echo "🎬 Creating video..."
  python3 src/shorts_creator.py --from-script "$SCRIPT" $VOICE_OPT $OUTPUT_DIR_ARG
  
  return $?
}

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
  
  # Extract topic name from the file to create output directory
  TOPIC=$(cat topic_hierarchy.json | jq -r '.main_topic' | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  
  # Create output directory if it doesn't exist
  OUTPUT_DIR="topic_output/$TOPIC/videos"
  mkdir -p "$OUTPUT_DIR"
  
  # Loop through each script in the JSON file
  echo "Reading scripts from video_scripts.json..."
  SCRIPTS=$(cat video_scripts.json | jq -r '.[] | .script_file')
  
  COUNT=0
  TOTAL=$(echo "$SCRIPTS" | wc -l)
  SUCCESS_COUNT=0
  
  for SCRIPT in $SCRIPTS; do
    COUNT=$((COUNT+1))
    echo
    echo "="*80
    echo "Processing video $COUNT/$TOTAL: $SCRIPT"
    echo "="*80
    
    # Generate the video from this script
    if generate_video_from_script "$SCRIPT" "$VOICE_OPTION" "$OUTPUT_DIR"; then
      SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    fi
  done
  
  echo
  echo "✅ Generated $SUCCESS_COUNT out of $TOTAL videos successfully!"
  echo "📁 Videos saved in: $OUTPUT_DIR"
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
  
  echo "🎓 PROFESSOR MODE: Creating video from script $SCRIPT"
  
  # Extract extra arguments for shorts_creator.py
  shift 2  # Remove the first two arguments
  
  # Extract voice option if provided
  VOICE_OPTION=""
  if [[ "$*" == *"--voice"* ]]; then
    VOICE=$(echo "$*" | grep -oP '(?<=--voice )[^ ]+')
    VOICE_OPTION="--voice $VOICE"
  fi
  
  generate_video_from_script "$SCRIPT" "$VOICE_OPTION"
  
  exit $?
fi

# If we reach here, we're in topic generation mode
TOPIC="$1"
shift  # Remove the first argument

echo "🎓 PROFESSOR MODE: Generating topic hierarchy for '$TOPIC'"

# Default options
DEPTH=3
BREADTH=5
MAX_VIDEOS=10
AUTO_GENERATE=true
VOICE_OPTION=""

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
    --voice)
      VOICE="$2"
      VOICE_OPTION="--voice $VOICE"
      shift 2
      ;;
    --no-auto-generate)
      AUTO_GENERATE=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

# Create a clean topic slug for directory naming
TOPIC_SLUG=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_-')

# Create topic directory structure
TOPIC_DIR="topic_output/$TOPIC_SLUG"
SCRIPTS_DIR="$TOPIC_DIR/scripts"
VIDEOS_DIR="$TOPIC_DIR/videos"
mkdir -p "$SCRIPTS_DIR" "$VIDEOS_DIR"

# Run the topic generator with custom output path
OUTPUT_FILE="$TOPIC_DIR/hierarchy.json"
CONCEPTS_FILE="$TOPIC_DIR/concepts.json"

echo "📁 Creating topic directory structure in: $TOPIC_DIR"
python3 src/topic_generator.py "$TOPIC" --depth $DEPTH --breadth $BREADTH --max-videos $MAX_VIDEOS --output "$OUTPUT_FILE"

# Check if topic generator was successful
if [ $? -ne 0 ]; then
  echo "❌ Error generating topic hierarchy. Exiting."
  exit 1
fi

# Move scripts to the topic's scripts directory
if [ -d "video_scripts" ]; then
  echo "📂 Moving scripts to topic directory..."
  cp video_scripts/*.json "$SCRIPTS_DIR/"
  
  # Update video_scripts.json to point to new paths
  if [ -f "video_scripts.json" ]; then
    cp video_scripts.json "$TOPIC_DIR/video_scripts.json"
    # Update script paths in the copy
    sed -i "s|video_scripts/|$SCRIPTS_DIR/|g" "$TOPIC_DIR/video_scripts.json"
  fi
fi

# If auto-generate is enabled, create videos for all scripts
if [ "$AUTO_GENERATE" = true ]; then
  echo
  echo "🎬 AUTO-GENERATING VIDEOS FOR ALL SCRIPTS"
  echo "="*80
  
  # Check if video_scripts.json exists
  if [ ! -f "video_scripts.json" ]; then
    echo "Error: video_scripts.json not found."
    exit 1
  fi
  
  # Loop through each script
  SCRIPTS=$(cat video_scripts.json | jq -r '.[] | .script_file')
  
  COUNT=0
  TOTAL=$(echo "$SCRIPTS" | wc -l)
  SUCCESS_COUNT=0
  
  for SCRIPT in $SCRIPTS; do
    # Get corresponding script in the topic directory
    TOPIC_SCRIPT="$SCRIPTS_DIR/$(basename "$SCRIPT")"
    
    COUNT=$((COUNT+1))
    echo
    echo "="*80
    echo "Processing video $COUNT/$TOTAL: $(basename "$SCRIPT")"
    echo "="*80
    
    # Generate the video from this script with output to the topic's videos directory
    if generate_video_from_script "$TOPIC_SCRIPT" "$VOICE_OPTION" "$VIDEOS_DIR"; then
      SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    fi
  done
  
  echo
  echo "✅ Generated $SUCCESS_COUNT out of $TOTAL videos successfully!"
  echo "📁 Topic directory: $TOPIC_DIR"
  echo "📁 Videos saved in: $VIDEOS_DIR"
else
  echo
  echo "✅ Topic hierarchy and scripts generated in: $TOPIC_DIR"
  echo "📝 To generate videos, run:"
  echo "./professor_mode.sh --generate-all $VOICE_OPTION"
fi