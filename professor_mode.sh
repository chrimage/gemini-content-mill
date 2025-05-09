#!/bin/bash

# Set up logging
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="professor_mode_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "==============================================================="
echo "Starting Professor Mode: $(date)"
echo "==============================================================="
echo "Log file: $LOG_FILE"

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
  echo "  ./professor_mode.sh --logs [N]              - Show the last N log files (default: 5)"
  echo
  echo "Options:"
  echo "  --depth [NUMBER]     - Depth of topic hierarchy (default: 3)"
  echo "  --breadth [NUMBER]   - Number of subtopics per topic (default: 5)"
  echo "  --max-videos [NUMBER] - Maximum number of videos to generate concepts for (default: 10)"
  echo "  --voice [VOICE]      - Voice to use for narration (nova, alloy, echo, fable, onyx, shimmer)"
  echo "  --style [STYLE]      - Override voice style (professor, excited_teacher, storyteller, etc.)"
  echo "  --skip-video         - Skip video assembly, just generate scripts, images and audio"  
  echo "  --no-auto-generate   - Only generate topic hierarchy without creating videos"
  echo "  --help               - Show this help message"
  echo
  echo "Examples:"
  echo "  ./professor_mode.sh \"Quantum Physics\" --depth 2 --breadth 3"
  echo "  ./professor_mode.sh \"Biology\" --max-videos 5 --voice echo"
  echo "  ./professor_mode.sh --from-script topic_output/quantum_physics/scripts/entanglement.json --voice nova"
  echo "  ./professor_mode.sh --logs 10               - Show the last 10 log files"
}

# Function to list and display log files
function show_logs {
  # Default to showing 5 most recent logs if not specified
  NUM_LOGS=${1:-5}
  
  echo "Showing $NUM_LOGS most recent log files:"
  echo "==============================================================="
  
  # Find all log files and sort by modification time (newest first)
  mapfile -t LOG_FILES < <(find . -maxdepth 1 -name "*.log" | sort -r -t_ -k2,2)
  
  # Check if we found any logs
  if [ ${#LOG_FILES[@]} -eq 0 ]; then
    echo "No log files found."
    return
  fi
  
  # Output summary of most recent logs
  COUNT=0
  for LOG in "${LOG_FILES[@]}"; do
    if [ $COUNT -lt $NUM_LOGS ]; then
      LOG_NAME=$(basename "$LOG")
      LOG_SIZE=$(du -h "$LOG" | cut -f1)
      FIRST_LINE=$(head -n 1 "$LOG")
      TIMESTAMP=$(echo "$FIRST_LINE" | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")
      LAST_LINE=$(tail -n 1 "$LOG")
      
      echo "[$((COUNT+1))] $LOG_NAME ($LOG_SIZE) - $TIMESTAMP"
      COUNT=$((COUNT+1))
    else
      break
    fi
  done
  
  # Ask which log to view
  echo "==============================================================="
  echo "Enter log number to view (or 'q' to quit): "
  read -r SELECTION
  
  if [[ "$SELECTION" =~ ^[0-9]+$ ]] && [ "$SELECTION" -ge 1 ] && [ "$SELECTION" -le $COUNT ]; then
    LOG_TO_VIEW="${LOG_FILES[$((SELECTION-1))]}"
    echo "==============================================================="
    echo "Viewing log: $LOG_TO_VIEW"
    echo "==============================================================="
    less "$LOG_TO_VIEW"
  elif [ "$SELECTION" != "q" ]; then
    echo "Invalid selection."
  fi
}

# Check if no arguments or help requested
if [ $# -eq 0 ] || [ "$1" == "--help" ]; then
  print_help
  exit 0
fi

# Check if logs command is used
if [ "$1" == "--logs" ]; then
  # If a number is provided, use it, otherwise default is handled in the function
  if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
    show_logs "$2"
  else
    show_logs
  fi
  exit 0
fi

# Function to create a video from script
generate_video_from_script() {
  SCRIPT="$1"
  VOICE_OPT="$2"
  OUTPUT_DIR="$3"
  STYLE_OPT="$4"
  SKIP_VIDEO="$5"
  
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
  
  # Show voice style info
  if [ -n "$STYLE_OPT" ]; then
    echo "🎙️ Using voice style: $(echo $STYLE_OPT | cut -d' ' -f2)"
  elif [ -f "$SCRIPT" ] && [ "$(cat "$SCRIPT" | jq 'has("voice_style")')" == "true" ]; then
    echo "🎙️ Script-defined voice style: $(cat "$SCRIPT" | jq -r '.voice_style')"
  else
    echo "🎙️ Using automatic voice style detection"
  fi
  
  # Check if we should skip video assembly
  if [ "$SKIP_VIDEO" = "--skip-video" ]; then
    echo "🔄 Will skip video assembly (generating assets only)"
  fi
  
  # Run shorts_creator.py with the script
  echo "🎬 Creating content..."
  python3 src/shorts_creator.py --from-script "$SCRIPT" $VOICE_OPT $STYLE_OPT $OUTPUT_DIR_ARG $SKIP_VIDEO
  
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
  
  # Extract style option if provided
  STYLE_OPTION=""
  if [[ "$*" == *"--style"* ]]; then
    STYLE=$(echo "$*" | grep -oP '(?<=--style )[^ ]+')
    STYLE_OPTION="--style $STYLE"
  fi
  
  # Check if we should skip video assembly
  SKIP_VIDEO=""
  if [[ "$*" == *"--skip-video"* ]]; then
    SKIP_VIDEO="--skip-video"
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
    if generate_video_from_script "$SCRIPT" "$VOICE_OPTION" "$OUTPUT_DIR" "$STYLE_OPTION" "$SKIP_VIDEO"; then
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
  
  # Extract style option if provided
  STYLE_OPTION=""
  if [[ "$*" == *"--style"* ]]; then
    STYLE=$(echo "$*" | grep -oP '(?<=--style )[^ ]+')
    STYLE_OPTION="--style $STYLE"
  fi
  
  # Check if we should skip video assembly
  SKIP_VIDEO=""
  if [[ "$*" == *"--skip-video"* ]]; then
    SKIP_VIDEO="--skip-video"
  fi
  
  generate_video_from_script "$SCRIPT" "$VOICE_OPTION" "" "$STYLE_OPTION" "$SKIP_VIDEO"
  
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
STYLE_OPTION=""
SKIP_VIDEO=""

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
    --style)
      STYLE="$2"
      STYLE_OPTION="--style $STYLE"
      shift 2
      ;;
    --skip-video)
      SKIP_VIDEO="--skip-video"
      shift
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
    if generate_video_from_script "$TOPIC_SCRIPT" "$VOICE_OPTION" "$VIDEOS_DIR" "$STYLE_OPTION" "$SKIP_VIDEO"; then
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