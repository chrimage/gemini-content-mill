#!/bin/bash

# Make script executable if not already
if [ ! -x "src/shorts_creator.py" ]; then
  chmod +x src/shorts_creator.py
fi

# Check if a concept was provided
if [ $# -eq 0 ]; then
  echo "Error: Please provide a concept for your short video"
  echo "Usage: ./create_short.sh \"Your video concept\" [--frames NUMBER] [--voice VOICE]"
  echo "Example: ./create_short.sh \"The history of computers\" --frames 3"
  exit 1
fi

echo "🎬 Creating YouTube Short: \"$1\""

# Default to a simple configuration for reliable testing
if [[ ! "$*" == *"--frames"* ]]; then
  echo "Using default: 2 frames for basic test"
  FRAMES="--frames 2"
else
  FRAMES=""
fi

# Run the shorts creator with all arguments passed to this script, plus any defaults
python3 src/shorts_creator.py --concept "$1" ${@:2} $FRAMES