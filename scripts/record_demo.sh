#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${1:-demo.gif}
DURATION=${DURATION:-8}
FPS=${FPS:-20}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to record a demo" >&2
  exit 1
fi

# macOS users can install and use `ffmpeg -f avfoundation` instead.
ffmpeg \
  -y \
  -f x11grab \
  -video_size ${VIDEO_SIZE:-1920x1080} \
  -i ${DISPLAY:-:0.0} \
  -t "$DURATION" \
  -vf "fps=$FPS,scale=1280:-1:flags=lanczos" \
  "$OUTPUT"

echo "Saved demo to $OUTPUT"
