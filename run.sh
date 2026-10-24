#!/usr/bin/env bash
# usage: ./run.sh <audio_dir> [out_dir]
python run.py --audio_dir "${1:?audio folder required}" --out_dir "${2:-outputs}"
