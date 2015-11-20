#!/bin/bash
# See for example:
# http://electron.mit.edu/~gsteele/ffmpeg/

ffmpeg -r 20 -pattern_type glob -i "${1}*.png" "${1}.mp4"
