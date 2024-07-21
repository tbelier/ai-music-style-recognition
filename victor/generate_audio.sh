#!/bin/bash

# Path to the genres folder
genres_folder="genres"

# Path to the audio folder
audio_folder="audio"

# Create the audio folder if it doesn't exist
mkdir -p "$audio_folder"

# Loop through each subfolder in the genres folder
for genre_folder in "$genres_folder"/*; do
    # Check if it is a directory
    if [ -d "$genre_folder" ]; then
        # Get a random file from the genre folder
        random_file=$(find "$genre_folder" -type f | sort -R | head -n 1)

        # Copy the random file to the audio folder
        cp "$random_file" "$audio_folder/"

        echo "Copied: $random_file to $audio_folder/"
    fi
done

echo "Script completed."

