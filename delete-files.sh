#!/bin/bash

# Script to delete all files ending with 'Zone.Identifier' from a specified directory.

# Usage: ./delete_zone_identifier.sh /path/to/directory

# Check if directory path is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the directory path."
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

DIRECTORY="$1"

# Verify that the provided path is a directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: The path provided is not a directory: $DIRECTORY"
    exit 1
fi

# Find and delete files ending with 'Zone.Identifier'
find "$DIRECTORY" -type f -name '*Zone.Identifier' -exec rm -f {} \;

echo "All files ending with 'Zone.Identifier' have been deleted from $DIRECTORY."