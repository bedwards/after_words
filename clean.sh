#!/bin/bash

# Remove Gutenberg headers/footers
sed -n '/\*\*\* START OF THIS PROJECT/,/\*\*\* END OF THIS PROJECT/p' "$1" > "$1-clean.txt"
