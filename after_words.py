#!/usr/bin/env python3
"""
Literary Translation and Style Rewriter using Ollama Thinking Models
Translates and rewrites public domain texts in the style of specific authors
"""

import os
import random
import re
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import ollama
from ollama import chat

# ========================================================================
# CONFIGURATION VARIABLES - Modify these for your use case
# ========================================================================

# Model Configuration
MODEL_NAME = "qwen3:8b" #  "deepseek-r1:8b"  # Default thinking model
TEMPERATURE = 0.7  # Balance between creativity and accuracy (0.0-1.0)
TOP_P = 0.9  # Nucleus sampling parameter

# Text Processing
TARGET_WORDS_PER_PAGE = 500  # Approximate words per processing chunk
MIN_WORDS_PER_PAGE = 200  # Minimum words before creating a new page
MAX_WORDS_PER_PAGE = 800  # Maximum words per page
PRESERVE_CHAPTER_BREAKS = True  # Keep chapter divisions intact

style = {
    'Sheila Heti': 'Write with a searching, self-aware energy that continually questions its own motives and hesitations, layering confession with detour, and detour with revelation. Let the sentences wander toward discovery, refusing neatness, sometimes exposing the scaffolding of thought, sometimes tumbling into lyric candor. The tone should feel both intimate and estranged, as though the act of writing is simultaneously creating and undoing the world it describes.',
}

# Style Configuration
TARGET_AUTHOR = "Sheila Heti" # "Karl Ove Knausgård"  # Target writing style
TARGET_AUTHOR_STYLE = style[TARGET_AUTHOR]
SOURCE_LANGUAGE = "German"  # "auto" for detection, or specify: "German", "French", etc.
TARGET_LANGUAGE = "English"

# Output Configuration
OUTPUT_DIR = Path("./translations")
OUTPUT_FILENAME = None  # command line arg position 2 required
SAVE_THINKING_LOG = True  # Save thinking process to separate file
THINKING_LOG_FILENAME = "thinking_log.json"
STREAM = True

# Processing Options
TEST_MODE = False  # Set to True to process only TEST_PAGES
TEST_PAGES = 5  # Number of pages to process in test mode
VERBOSE = True  # Print progress and thinking summaries
RETRY_ATTEMPTS = 3  # Retry failed pages
RETRY_DELAY = 5  # Seconds between retries

# Rate Limiting
DELAY_BETWEEN_PAGES = 0.1  # Seconds to wait between pages (be kind to your GPU)

# ========================================================================
# PROMPT TEMPLATES
# ========================================================================

SYSTEM_PROMPT = """You are a master literary translator and writer with deep expertise in {target_author}'s distinctive writing style. Your task is to translate and rewrite the given text from {source_language} into {target_language}, capturing not just the meaning but transforming it into {target_author}'s unique voice and style.

ALL OUTPUT WORDS MUST BE IN ENGLISH

CRITICAL INSTRUCTIONS:
- Output ONLY the translated and rewritten text
- Use no formatting markers, no titles, no metadata
- Do not add explanatory notes or commentary
- Do not ask questions or make suggestions
- Simply produce the raw literary text in the target style
- Maintain paragraph breaks as in the original
- This is creative literary translation, not literal translation
- Preserve paragraph breaks, but do not preserve line breaks (do not break sentences)
- ALL OUTPUT WORDS MUST BE IN ENGLISH"""

USER_PROMPT = """Translate and rewrite the following text into {target_language} in the distinctive style of {target_author}. Remember: output ONLY the translated literary text, nothing else.

ALL OUTPUT WORDS MUST BE IN ENGLISH

This is not a mechanical act of carrying words across borders, but a re-imagining in which the story is pressed through a new consciousness. The text should emerge altered, like fabric washed and wrung out until the weave shows different patterns of light. It must not only reproduce meaning but metabolize it, carrying the marks of the present voice, the accidents of choice, the inevitable distortions that become its signature.

{target_author_style}

Original text:
---
{text}
---

ALL OUTPUT WORDS MUST BE IN ENGLISH

Now produce the translation in {target_author}'s style:"""

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_text_file(filepath: Path) -> str:
    """Load text from various formats"""
    suffix = filepath.suffix.lower()
    
    try:
        if suffix in ['.txt', '.text']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif suffix == '.html':
            # Basic HTML stripping
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Decode HTML entities
            import html
            text = html.unescape(text)
            return text
        else:
            print(f"Note: Treating {suffix} as plain text. For better results, convert to .txt first.")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def split_into_pages(text: str) -> List[str]:
    """Split text into pages ending at paragraph boundaries"""
    # Split into paragraphs (handle various line endings)
    paragraphs = re.split(r'\n\s*\n+', text)
    
    pages = []
    current_page = []
    current_word_count = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        word_count = len(para.split())
        
        # Check for chapter breaks if preserving them
        if PRESERVE_CHAPTER_BREAKS and re.match(r'^(Chapter|CHAPTER|Part|PART)\s+[\dIVXLCDM]+', para):
            # Save current page if it has content
            if current_page:
                pages.append('\n\n'.join(current_page))
            # Start new page with chapter heading
            current_page = [para]
            current_word_count = word_count
            continue
        
        # Check if adding this paragraph would exceed max words
        if current_word_count + word_count > MAX_WORDS_PER_PAGE and current_page:
            pages.append('\n\n'.join(current_page))
            current_page = [para]
            current_word_count = word_count
        elif current_word_count >= TARGET_WORDS_PER_PAGE and current_page:
            # We've hit our target, save the page
            pages.append('\n\n'.join(current_page))
            current_page = [para]
            current_word_count = word_count
        else:
            current_page.append(para)
            current_word_count += word_count
    
    # Don't forget the last page
    if current_page:
        pages.append('\n\n'.join(current_page))
    
    return pages

def detect_language(text: str) -> str:
    """Simple language detection based on common words"""
    # This is a simplified detection - you might want to use a library like langdetect
    german_words = set(['der', 'die', 'das', 'und', 'ist', 'ein', 'nicht', 'von'])
    french_words = set(['le', 'la', 'les', 'et', 'est', 'un', 'une', 'de', 'ne', 'pas'])
    spanish_words = set(['el', 'la', 'los', 'las', 'y', 'es', 'un', 'una', 'de', 'no'])
    
    words = text.lower().split()[:100]  # Check first 100 words
    
    german_count = sum(1 for w in words if w in german_words)
    french_count = sum(1 for w in words if w in french_words)
    spanish_count = sum(1 for w in words if w in spanish_words)
    
    if german_count > max(french_count, spanish_count) and german_count > 3:
        return "German"
    elif french_count > max(german_count, spanish_count) and french_count > 3:
        return "French"
    elif spanish_count > max(german_count, french_count) and spanish_count > 3:
        return "Spanish"
    else:
        return "Unknown (possibly English or other)"

def translate_page(page_text: str, page_num: int, total_pages: int) -> Tuple[str, str]:
    """Translate a single page using Ollama with thinking mode"""
    system = SYSTEM_PROMPT.format(
        target_author=TARGET_AUTHOR,
        source_language=SOURCE_LANGUAGE if SOURCE_LANGUAGE != "auto" else "the source language",
        target_language=TARGET_LANGUAGE,
    )
    
    user = USER_PROMPT.format(
        target_language=TARGET_LANGUAGE,
        target_author=TARGET_AUTHOR,
        target_author_style=TARGET_AUTHOR_STYLE,
        text=page_text
    )
    
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user}
    ]
    
    attempts = 0
    while attempts < RETRY_ATTEMPTS:
        try:
            if VERBOSE:
                print(f"  Processing page {page_num}/{total_pages}...", end='', flush=True)
            
            thinking = ""
            content = ""

            if STREAM:
                print(f"  Processing page {page_num}/{total_pages}...", flush=True)
                
                # Open files for streaming
                thinking_file = open(f"translations/thinking_page_{page_num}.txt", 'w', encoding='utf-8')
                content_file = open(f"translations/content_page_{page_num}.txt", 'w', encoding='utf-8')
                
                thinking_started = False
                content_started = False

                for part in chat(
                    model=MODEL_NAME,
                    messages=messages,
                    think=True,
                    stream=True,
                    options={'temperature': TEMPERATURE, 'top_p': TOP_P}
                ):
                    if part.get('message', {}).get('thinking'):
                        chunk = part['message']['thinking']
                        thinking += chunk
                        thinking_file.write(chunk)
                        thinking_file.flush()  # Immediate write
                        if not thinking_started:
                            print("  [THINKING] ", end='', flush=True)
                            thinking_started = True
                        if random.random() < 0.01:
                            print('.', end='', flush=True)  # Progress dots
                    
                    if part.get('message', {}).get('content'):
                        chunk = part['message']['content']
                        content += chunk
                        content_file.write(chunk)
                        content_file.flush()  # Immediate write
                        if not content_started:
                            if thinking_started:
                                print()
                            print("  [CONTENT] ", end='', flush=True)
                            content_started = True
                        print(chunk, end='', flush=True)  # Show content live
                
                thinking_file.close()
                content_file.close()
                print()  # New line after streaming

            else:
                # Call Ollama with thinking enabled
                response = chat(
                    model=MODEL_NAME,
                    messages=messages,
                    think=True,
                    options={
                        'temperature': TEMPERATURE,
                        'top_p': TOP_P
                    }
                )
                
                thinking = response.message.thinking or ""
                content = response.message.content or ""

            # Clean up the content (remove any meta-commentary)
            content = content.strip()

            # Add this debug line
            if VERBOSE or not thinking:
                print(f"\n  [Debug] Thinking length: {len(thinking)} chars", flush=True)

            # Remove common prefixes that models might add
            prefixes_to_remove = [
                "Here is the translation:",
                "Translation:",
                "Here's the text translated",
                "Translated text:",
            ]
            for prefix in prefixes_to_remove:
                if content.lower().startswith(prefix.lower()):
                    content = content[len(prefix):].strip()

            if VERBOSE:
                print(" ✓")
                if thinking and len(thinking) > 100:
                    print(f"  Thinking preview: {thinking[:100]}...")
            
            return content, thinking
            
        except Exception as e:
            attempts += 1
            if VERBOSE:
                print(f"\n  Error on attempt {attempts}: {str(e)}")
            if attempts < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                raise

def main():
    """Main processing function"""
    global MODEL_NAME, TARGET_AUTHOR, OUTPUT_FILENAME, TEST_MODE, TEST_PAGES

    parser = argparse.ArgumentParser(description='Translate and rewrite texts using Ollama thinking models')
    parser.add_argument('input_file', type=str, help='Path to input text file')
    parser.add_argument('output_file', type=str, help='Output filename')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='Ollama model to use')
    parser.add_argument('--author', type=str, default=TARGET_AUTHOR, help='Target author style')
    parser.add_argument('--test', action='store_true', help='Test mode - process only a few pages')
    parser.add_argument('--pages', type=int, default=TEST_PAGES, help='Number of pages in test mode')
    
    args = parser.parse_args()
    
    # Update global variables from arguments
    MODEL_NAME = args.model
    TARGET_AUTHOR = args.author
    OUTPUT_FILENAME = args.output_file
    TEST_MODE = args.test
    if args.test:
        TEST_PAGES = args.pages
    
    # Setup
    ensure_output_dir()
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Literary Translation & Style Rewriter")
    print(f"{'='*60}")
    print(f"Input: {input_path.name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Target Style: {TARGET_AUTHOR}")
    print(f"Output: {OUTPUT_DIR / OUTPUT_FILENAME}")
    
    # Load and prepare text
    print(f"\nLoading text...")
    text = load_text_file(input_path)
    
    # Detect language if auto
    if SOURCE_LANGUAGE == "auto":
        detected = detect_language(text)
        print(f"Detected language: {detected}")
    
    # Split into pages
    print(f"Splitting into pages (~{TARGET_WORDS_PER_PAGE} words each)...")
    pages = split_into_pages(text)
    
    total_pages = len(pages) if not TEST_MODE else min(TEST_PAGES, len(pages))
    print(f"Total pages to process: {total_pages} {'(TEST MODE)' if TEST_MODE else ''}")
    
    # Process pages
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    thinking_log_path = OUTPUT_DIR / THINKING_LOG_FILENAME if SAVE_THINKING_LOG else None
    
    thinking_log = []
    
    print(f"\nStarting translation...")
    print(f"{'='*60}")
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, page in enumerate(pages[:total_pages], 1):
            try:
                # Show preview of original
                preview = page[:100].replace('\n', ' ')
                print(f"\nPage {i}: {preview}...")
                
                # Translate
                translated, thinking = translate_page(page, i, total_pages)
                
                # Write translation
                if i > 1:
                    outfile.write('\n\n')
                outfile.write(translated)
                outfile.flush()

                # Log thinking if enabled (ADD THIS)
                if SAVE_THINKING_LOG and thinking:
                    thinking_log.append({
                        'page': i,
                        'original_preview': page[:200],
                        'thinking': thinking,
                        'translation_preview': translated[:200]
                    })

                # Save thinking log immediately after each page (KEEP THIS)
                if SAVE_THINKING_LOG and thinking_log:
                    with open(thinking_log_path, 'w', encoding='utf-8') as f:
                        json.dump(thinking_log, f, indent=2, ensure_ascii=False)
                    print(f"\nThinking log saved to: {thinking_log_path}")
                
                # Rate limiting
                if i < total_pages:
                    time.sleep(DELAY_BETWEEN_PAGES)
                    
            except Exception as e:
                print(f"\nError processing page {i}: {str(e)}")
                print("Skipping to next page...")
                continue
        
    print(f"\n{'='*60}")
    print(f"Translation complete!")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()