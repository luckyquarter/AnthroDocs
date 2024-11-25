# Written by LuckyQuater - 11/25/2024

import anthropic
import requests
import logging
import numpy as np
import os
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Constants
API_KEY = ""
ANTHROPIC_MODEL = "claude-2"
anthropic_client = anthropic.Client(API_KEY)
MAX_CONTENT_LENGTH = 5000

# Helper Functions
def fetch_content(url: str) -> str:
    """Fetch content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Successfully fetched content from {url}")
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch content from {url}: {e}")
        raise

def anthropic_analysis(prompt: str, max_tokens: int = 2000) -> str:
    """Perform analysis using Anthropic AI."""
    try:
        response = anthropic_client.completion(
            prompt=prompt,
            stop=["\n\n"],
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens
        )
        return response.get("completion", "")
    except Exception as e:
        logging.error(f"Anthropic API error: {e}")
        raise

def analyze_documentation(content: str) -> str:
    """Analyze and suggest improvements to documentation."""
    prompt = f"""
    Analyze the following documentation and suggest improvements:
    ```markdown
    {content[:MAX_CONTENT_LENGTH]}
    ```
    Provide recommendations for improved clarity, structure, and examples.
    """
    return anthropic_analysis(prompt)

def suggest_visuals(content: str) -> str:
    """Generate visual representation suggestions."""
    prompt = f"""
    Based on the following documentation, suggest diagrams, charts, or visuals:
    ```markdown
    {content[:MAX_CONTENT_LENGTH]}
    ```
    """
    return anthropic_analysis(prompt)

def compute_readability(content: str) -> float:
    sentences = [s.strip() for s in content.split(".") if s.strip()]
    avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
    return avg_sentence_length

def save_to_file(filename: str, content: str):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Saved content to {filename}")
    except Exception as e:
        logging.error(f"Failed to save content to {filename}: {e}")

# Main Processing Pipeline
def process_documentation(urls: list):
    """Main pipeline for fetching, improving, and evaluating documentation."""
    original_scores = []
    improved_scores = []
    improved_docs = []

    for url in urls:
        try:
            # Fetch content
            content = fetch_content(url)
            original_scores.append(compute_readability(content))

            # Improve content
            improved_text = analyze_documentation(content)
            visuals = suggest_visuals(content)
            enhanced_doc = f"# Improved Documentation\n\n{improved_text}\n\n# Visual Suggestions\n\n{visuals}"
            improved_docs.append(enhanced_doc)
            improved_scores.append(compute_readability(improved_text))
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")

    # Compute Error Metrics
    mse = mean_squared_error(original_scores, improved_scores)
    logging.info(f"Original Readability Scores: {original_scores}")
    logging.info(f"Improved Readability Scores: {improved_scores}")
    logging.info(f"Mean Squared Error (MSE): {mse}")

    # Save Results
    for i, doc in enumerate(improved_docs):
        save_to_file(f"improved_doc_{i + 1}.md", doc)

# Entrypoint
if __name__ == "__main__":
    DOCUMENTATION_URLS = [
        "https://raw.githubusercontent.com/anthropics/courses/master/anthropic_api_fundamentals/01_getting_started.ipynb",
        "https://raw.githubusercontent.com/anthropics/courses/master/real_world_prompting/02_medical_prompt.ipynb",
        "https://raw.githubusercontent.com/zeta-chain/docs/main/some_file.md"
    ]
    process_documentation(DOCUMENTATION_URLS)