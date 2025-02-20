import re
import sys
import random
import math
import json
from collections import Counter
from itertools import islice

print("N-Gram Model Implementation")
def tokenize(text):
    """Tokenize text into words, ignoring punctuation."""
    return re.findall(r'\b\w+\b', text.lower())

def compute_bigram_probabilities(tokens):
    """Compute bigram probabilities using Maximum Likelihood Estimation (MLE)."""
    bigrams = list(zip(tokens, islice(tokens, 1, None)))
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(tokens)
    
    bigram_probs = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}
    return bigram_probs

def print_top_bigrams(bigram_probs, top_n=5):
    """Print the top N most probable bigrams."""
    top_bigrams = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for bigram, prob in top_bigrams:
        print(f"Bigram: {bigram} -> Probability: {prob:.4f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python bigram_model.py <filename>")
        return
    
    filename = sys.argv[1]
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        
        tokens = tokenize(text)
        bigram_probs = compute_bigram_probabilities(tokens)
        print_top_bigrams(bigram_probs)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

