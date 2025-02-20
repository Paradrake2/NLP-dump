import re
import sys
import random
import math
import json
from collections import Counter
from itertools import islice

def tokenize(text):
    """Tokenize text into words, ignoring punctuation."""
    return re.findall(r'\b\w+\b', text.lower())

def compute_bigram_probabilities_laplace(tokens):
    """Compute bigram probabilities using Laplace (Add-One) Smoothing."""
    bigrams = list(zip(tokens, islice(tokens, 1, None)))
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(tokens)
    vocabulary_size = len(unigram_counts)
    
    # Convert bigram tuples to strings for JSON compatibility
    bigram_probs = {
        f"{bigram[0]} {bigram[1]}": (count + 1) / (unigram_counts[bigram[0]] + vocabulary_size)
        for bigram, count in bigram_counts.items()
    }
    return bigram_probs

def save_bigram_model(bigram_probs, filename="bigram_model.json"):
    """Save bigram probabilities to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(bigram_probs, f, indent=4)

def load_bigram_model(filename="bigram_model.json"):
    """Load bigram probabilities from a JSON file and convert keys back to tuples."""
    with open(filename, "r", encoding="utf-8") as f:
        bigram_probs_str_keys = json.load(f)
    
    # Convert bigram keys back to tuples
    bigram_probs = {
        tuple(key.split()): value for key, value in bigram_probs_str_keys.items()
    }
    return bigram_probs

def compute_perplexity(test_corpus, bigram_probs):
    """Compute the perplexity of a given bigram model on a test corpus."""
    tokens = tokenize(test_corpus)
    bigrams = list(zip(tokens, islice(tokens, 1, None)))
    N = len(bigrams)
    
    if N == 0:
        return float('inf')  # Avoid division by zero
    
    log_prob_sum = 0
    for bigram in bigrams:
        prob = bigram_probs.get(bigram, 1e-10)  # Get probability, default to small value if missing
        log_prob_sum += math.log(prob)
    
    perplexity = math.exp(-log_prob_sum / N)
    return perplexity

def print_random_bigrams(bigram_probs, top_n=5):
    """Print a random selection of the top N most probable bigrams."""
    top_bigrams = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)
    random_bigrams = random.sample(top_bigrams, min(top_n, len(top_bigrams)))
    for bigram, prob in random_bigrams:
        print(f"Bigram: {bigram} -> Probability: {prob:.4f}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python bigram_laplace.py <train_filename> <test_filename> [model_filename]")
        return
    
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    model_filename = sys.argv[3] if len(sys.argv) > 3 else "bigram_model.json"
    
    try:
        with open(train_filename, 'r', encoding='utf-8') as train_file:
            train_text = train_file.read()
        with open(test_filename, 'r', encoding='utf-8') as test_file:
            test_text = test_file.read()
        
        tokens = tokenize(train_text)
        bigram_probs = compute_bigram_probabilities_laplace(tokens)
        save_bigram_model(bigram_probs, model_filename)
        print_random_bigrams(bigram_probs)
        
        loaded_bigram_probs = load_bigram_model(model_filename)
        perplexity = compute_perplexity(test_text, loaded_bigram_probs)
        print(f"Perplexity of test corpus: {perplexity:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
