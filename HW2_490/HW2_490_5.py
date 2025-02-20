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
    
    bigram_probs = {
        f"{bigram[0]} {bigram[1]}": (count + 1) / (unigram_counts[bigram[0]] + vocabulary_size)
        for bigram, count in bigram_counts.items()
    }
    return bigram_probs

def compute_trigram_probabilities_mle(tokens):
    """Compute trigram probabilities using Maximum Likelihood Estimation (MLE)."""
    trigrams = list(zip(tokens, islice(tokens, 1, None), islice(tokens, 2, None)))
    trigram_counts = Counter(trigrams)
    bigram_counts = Counter(zip(tokens, islice(tokens, 1, None)))
    
    trigram_probs = {
        f"{trigram[0]} {trigram[1]} {trigram[2]}": count / bigram_counts[(trigram[0], trigram[1])]
        for trigram, count in trigram_counts.items()
    }
    return trigram_probs

def save_model(bigram_probs, trigram_probs, filename="ngram_model.json"):
    """Save bigram and trigram probabilities to a JSON file."""
    model_data = {
        "bigrams": bigram_probs,
        "trigrams": trigram_probs
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=4)

def load_model(filename="ngram_model.json"):
    """Load bigram and trigram probabilities from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def print_top_trigrams(trigram_probs, top_n=5):
    """Print the top N most probable trigrams."""
    top_trigrams = sorted(trigram_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("\nTop Trigrams:")
    for trigram, prob in top_trigrams:
        print(f"Trigram: {trigram} -> Probability: {prob:.4f}")

def compute_perplexity(test_corpus, bigram_probs):
    """Compute the perplexity of a given bigram model on a test corpus."""
    tokens = tokenize(test_corpus)
    bigrams = list(zip(tokens, islice(tokens, 1, None)))
    N = len(bigrams)
    
    if N == 0:
        return float('inf')  # Avoid division by zero
    
    log_prob_sum = 0
    for bigram in bigrams:
        prob = bigram_probs.get(f"{bigram[0]} {bigram[1]}", 1e-10)  # Get probability or small value
        log_prob_sum += math.log(prob)
    
    perplexity = math.exp(-log_prob_sum / N)
    return perplexity

def generate_sentence(ngram_model, max_words=20):
    """Generate a sentence using the trained bigram model."""
    bigram_probs = ngram_model["bigrams"]

    possible_start_words = list(set(bigram.split()[0] for bigram in bigram_probs.keys()))
    if not possible_start_words:
        return "No words available to generate a sentence."
    
    current_word = random.choice(possible_start_words)
    sentence = [current_word]

    for _ in range(max_words - 1):
        candidates = {bigram.split()[1]: prob for bigram, prob in bigram_probs.items() if bigram.split()[0] == current_word}
        
        if not candidates:
            break  # No possible continuation
        
        next_word = random.choices(list(candidates.keys()), weights=list(candidates.values()), k=1)[0]
        sentence.append(next_word)
        current_word = next_word
    
    return " ".join(sentence).capitalize() + "."

def main():
    if len(sys.argv) < 3:
        print("Usage: python ngram_model.py <train_filename> <test_filename> [model_filename]")
        return
    
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    model_filename = sys.argv[3] if len(sys.argv) > 3 else "ngram_model.json"
    
    try:
        with open(train_filename, 'r', encoding='utf-8') as train_file:
            train_text = train_file.read()
        with open(test_filename, 'r', encoding='utf-8') as test_file:
            test_text = test_file.read()
        
        tokens = tokenize(train_text)
        bigram_probs = compute_bigram_probabilities_laplace(tokens)
        trigram_probs = compute_trigram_probabilities_mle(tokens)
        save_model(bigram_probs, trigram_probs, model_filename)

        # Load model and print results
        loaded_model = load_model(model_filename)
        print_top_trigrams(loaded_model["trigrams"])
        
        perplexity = compute_perplexity(test_text, loaded_model["bigrams"])
        print(f"\nPerplexity of test corpus: {perplexity:.4f}")
        
        # Generate a sentence
        print("\nGenerated Sentence:")
        print(generate_sentence(loaded_model))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
