"""
Demo: Explore Nike catalog tokens and find similar products
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt


def load_data():
    """Load encoded products and embeddings"""
    models_dir = Path("models")
    
    with open(models_dir / "encoded_products.json", 'r') as f:
        products = json.load(f)
    
    embeddings = np.load(models_dir / "embeddings.npy")
    
    with open(models_dir / "token_vocabulary.json", 'r') as f:
        vocab = json.load(f)
    
    with open(models_dir / "statistics.json", 'r') as f:
        stats = json.load(f)
    
    return products, embeddings, vocab, stats


def print_statistics(stats, products):
    """Print catalog statistics"""
    print("=" * 60)
    print("NIKE CATALOG TOKEN STATISTICS")
    print("=" * 60)
    print(f"Total Products:        {stats['total_products']}")
    print(f"Unique Tokens:         {stats['total_tokens']}")
    print(f"Vocabulary Size:       {stats['vocabulary_size']}")
    print(f"Clustering Levels:     {stats['clustering_levels']}")
    print(f"Clusters per Level:    {stats['clusters_per_level']}")
    print()
    
    # Category distribution (from level 0 codes)
    level0_codes = defaultdict(int)
    for p in products:
        if p.get('hierarchical_code'):
            level0_codes[p['hierarchical_code'][0]] += 1
    
    print(f"Top-level Categories:  {len(level0_codes)}")
    print("\nTop 5 Categories by Size:")
    for code, count in sorted(level0_codes.items(), key=lambda x: -x[1])[:5]:
        print(f"  Code {code:2d}: {count:3d} products ({count/len(products)*100:.1f}%)")
    print()


def show_product_with_code(product, index=None):
    """Display a single product with its token code"""
    print("-" * 60)
    if index is not None:
        print(f"Product #{index}")
    print(f"Title:  {product.get('title', 'N/A')}")
    print(f"Subtitle: {product.get('subtitle', 'N/A')}")
    print(f"Price:  {product.get('price', 'N/A')}")
    print(f"Color:  {product.get('color', 'N/A')}")
    print(f"Style:  {product.get('style_code', 'N/A')}")
    print()
    print(f"Token Code:  {product.get('hierarchical_code', 'N/A')}")
    print(f"Token ID:    {product.get('token_id', 'N/A')}")
    print(f"Token String: {product.get('token_str', 'N/A')}")
    print()


def find_similar_products(product_idx, products, embeddings, top_k=5):
    """Find similar products using cosine similarity"""
    query_emb = embeddings[product_idx]
    
    # Compute cosine similarities
    similarities = np.dot(embeddings, query_emb)
    
    # Get top-k (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    print(f"Top {top_k} Similar Products to Product #{product_idx}:")
    print()
    
    show_product_with_code(products[product_idx], index=product_idx)
    
    print("\nSimilar Products:")
    print("=" * 60)
    
    for rank, idx in enumerate(similar_indices, 1):
        similar_product = products[idx]
        similarity_score = similarities[idx]
        
        print(f"\n#{rank} (Similarity: {similarity_score:.3f})")
        show_product_with_code(similar_product, index=int(idx))
        
        # Check code prefix overlap
        query_code = products[product_idx].get('hierarchical_code', [])
        similar_code = similar_product.get('hierarchical_code', [])
        
        if query_code and similar_code:
            prefix_match = 0
            for i in range(min(len(query_code), len(similar_code))):
                if query_code[i] == similar_code[i]:
                    prefix_match += 1
                else:
                    break
            
            print(f"Code Prefix Match: {prefix_match}/{len(query_code)} levels")


def analyze_token_structure(products):
    """Analyze the hierarchical token structure"""
    print("=" * 60)
    print("TOKEN STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Collect all codes at each level
    level_codes = [defaultdict(int) for _ in range(4)]
    
    for p in products:
        code = p.get('hierarchical_code', [])
        for level, code_val in enumerate(code):
            level_codes[level][code_val] += 1
    
    # Print distribution at each level
    for level in range(4):
        print(f"\nLevel {level}:")
        print(f"  Unique codes: {len(level_codes[level])}")
        print(f"  Top 3 codes:")
        for code, count in sorted(level_codes[level].items(), key=lambda x: -x[1])[:3]:
            print(f"    Code {code}: {count} products ({count/len(products)*100:.1f}%)")


def main():
    print("\n" + "=" * 60)
    print("NIKE CATALOG LLM - DEMO")
    print("=" * 60 + "\n")
    
    # Load data
    print("Loading data...")
    products, embeddings, vocab, stats = load_data()
    print(f"✓ Loaded {len(products)} products\n")
    
    # Show statistics
    print_statistics(stats, products)
    
    # Analyze token structure
    analyze_token_structure(products)
    
    # Show example products
    print("\n" + "=" * 60)
    print("EXAMPLE PRODUCTS")
    print("=" * 60)
    
    for i in range(min(3, len(products))):
        show_product_with_code(products[i], index=i)
    
    # Find similar products
    if len(products) >= 10:
        print("\n" + "=" * 60)
        print("SIMILARITY SEARCH DEMO")
        print("=" * 60 + "\n")
        
        # Pick a random product
        query_idx = 5
        find_similar_products(query_idx, products, embeddings, top_k=5)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("- Train an LLM to generate token sequences")
    print("- Implement recommendation via token prediction")
    print("- Add text search (query → tokens)")
    print("- Build explanation generation")


if __name__ == "__main__":
    main()
