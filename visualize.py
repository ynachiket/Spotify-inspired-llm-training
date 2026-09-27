"""
Visualize product embeddings and clustering
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_embeddings(output_file="visualization.png"):
    """Create 2D visualization of product embeddings"""
    models_dir = Path("models")
    
    # Load data
    with open(models_dir / "encoded_products.json", 'r') as f:
        products = json.load(f)
    
    embeddings = np.load(models_dir / "embeddings.npy")
    
    with open(models_dir / "statistics.json", 'r') as f:
        stats = json.load(f)
    
    print(f"Loaded {len(products)} products")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Extract hierarchical codes
    level0_codes = [p['hierarchical_code'][0] for p in products if p.get('hierarchical_code')]
    level1_codes = [p['hierarchical_code'][1] for p in products if p.get('hierarchical_code')]
    
    # Dimensionality reduction
    print("Computing PCA...")
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(products)-1))
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Color by top-level category (level 0)
    scatter1 = axes[0].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=level0_codes,
        cmap='tab20',
        s=50,
        alpha=0.7
    )
    axes[0].set_title(f'Nike Products by Top-Level Category\n({len(set(level0_codes))} categories)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0], label='Category Code (Level 0)')
    
    # Plot 2: Color by second-level category (level 1)
    scatter2 = axes[1].scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=level1_codes,
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    axes[1].set_title(f'Nike Products by Sub-Category\n({len(set(level1_codes))} subcategories)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1], label='Category Code (Level 1)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")
    
    # Print some statistics
    print("\n" + "="*60)
    print("CLUSTERING STATISTICS")
    print("="*60)
    print(f"Total products:          {len(products)}")
    print(f"Total unique tokens:     {stats['total_tokens']}")
    print(f"Vocabulary size:         {stats['vocabulary_size']}")
    print(f"Level 0 categories:      {len(set(level0_codes))}")
    print(f"Level 1 subcategories:   {len(set(level1_codes))}")
    
    # Category distribution
    from collections import Counter
    level0_dist = Counter(level0_codes)
    print("\nTop 5 Level-0 Categories:")
    for code, count in level0_dist.most_common(5):
        print(f"  Category {code}: {count} products ({count/len(products)*100:.1f}%)")
    
    return embeddings_2d, level0_codes, level1_codes


def show_example_clusters(n_examples=3):
    """Show example products from each top-level cluster"""
    models_dir = Path("models")
    
    with open(models_dir / "encoded_products.json", 'r') as f:
        products = json.load(f)
    
    from collections import defaultdict
    clusters = defaultdict(list)
    
    for p in products:
        if p.get('hierarchical_code'):
            level0 = p['hierarchical_code'][0]
            clusters[level0].append(p)
    
    print("\n" + "="*60)
    print("EXAMPLE PRODUCTS FROM EACH CLUSTER")
    print("="*60)
    
    for code in sorted(clusters.keys())[:5]:  # Top 5 clusters
        prods = clusters[code][:n_examples]
        print(f"\nCluster {code} ({len(clusters[code])} products):")
        print("-" * 60)
        for p in prods:
            title = p.get('title', 'N/A')
            subtitle = p.get('subtitle', '')
            token = p.get('token_str', 'N/A')
            print(f"  • {title}")
            print(f"    {subtitle}")
            print(f"    Token: {token}")
            print()


if __name__ == "__main__":
    embeddings_2d, level0, level1 = visualize_embeddings("nike_embeddings_visualization.png")
    show_example_clusters(n_examples=3)
