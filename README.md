# Nike Catalog LLM Prototype

A prototype implementation of Spotify's catalog-grounded LLM approach applied to Nike's product catalog, using multimodal (text + vision) embeddings.

## Overview

This project implements the methodology from "A Unified Language Model for Large Scale Search, Recommendation, and Reasoning" (De Nadai et al., 2026) but adapted for e-commerce with:

1. **Multimodal embeddings**: Combines CLIP (vision) + Sentence Transformers (text)
2. **Hierarchical token codes**: Each product gets a 4-level code like `[17, 203, 8, 91]`
3. **Open source models**: No API costs, fully self-hosted
4. **Visual similarity**: Product clustering captures style, color, and silhouette

## Architecture

```
Nike Product → [CLIP Vision + Text Encoder] → Multimodal Embedding
                                                      ↓
                                            Hierarchical Clustering
                                                      ↓
                                              4-Level Token Code
                                                (17_203_8_91)
```

## Project Structure

```
nike-catalog-llm/
├── scraper.py              # Crawl Nike product pages
├── encoder.py              # Create multimodal embeddings & token codes
├── demo.py                 # Demo notebook/script
├── requirements.txt        # Python dependencies
├── data/
│   ├── catalog.json       # Scraped product metadata
│   ├── images/            # Downloaded product images
│   └── products/          # Individual product JSONs
└── models/
    ├── encoded_products.json    # Products with embeddings & codes
    ├── embeddings.npy           # Numpy array of embeddings
    ├── clustering_hierarchy.pkl # Trained clustering models
    ├── token_vocabulary.json    # Token ID mappings
    └── statistics.json          # Summary statistics
```

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers (for web scraping)
playwright install chromium
```

## Usage

### Step 1: Scrape Nike Products

```bash
python scraper.py
```

This will:
- Discover 500 Nike product URLs from multiple categories
- Extract product details (title, description, price, features)
- Download high-quality product images (multiple angles per product)
- Save to `data/catalog.json`

### Step 2: Create Multimodal Embeddings & Token Codes

```bash
python encoder.py
```

This will:
- Load CLIP (ViT-B/32) and Sentence Transformers
- Encode each product's text and images
- Create hierarchical clustering (4 levels: 32→64→128→256 clusters)
- Generate unique token codes for each product
- Save models and token vocabulary

### Step 3: Explore Results

```bash
jupyter notebook notebooks/explore_embeddings.ipynb
```

Or run the demo script:
```bash
python demo.py
```

## Models Used (Open Source)

| Component | Model | Source |
|-----------|-------|--------|
| **Vision** | CLIP ViT-B/32 | OpenCLIP (LAION-2B) |
| **Text** | all-MiniLM-L6-v2 | Sentence Transformers |
| **Clustering** | K-Means (hierarchical) | scikit-learn |

## Token Vocabulary Design

Following Spotify's approach:

- **4 hierarchical levels**: Coarse to fine clustering
- **Token format**: `level0_level1_level2_level3` (e.g., `17_203_8_91`)
- **Vocabulary size**: ~7,000 tokens for 500 products
- **Semantic structure**: Similar products share prefixes

Example:
```
Product A: 17_203_8_91   (Nike Air Max 90, red/white)
Product B: 17_203_8_114  (Nike Air Max 90, blue/white)  ← Shares prefix!
Product C: 17_203_54_22  (Nike Air Max 95, silver)      ← Same subcategory
Product D: 8_112_7_203   (Nike Dri-FIT shirt)           ← Different category
```

## Key Advantages for Nike

1. **Visual similarity captured**: Clustering uses image embeddings
2. **Style-based recommendations**: "Shoes that look like these"
3. **Cross-category matching**: "Shoes to match this outfit"
4. **Efficient token space**: 7K tokens vs millions of embeddings
5. **Interpretable codes**: Hierarchical structure maps to product taxonomy

## Next Steps

- [ ] Train a small LLM to generate token sequences
- [ ] Implement recommendation via next-token prediction
- [ ] Add search functionality (text → tokens)
- [ ] Build "explain recommendations" capability
- [ ] Deploy as API service

## Performance Expectations

Based on Spotify's paper:
- **Search hit rate**: +40-60% vs baseline
- **Recommendation quality**: +50-80% for users with history
- **Cold start**: -40-50% for new users (needs separate fallback)

## References

- **Paper**: "A Unified Language Model for Large Scale Search, Recommendation, and Reasoning", De Nadai et al., 2026
- **ArXiv**: https://arxiv.org/abs/2603.17533
- **OpenCLIP**: https://github.com/mlfoundations/open_clip
- **Sentence Transformers**: https://www.sbert.net/

## License

MIT (for prototype code)

Note: Nike product data is scraped for research/educational purposes. Respect Nike's terms of service and robots.txt.
