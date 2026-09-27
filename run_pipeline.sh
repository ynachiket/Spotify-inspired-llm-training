#!/bin/bash
# Run the complete Nike Catalog LLM pipeline

set -e

echo "=================================="
echo "Nike Catalog LLM Pipeline"
echo "=================================="
echo

# Check if products exist
PRODUCT_COUNT=$(ls data/products/*.json 2>/dev/null | wc -l)
echo "Found $PRODUCT_COUNT products"

if [ "$PRODUCT_COUNT" -lt 50 ]; then
    echo "Error: Need at least 50 products. Run scraper.py first."
    exit 1
fi

# Run encoder
echo
echo "Step 1: Creating multimodal embeddings..."
echo "-----------------------------------"
python3 encoder.py

if [ ! -f "models/encoded_products.json" ]; then
    echo "Error: Encoding failed"
    exit 1
fi

echo
echo "✓ Encoding complete"

# Run demo
echo
echo "Step 2: Running demo analysis..."
echo "-----------------------------------"
python3 demo.py

# Create visualization
echo
echo "Step 3: Creating visualizations..."
echo "-----------------------------------"
python3 visualize.py

echo
echo "=================================="
echo "Pipeline Complete!"
echo "=================================="
echo
echo "Results:"
echo "  - Encoded products: models/encoded_products.json"
echo "  - Token vocabulary: models/token_vocabulary.json"
echo "  - Embeddings: models/embeddings.npy"
echo "  - Visualization: nike_embeddings_visualization.png"
echo
echo "Next steps:"
echo "  - Train LLM to generate token sequences"
echo "  - Implement recommendation engine"
echo "  - Build search interface"
