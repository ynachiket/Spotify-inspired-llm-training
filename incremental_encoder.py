"""
Incremental encoder - process products as they're scraped
"""
import json
import time
from pathlib import Path
from encoder import MultimodalProductEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def watch_and_encode(data_dir="data", model_dir="models", min_products=50):
    """
    Watch for new products and encode them incrementally
    Useful for processing while scraping is still running
    """
    products_dir = Path(data_dir) / "products"
    processed_file = Path(model_dir) / "processed_ids.txt"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load already processed IDs
    processed_ids = set()
    if processed_file.exists():
        with open(processed_file, 'r') as f:
            processed_ids = set(line.strip() for line in f)
    
    encoder = None
    all_products = []
    all_embeddings = []
    
    logger.info(f"Watching {products_dir} for new products...")
    logger.info(f"Will start encoding after {min_products} products")
    
    while True:
        # Find new product files
        product_files = list(products_dir.glob("*.json"))
        new_products = []
        
        for pf in product_files:
            product_id = pf.stem
            if product_id not in processed_ids:
                try:
                    with open(pf, 'r') as f:
                        product = json.load(f)
                    
                    # Only process if has images
                    if product.get('downloaded_images'):
                        new_products.append(product)
                        processed_ids.add(product_id)
                except Exception as e:
                    logger.error(f"Error loading {pf}: {e}")
        
        if new_products:
            logger.info(f"Found {len(new_products)} new products (total: {len(processed_ids)})")
            
            # Initialize encoder on first batch
            if encoder is None and len(processed_ids) >= min_products:
                logger.info("Initializing encoder...")
                encoder = MultimodalProductEncoder(data_dir=data_dir, model_dir=model_dir)
            
            # Encode new products
            if encoder:
                for product in new_products:
                    logger.info(f"Encoding {product['title'][:50]}...")
                    emb_data = encoder.encode_product(product)
                    
                    if emb_data['text_embedding'] and emb_data['image_embedding']:
                        combined_emb = encoder.create_multimodal_embedding(
                            emb_data['text_embedding'],
                            emb_data['image_embedding']
                        )
                        all_embeddings.append(combined_emb)
                        all_products.append({**product, **emb_data})
                
                # Rebuild clustering every 50 products
                if len(all_products) >= 50 and len(all_products) % 50 == 0:
                    logger.info(f"Rebuilding clustering with {len(all_products)} products...")
                    product_codes = encoder.build_hierarchical_clustering(all_embeddings)
                    token_vocab = encoder.build_token_vocabulary(product_codes)
                    
                    for i, product in enumerate(all_products):
                        product['hierarchical_code'] = product_codes[i]
                        product['token_id'] = token_vocab[tuple(product_codes[i])]['token_id']
                        product['token_str'] = token_vocab[tuple(product_codes[i])]['token_str']
                    
                    encoder._save_results(all_products, all_embeddings, token_vocab)
                    logger.info(f"✓ Saved {len(all_products)} products with {len(token_vocab)} tokens")
            
            # Update processed IDs file
            with open(processed_file, 'w') as f:
                f.write('\n'.join(sorted(processed_ids)))
        
        # Check if scraping is complete
        catalog_file = Path(data_dir) / "catalog.json"
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                catalog = json.load(f)
            
            if len(catalog) > 0 and len(processed_ids) >= len(catalog):
                logger.info("All products processed!")
                break
        
        time.sleep(30)  # Check every 30 seconds
    
    # Final encoding
    if encoder and all_products:
        logger.info("Performing final clustering...")
        product_codes = encoder.build_hierarchical_clustering(all_embeddings)
        token_vocab = encoder.build_token_vocabulary(product_codes)
        
        for i, product in enumerate(all_products):
            product['hierarchical_code'] = product_codes[i]
            product['token_id'] = token_vocab[tuple(product_codes[i])]['token_id']
            product['token_str'] = token_vocab[tuple(product_codes[i])]['token_str']
        
        encoder._save_results(all_products, all_embeddings, token_vocab)
        logger.info(f"✓ Final save: {len(all_products)} products with {len(token_vocab)} tokens")


if __name__ == "__main__":
    watch_and_encode(min_products=50)
