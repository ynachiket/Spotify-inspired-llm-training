"""
Multimodal Product Encoder
Uses open source models (CLIP + Sentence Transformers) to create product embeddings
and generate hierarchical token codes similar to the Spotify approach
"""
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict
import open_clip
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalProductEncoder:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        logger.info("Loading CLIP model...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.clip_model.eval()
        
        logger.info("Loading text encoder...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Clustering models will be fitted later
        self.clustering_hierarchy = None
        self.token_vocab = {}
        
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image using CLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using Sentence Transformers"""
        try:
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def encode_product(self, product: Dict) -> Dict:
        """Create multimodal embedding for a product"""
        embeddings = []
        
        # Text embedding
        text_parts = []
        if product.get('title'):
            text_parts.append(product['title'])
        if product.get('subtitle'):
            text_parts.append(product['subtitle'])
        if product.get('description'):
            text_parts.append(product['description'])
        if product.get('color'):
            text_parts.append(f"Color: {product['color']}")
        if product.get('features'):
            text_parts.extend(product['features'])
        
        text_combined = " ".join(text_parts)
        text_emb = self.encode_text(text_combined)
        if text_emb is not None:
            embeddings.append(('text', text_emb))
        
        # Image embeddings (average multiple images)
        image_embs = []
        for img_info in product.get('downloaded_images', [])[:4]:  # Use first 4 images
            img_path = self.data_dir / img_info['local_path']
            img_emb = self.encode_image(str(img_path))
            if img_emb is not None:
                image_embs.append(img_emb)
        
        if image_embs:
            avg_image_emb = np.mean(image_embs, axis=0)
            embeddings.append(('image', avg_image_emb))
        
        return {
            'product_id': product['id'],
            'text_embedding': text_emb.tolist() if text_emb is not None else None,
            'image_embedding': avg_image_emb.tolist() if image_embs else None,
            'text': text_combined
        }
    
    def create_multimodal_embedding(self, text_emb: np.ndarray, image_emb: np.ndarray, 
                                   text_weight=0.3, image_weight=0.7) -> np.ndarray:
        """Combine text and image embeddings"""
        # Normalize
        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        image_norm = image_emb / (np.linalg.norm(image_emb) + 1e-8)
        
        # Weighted combination
        combined = np.concatenate([
            text_norm * text_weight,
            image_norm * image_weight
        ])
        
        return combined / (np.linalg.norm(combined) + 1e-8)
    
    def build_hierarchical_clustering(self, embeddings: List[np.ndarray], 
                                     num_levels=4, clusters_per_level=None):
        """
        Build hierarchical clustering similar to Spotify's coarse-to-fine approach
        Each product gets a 4-level code like [17, 203, 8, 91]
        """
        if clusters_per_level is None:
            # Similar to Spotify: 4 levels that can address billions of combinations
            clusters_per_level = [32, 64, 128, 256]  # Total vocab: ~7,000 tokens
        
        logger.info(f"Building {num_levels}-level hierarchical clustering...")
        logger.info(f"Clusters per level: {clusters_per_level}")
        
        embeddings_array = np.array(embeddings)
        n_samples = len(embeddings_array)
        
        self.clustering_hierarchy = []
        current_embeddings = embeddings_array
        current_indices = np.arange(n_samples)
        
        # Map from original index to hierarchical code
        product_codes = [[] for _ in range(n_samples)]
        
        for level, n_clusters in enumerate(clusters_per_level):
            logger.info(f"Level {level}: clustering {len(current_embeddings)} items into {n_clusters} clusters")
            
            # Adjust cluster count if needed
            n_clusters = min(n_clusters, len(current_embeddings))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(current_embeddings)
            
            self.clustering_hierarchy.append({
                'level': level,
                'n_clusters': n_clusters,
                'kmeans': kmeans,
                'cluster_sizes': np.bincount(labels)
            })
            
            # Assign codes at this level
            for i, idx in enumerate(current_indices):
                product_codes[idx].append(int(labels[i]))
            
            # For next level, we would typically recurse into each cluster
            # For simplicity, we'll continue with all items but the codes will naturally group similar items
        
        return product_codes
    
    def build_token_vocabulary(self, product_codes: List[List[int]]):
        """
        Build vocabulary mapping hierarchical codes to token IDs
        Each unique code path gets a unique token (like Spotify's approach)
        """
        logger.info("Building token vocabulary...")
        
        token_id = 0
        self.token_vocab = {}
        
        for code_path in product_codes:
            # Convert code path to tuple for hashing
            code_key = tuple(code_path)
            
            if code_key not in self.token_vocab:
                self.token_vocab[code_key] = {
                    'token_id': token_id,
                    'code': code_path,
                    'token_str': '_'.join(map(str, code_path))
                }
                token_id += 1
        
        logger.info(f"Created vocabulary with {len(self.token_vocab)} unique tokens")
        return self.token_vocab
    
    def process_catalog(self, catalog_path: str = None):
        """Process entire product catalog"""
        if catalog_path is None:
            catalog_path = self.data_dir / "catalog.json"
        
        logger.info(f"Loading catalog from {catalog_path}")
        with open(catalog_path, 'r') as f:
            products = json.load(f)
        
        logger.info(f"Processing {len(products)} products...")
        
        # Encode all products
        product_embeddings = []
        valid_products = []
        
        for product in tqdm(products, desc="Encoding products"):
            emb_data = self.encode_product(product)
            
            if emb_data['text_embedding'] and emb_data['image_embedding']:
                # Create combined multimodal embedding
                combined_emb = self.create_multimodal_embedding(
                    np.array(emb_data['text_embedding']),
                    np.array(emb_data['image_embedding'])
                )
                product_embeddings.append(combined_emb)
                valid_products.append({**product, **emb_data})
        
        logger.info(f"Successfully encoded {len(product_embeddings)} products")
        
        # Build hierarchical clustering
        product_codes = self.build_hierarchical_clustering(product_embeddings)
        
        # Build token vocabulary
        token_vocab = self.build_token_vocabulary(product_codes)
        
        # Assign codes to products
        for i, product in enumerate(valid_products):
            product['hierarchical_code'] = product_codes[i]
            product['token_id'] = self.token_vocab[tuple(product_codes[i])]['token_id']
            product['token_str'] = self.token_vocab[tuple(product_codes[i])]['token_str']
        
        # Save results
        self._save_results(valid_products, product_embeddings, token_vocab)
        
        return valid_products, product_embeddings, token_vocab
    
    def _save_results(self, products, embeddings, token_vocab):
        """Save all results"""
        # Save encoded products
        output_file = self.model_dir / "encoded_products.json"
        with open(output_file, 'w') as f:
            json.dump(products, f, indent=2)
        logger.info(f"Saved encoded products to {output_file}")
        
        # Save embeddings
        embeddings_file = self.model_dir / "embeddings.npy"
        np.save(embeddings_file, np.array(embeddings))
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Save clustering models
        clustering_file = self.model_dir / "clustering_hierarchy.pkl"
        with open(clustering_file, 'wb') as f:
            pickle.dump(self.clustering_hierarchy, f)
        logger.info(f"Saved clustering hierarchy to {clustering_file}")
        
        # Save token vocabulary
        vocab_file = self.model_dir / "token_vocabulary.json"
        with open(vocab_file, 'w') as f:
            # Convert tuple keys to strings for JSON
            vocab_serializable = {
                str(k): v for k, v in token_vocab.items()
            }
            json.dump(vocab_serializable, f, indent=2)
        logger.info(f"Saved token vocabulary to {vocab_file}")
        
        # Save statistics
        stats = {
            'total_products': len(products),
            'total_tokens': len(token_vocab),
            'vocabulary_size': len(token_vocab),
            'clustering_levels': len(self.clustering_hierarchy),
            'clusters_per_level': [h['n_clusters'] for h in self.clustering_hierarchy]
        }
        
        stats_file = self.model_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")


def main():
    encoder = MultimodalProductEncoder(data_dir="data", model_dir="models")
    products, embeddings, vocab = encoder.process_catalog()
    
    print(f"\n✓ Processed {len(products)} products")
    print(f"✓ Created {len(vocab)} unique tokens")
    print(f"✓ Models saved to {encoder.model_dir}")


if __name__ == "__main__":
    main()
