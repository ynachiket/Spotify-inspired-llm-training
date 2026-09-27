"""
Prepare training data for LLM fine-tuning
Following Spotify's approach: train on token ↔ description pairs
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import jsonlines


class TrainingDataGenerator:
    def __init__(self, data_dir="models"):
        self.data_dir = Path(data_dir)
        
        # Load encoded products
        with open(self.data_dir / "encoded_products.json", 'r') as f:
            self.products = json.load(f)
        
        print(f"Loaded {len(self.products)} products")
    
    def create_description_from_product(self, product: Dict) -> str:
        """Create a natural language description of a product"""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('subtitle'):
            parts.append(product['subtitle'])
        
        if product.get('color'):
            parts.append(f"Color: {product['color']}")
        
        if product.get('features') and len(product['features']) > 0:
            # Add first 2 features
            parts.extend(product['features'][:2])
        
        return ". ".join(parts)
    
    def generate_token_to_description_pairs(self) -> List[Dict]:
        """Task 1: Given token code, generate description"""
        training_pairs = []
        
        for product in self.products:
            if not product.get('token_str'):
                continue
            
            token_str = product['token_str']
            description = self.create_description_from_product(product)
            
            # Format as instruction-following task
            training_pairs.append({
                'instruction': f"Describe the product with token code {token_str}:",
                'input': token_str,
                'output': description,
                'task_type': 'token_to_description'
            })
        
        return training_pairs
    
    def generate_description_to_token_pairs(self) -> List[Dict]:
        """Task 2: Given description, generate token code"""
        training_pairs = []
        
        for product in self.products:
            if not product.get('token_str'):
                continue
            
            token_str = product['token_str']
            description = self.create_description_from_product(product)
            
            training_pairs.append({
                'instruction': f"Find the token code for this product:",
                'input': description,
                'output': token_str,
                'task_type': 'description_to_token'
            })
        
        return training_pairs
    
    def generate_similarity_pairs(self) -> List[Dict]:
        """Task 3: Given a product, recommend similar products"""
        training_pairs = []
        
        # Load embeddings for similarity
        import numpy as np
        embeddings = np.load(self.data_dir / "embeddings.npy")
        
        for i, product in enumerate(self.products):
            if not product.get('token_str'):
                continue
            
            # Find top 5 similar products
            query_emb = embeddings[i]
            similarities = np.dot(embeddings, query_emb)
            similar_indices = np.argsort(similarities)[::-1][1:6]  # Skip self
            
            query_token = product['token_str']
            similar_tokens = [self.products[idx]['token_str'] for idx in similar_indices 
                            if self.products[idx].get('token_str')]
            
            if len(similar_tokens) >= 3:
                training_pairs.append({
                    'instruction': "Recommend similar products. Input is a token code, output is comma-separated token codes:",
                    'input': query_token,
                    'output': ', '.join(similar_tokens[:3]),
                    'task_type': 'recommendation'
                })
        
        return training_pairs
    
    def generate_prefix_completion_pairs(self) -> List[Dict]:
        """Task 4: Given token prefix, suggest completions"""
        training_pairs = []
        
        # Group products by prefix
        from collections import defaultdict
        prefix_groups = defaultdict(list)
        
        for product in self.products:
            if not product.get('hierarchical_code'):
                continue
            
            code = product['hierarchical_code']
            # Use first 2 levels as prefix
            prefix = f"{code[0]}_{code[1]}"
            full_token = product['token_str']
            prefix_groups[prefix].append(full_token)
        
        # Create training pairs
        for prefix, tokens in prefix_groups.items():
            if len(tokens) >= 2:
                # Sample 2-3 tokens from this prefix
                sample_size = min(3, len(tokens))
                sampled = random.sample(tokens, sample_size)
                
                training_pairs.append({
                    'instruction': "Complete the token prefix with full token codes (comma-separated):",
                    'input': f"Prefix: {prefix}",
                    'output': ', '.join(sampled),
                    'task_type': 'prefix_completion'
                })
        
        return training_pairs
    
    def generate_category_classification_pairs(self) -> List[Dict]:
        """Task 5: Given description, predict top-level category"""
        training_pairs = []
        
        for product in self.products:
            if not product.get('hierarchical_code'):
                continue
            
            description = self.create_description_from_product(product)
            category = product['hierarchical_code'][0]
            
            training_pairs.append({
                'instruction': "What is the category code (level 0) for this product?",
                'input': description,
                'output': str(category),
                'task_type': 'category_classification'
            })
        
        return training_pairs
    
    def generate_sequence_prediction_pairs(self) -> List[Dict]:
        """Task 6: Given sequence of tokens (user history), predict next"""
        training_pairs = []
        
        # Create synthetic user sessions
        random.seed(42)
        num_sessions = 200
        
        for _ in range(num_sessions):
            # Random session length 3-7
            session_length = random.randint(3, 7)
            
            # Sample products for session
            session_products = random.sample(self.products, 
                                           min(session_length, len(self.products)))
            
            # Use all but last as history, last as target
            if len(session_products) >= 3:
                history_tokens = [p['token_str'] for p in session_products[:-1] 
                                if p.get('token_str')]
                target_token = session_products[-1].get('token_str')
                
                if history_tokens and target_token:
                    training_pairs.append({
                        'instruction': "Given user's browsing history (token sequence), predict the next product they will view:",
                        'input': ' -> '.join(history_tokens),
                        'output': target_token,
                        'task_type': 'sequence_prediction'
                    })
        
        return training_pairs
    
    def format_for_instruction_tuning(self, pairs: List[Dict], format_type='alpaca') -> List[Dict]:
        """Format training pairs for instruction tuning"""
        formatted = []
        
        for pair in pairs:
            if format_type == 'alpaca':
                # Alpaca format
                formatted.append({
                    'instruction': pair['instruction'],
                    'input': pair['input'],
                    'output': pair['output'],
                    'task_type': pair['task_type']
                })
            elif format_type == 'chat':
                # Chat format (for newer models)
                formatted.append({
                    'messages': [
                        {'role': 'system', 'content': 'You are a Nike product recommendation assistant. You work with token codes that represent products.'},
                        {'role': 'user', 'content': f"{pair['instruction']}\n{pair['input']}"},
                        {'role': 'assistant', 'content': pair['output']}
                    ],
                    'task_type': pair['task_type']
                })
        
        return formatted
    
    def generate_all_training_data(self, output_dir='training_data'):
        """Generate all training data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating training data...")
        
        # Generate different task types
        all_pairs = []
        
        print("- Token → Description pairs...")
        token_to_desc = self.generate_token_to_description_pairs()
        all_pairs.extend(token_to_desc)
        print(f"  Generated {len(token_to_desc)} pairs")
        
        print("- Description → Token pairs...")
        desc_to_token = self.generate_description_to_token_pairs()
        all_pairs.extend(desc_to_token)
        print(f"  Generated {len(desc_to_token)} pairs")
        
        print("- Similarity/Recommendation pairs...")
        similarity = self.generate_similarity_pairs()
        all_pairs.extend(similarity)
        print(f"  Generated {len(similarity)} pairs")
        
        print("- Prefix completion pairs...")
        prefix = self.generate_prefix_completion_pairs()
        all_pairs.extend(prefix)
        print(f"  Generated {len(prefix)} pairs")
        
        print("- Category classification pairs...")
        category = self.generate_category_classification_pairs()
        all_pairs.extend(category)
        print(f"  Generated {len(category)} pairs")
        
        print("- Sequence prediction pairs...")
        sequence = self.generate_sequence_prediction_pairs()
        all_pairs.extend(sequence)
        print(f"  Generated {len(sequence)} pairs")
        
        print(f"\nTotal training pairs: {len(all_pairs)}")
        
        # Shuffle
        random.shuffle(all_pairs)
        
        # Split train/val
        split_idx = int(0.9 * len(all_pairs))
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        
        print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
        
        # Save in both formats
        for format_type in ['alpaca', 'chat']:
            print(f"\nSaving {format_type} format...")
            
            train_formatted = self.format_for_instruction_tuning(train_pairs, format_type)
            val_formatted = self.format_for_instruction_tuning(val_pairs, format_type)
            
            # Save as JSONL
            train_file = output_dir / f'train_{format_type}.jsonl'
            val_file = output_dir / f'val_{format_type}.jsonl'
            
            with jsonlines.open(train_file, 'w') as f:
                f.write_all(train_formatted)
            
            with jsonlines.open(val_file, 'w') as f:
                f.write_all(val_formatted)
            
            print(f"  Saved {train_file}")
            print(f"  Saved {val_file}")
        
        # Save statistics
        stats = {
            'total_pairs': len(all_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'task_distribution': {
                task: sum(1 for p in all_pairs if p['task_type'] == task)
                for task in set(p['task_type'] for p in all_pairs)
            }
        }
        
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Training data saved to {output_dir}")
        print("\nTask Distribution:")
        for task, count in stats['task_distribution'].items():
            print(f"  {task}: {count} pairs ({count/len(all_pairs)*100:.1f}%)")


def main():
    generator = TrainingDataGenerator(data_dir="models")
    generator.generate_all_training_data(output_dir="training_data")
    
    print("\n" + "="*60)
    print("Next step: Fine-tune LLM on this data")
    print("="*60)


if __name__ == "__main__":
    main()
