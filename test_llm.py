"""
Test and inference script for trained Nike token LLM
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NikeTokenLLM:
    def __init__(self, model_path="trained_model/final"):
        """Load trained model for inference"""
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading trained model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        
        # Load product catalog for reference
        with open("models/encoded_products.json", 'r') as f:
            self.products = json.load(f)
        
        # Create token to product mapping
        self.token_to_product = {
            p['token_str']: p for p in self.products if p.get('token_str')
        }
    
    def format_prompt(self, instruction: str, input_text: str) -> str:
        """Format prompt for inference"""
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    def generate(self, instruction: str, input_text: str, 
                max_length=150, temperature=0.7, top_p=0.9) -> str:
        """Generate response"""
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response portion
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def find_product_by_token(self, token_str: str):
        """Look up product by token"""
        return self.token_to_product.get(token_str)
    
    def describe_product(self, token_str: str, temperature=0.3) -> str:
        """Generate description from token"""
        return self.generate(
            instruction="Describe the product with token code:",
            input_text=token_str,
            temperature=temperature,
            max_length=100
        )
    
    def find_token(self, description: str, temperature=0.3) -> str:
        """Find token from description"""
        return self.generate(
            instruction="Find the token code for this product:",
            input_text=description,
            temperature=temperature,
            max_length=50
        )
    
    def recommend_similar(self, token_str: str, temperature=0.5) -> str:
        """Recommend similar products"""
        return self.generate(
            instruction="Recommend similar products. Input is a token code, output is comma-separated token codes:",
            input_text=token_str,
            temperature=temperature,
            max_length=80
        )
    
    def predict_next(self, history: str, temperature=0.7) -> str:
        """Predict next product from browsing history"""
        return self.generate(
            instruction="Given user's browsing history (token sequence), predict the next product they will view:",
            input_text=history,
            temperature=temperature,
            max_length=50
        )
    
    def classify_category(self, description: str, temperature=0.3) -> str:
        """Predict category from description"""
        return self.generate(
            instruction="What is the category code (level 0) for this product?",
            input_text=description,
            temperature=temperature,
            max_length=30
        )


def run_interactive_demo(llm: NikeTokenLLM):
    """Interactive demo of the trained model"""
    print("\n" + "="*70)
    print("NIKE TOKEN LLM - INTERACTIVE DEMO")
    print("="*70)
    
    demos = [
        {
            'name': 'Token → Description',
            'tokens': ['10_14_14_14', '0_4_97_97', '2_5_85_85'],
        },
        {
            'name': 'Description → Token',
            'descriptions': [
                'Nike Air Max 90. Men\'s Shoes',
                'Nike Pegasus Premium. Men\'s Road Running Shoes',
                'Nike Dunk Low Retro SE. Men\'s Shoes'
            ]
        },
        {
            'name': 'Similarity Recommendation',
            'tokens': ['0_4_97_97', '10_14_14_14'],
        },
        {
            'name': 'Next Product Prediction',
            'histories': [
                '0_4_97_97 -> 0_4_77_77 -> 0_4_0_0',
                '10_14_14_14 -> 10_14_86_86'
            ]
        }
    ]
    
    for demo in demos:
        print(f"\n{'='*70}")
        print(f"DEMO: {demo['name']}")
        print('='*70)
        
        if 'tokens' in demo and demo['name'].startswith('Token'):
            for token in demo['tokens']:
                print(f"\nInput token: {token}")
                
                # Get actual product
                product = llm.find_product_by_token(token)
                if product:
                    actual = f"{product.get('title', 'Unknown')} - {product.get('subtitle', '')}"
                    print(f"Actual product: {actual}")
                
                # Generate description
                description = llm.describe_product(token)
                print(f"LLM generated: {description}")
        
        elif 'descriptions' in demo:
            for desc in demo['descriptions']:
                print(f"\nInput description: {desc}")
                
                # Generate token
                token = llm.find_token(desc)
                print(f"LLM predicted token: {token}")
                
                # Try to look up
                product = llm.find_product_by_token(token.strip())
                if product:
                    print(f"✓ Found product: {product.get('title')}")
                else:
                    print("✗ Token not found in catalog (may be hallucination)")
        
        elif 'tokens' in demo and 'Recommendation' in demo['name']:
            for token in demo['tokens']:
                print(f"\nQuery token: {token}")
                
                product = llm.find_product_by_token(token)
                if product:
                    print(f"Query product: {product.get('title')}")
                
                # Get recommendations
                recs = llm.recommend_similar(token)
                print(f"LLM recommendations: {recs}")
                
                # Validate recommendations
                rec_tokens = [t.strip() for t in recs.split(',')]
                print("Recommended products:")
                for rt in rec_tokens[:3]:
                    prod = llm.find_product_by_token(rt)
                    if prod:
                        print(f"  • {rt}: {prod.get('title')} ✓")
                    else:
                        print(f"  • {rt}: Unknown ✗")
        
        elif 'histories' in demo:
            for history in demo['histories']:
                print(f"\nBrowsing history: {history}")
                
                # Show history products
                history_tokens = [t.strip() for t in history.split('->')]
                print("User viewed:")
                for ht in history_tokens:
                    prod = llm.find_product_by_token(ht)
                    if prod:
                        print(f"  • {prod.get('title')}")
                
                # Predict next
                next_token = llm.predict_next(history)
                print(f"LLM predicts next: {next_token}")
                
                prod = llm.find_product_by_token(next_token.strip())
                if prod:
                    print(f"✓ Predicted: {prod.get('title')}")
                else:
                    print("✗ Token not found")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


def run_benchmark(llm: NikeTokenLLM):
    """Benchmark the model's performance"""
    print("\n" + "="*70)
    print("BENCHMARK: Model Accuracy")
    print("="*70)
    
    # Test 1: Token → Description (does it mention the right product?)
    print("\nTest 1: Token → Description")
    correct = 0
    total = 0
    
    for product in llm.products[:20]:  # Test on first 20
        if not product.get('token_str'):
            continue
        
        token = product['token_str']
        actual_title = product.get('title', '').lower()
        
        generated = llm.describe_product(token)
        
        # Simple check: is the product name mentioned?
        matches = any(word in generated.lower() for word in actual_title.split()[:3])
        
        if matches:
            correct += 1
        total += 1
    
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    
    # Test 2: Description → Token (does it return a valid token?)
    print("\nTest 2: Description → Token")
    valid = 0
    total = 0
    
    for product in llm.products[:20]:
        if not product.get('token_str'):
            continue
        
        desc = f"{product.get('title')}. {product.get('subtitle')}"
        predicted_token = llm.find_token(desc).strip()
        
        # Check if it's a valid token
        if predicted_token in llm.token_to_product:
            valid += 1
        total += 1
    
    print(f"Valid tokens: {valid}/{total} = {valid/total*100:.1f}%")
    
    print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='trained_model/final',
                       help='Path to trained model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark tests')
    
    args = parser.parse_args()
    
    # Load model
    llm = NikeTokenLLM(model_path=args.model_path)
    
    # Run demo
    run_interactive_demo(llm)
    
    if args.benchmark:
        run_benchmark(llm)


if __name__ == "__main__":
    main()
