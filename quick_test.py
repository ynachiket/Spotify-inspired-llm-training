"""
Quick test - single inference
"""
from test_llm import NikeTokenLLM

print("Loading model...")
llm = NikeTokenLLM(model_path="trained_model/final")

print("\n" + "="*60)
print("QUICK TEST - Token to Description")
print("="*60)

test_tokens = ['10_14_14_14', '0_4_97_97']

for token in test_tokens:
    print(f"\nToken: {token}")
    
    # Get actual product
    product = llm.find_product_by_token(token)
    if product:
        actual = f"{product.get('title', 'Unknown')} - {product.get('subtitle', '')}"
        print(f"Actual: {actual}")
    
    # Generate (with lower max_length for speed)
    print("Generating...")
    description = llm.describe_product(token, temperature=0.3)
    print(f"Generated: {description}")

print("\n" + "="*60)
print("DONE!")
