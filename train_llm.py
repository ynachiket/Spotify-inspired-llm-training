"""
Fine-tune LLM for Nike product token generation
Uses GPT-2 or small Llama model for efficiency
"""
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenLLMTrainer:
    def __init__(self, model_name="gpt2", output_dir="trained_model"):
        """
        Initialize trainer with a small open-source model
        
        Options:
        - "gpt2" (124M params) - fast, good for prototype
        - "gpt2-medium" (355M params) - better quality
        - "meta-llama/Llama-3.2-1B" (1B params) - state of the art, requires auth
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU, float16 for GPU
        )
        
        # Resize token embeddings if we added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
    
    def format_prompt(self, instruction: str, input_text: str, output: str = None) -> str:
        """Format data as instruction-following prompt"""
        if output:
            # Training format
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            # Inference format
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    def prepare_dataset(self, data_dir="training_data"):
        """Load and prepare training dataset"""
        data_dir = Path(data_dir)
        
        logger.info("Loading training data...")
        
        # Load JSONL files
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(data_dir / 'train_alpaca.jsonl'),
                'validation': str(data_dir / 'val_alpaca.jsonl')
            }
        )
        
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Val samples: {len(dataset['validation'])}")
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Format prompts
            prompts = [
                self.format_prompt(inst, inp, out)
                for inst, inp, out in zip(
                    examples['instruction'],
                    examples['input'],
                    examples['output']
                )
            ]
            
            # Tokenize
            result = self.tokenizer(
                prompts,
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            
            # Labels are the same as input_ids for causal LM
            result['labels'] = result['input_ids'].copy()
            
            return result
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def train(self, data_dir="training_data", num_epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune the model"""
        
        # Prepare dataset
        dataset = self.prepare_dataset(data_dir)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",  # Disable wandb/tensorboard for now
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        # Evaluate
        logger.info("Evaluating...")
        eval_results = trainer.evaluate()
        
        with open(self.output_dir / "eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation results: {eval_results}")
        
        return trainer
    
    def generate(self, instruction: str, input_text: str, max_length=100, temperature=0.7):
        """Generate response for a given instruction/input"""
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response


def test_model(trainer: TokenLLMTrainer):
    """Test the trained model with example queries"""
    logger.info("\n" + "="*60)
    logger.info("TESTING TRAINED MODEL")
    logger.info("="*60)
    
    test_cases = [
        {
            'instruction': "Describe the product with token code 10_14_14_14:",
            'input': "10_14_14_14",
            'task': "Token to Description"
        },
        {
            'instruction': "Find the token code for this product:",
            'input': "Nike Air Max 90. Men's Shoes",
            'task': "Description to Token"
        },
        {
            'instruction': "Recommend similar products. Input is a token code, output is comma-separated token codes:",
            'input': "0_4_97_97",
            'task': "Recommendation"
        },
        {
            'instruction': "What is the category code (level 0) for this product?",
            'input': "Nike Pegasus Premium. Men's Road Running Shoes",
            'task': "Category Classification"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test['task']}")
        logger.info(f"Input: {test['input']}")
        
        response = trainer.generate(
            instruction=test['instruction'],
            input_text=test['input'],
            max_length=200,
            temperature=0.7
        )
        
        logger.info(f"Output: {response}")
    
    logger.info("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', 
                       help='Model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test existing model')
    
    args = parser.parse_args()
    
    trainer = TokenLLMTrainer(
        model_name=args.model,
        output_dir="trained_model"
    )
    
    if not args.test_only:
        # Train
        trainer.train(
            data_dir="training_data",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    
    # Test
    test_model(trainer)
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Model saved to: trained_model/final")


if __name__ == "__main__":
    main()
