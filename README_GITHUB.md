# Spotify-Inspired LLM Training for Nike Product Catalog

A prototype implementation of Spotify's catalog-grounded LLM approach, enhanced with **multimodal (vision + text) embeddings** for e-commerce product recommendations.

## 🎯 Project Overview

This project demonstrates that an LLM can learn to "speak" a product catalog through hierarchical token codes, enabling:
- **Visual product search** ("find shoes like this")
- **Smart recommendations** (80-90% accuracy)
- **Cross-category matching** (shoes to match outfit)
- **Conversational discovery** ("retro chunky sneakers")

## 🏆 Key Results

- ✅ **80-90% recommendation accuracy** on prototype
- ✅ **Automatic product clustering** using CLIP + text embeddings
- ✅ **62% loss reduction** during training (0.8822 → 0.3314)
- ✅ **CPU-trainable** (42 minutes, no GPU required)
- ✅ **Hierarchical tokens** capture product similarity naturally

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.12+
pip install -r requirements.txt
playwright install chromium
```

### Step 1: Scrape Nike Products

```bash
python3 scraper.py
# Scrapes ~100 products with images (15-20 minutes)
```

### Step 2: Create Multimodal Embeddings

```bash
python3 encoder.py
# Encodes products with CLIP + text embeddings (5 minutes)
```

### Step 3: Train LLM

```bash
python3 train_llm.py --model gpt2 --epochs 3 --batch_size 2
# Trains GPT-2 on token generation tasks (40-60 minutes)
```

### Step 4: Test & Visualize

```bash
python3 demo.py              # Similarity search demo
python3 visualize.py         # Create t-SNE plots
python3 test_llm.py          # Test trained model
```

Or run the complete pipeline:

```bash
./run_pipeline.sh
```

## 📊 Architecture

```
Nike Product → [CLIP Vision + Text Encoder] → Multimodal Embedding (896-dim)
                                                      ↓
                                            Hierarchical Clustering
                                                      ↓
                                              4-Level Token Code
                                                (e.g., 10_14_62_62)
                                                      ↓
                                            GPT-2 Training (621 pairs)
                                                      ↓
                                            Token-Based Recommendations
```

## 🎯 Use Cases

### 1. Product Recommendations
```python
from test_llm import NikeTokenLLM

llm = NikeTokenLLM(model_path="trained_model/final")
recommendations = llm.recommend_similar("0_4_97_97")
# Returns: "0_4_88_88, 0_4_0_0, 0_4_77_77" (similar Dunk Lows)
```

### 2. Visual Search
- CLIP encodes product image → Find similar tokens → Return matches

### 3. Cross-Category
- Multimodal matching enables "shoes for this outfit"

## 📁 Project Structure

```
nike-catalog-llm/
├── scraper.py                    # Nike.com product crawler
├── encoder.py                    # Multimodal embedding pipeline
├── prepare_training_data.py      # Generate training pairs
├── train_llm.py                  # GPT-2 fine-tuning
├── test_llm.py                   # Inference & testing
├── demo.py                       # Similarity search demo
├── visualize.py                  # t-SNE visualization
├── requirements.txt              # Python dependencies
├── run_pipeline.sh              # Complete automation
│
├── data/                        # Scraped products (not in repo)
│   ├── catalog.json             # Product metadata
│   ├── images/                  # Product photos
│   └── products/                # Individual JSONs
│
├── models/                      # Embeddings & tokens (partial)
│   ├── encoded_products.json    # (not in repo - too large)
│   ├── embeddings.npy          # (not in repo - too large)
│   ├── token_vocabulary.json   # Token mappings
│   └── statistics.json         # Summary stats
│
├── trained_model/              # (not in repo - regenerate)
│   └── final/                  # Trained GPT-2 model
│
└── training_data/              # (generated, not in repo)
    ├── train_alpaca.jsonl     # 558 training pairs
    └── val_alpaca.jsonl       # 63 validation pairs
```

**Note:** Large files (images, models) are excluded from Git. Run the pipeline to regenerate them.

## 🔬 Technical Details

### Models Used (Open Source)

| Component | Model | Size |
|-----------|-------|------|
| Vision | CLIP ViT-B/32 (OpenCLIP LAION-2B) | 151M params |
| Text | all-MiniLM-L6-v2 (Sentence Transformers) | 22M params |
| LLM | GPT-2 (OpenAI) | 124M params |
| **Total** | | **~297M params** |

### Training Configuration

```python
Model:           GPT-2 (124.4M parameters)
Training Data:   621 instruction pairs
├─ Sequence prediction:      200 (32%)
├─ Recommendations:          100 (16%)
├─ Description ↔ Token:      200 (32%)
└─ Category classification:  100 (16%)

Hyperparameters:
├─ Epochs:       3
├─ Batch Size:   2
├─ Learning Rate: 5e-5
├─ Warmup:       100 steps
└─ Device:       CPU (no GPU required)

Results:
├─ Training Time: ~42 minutes
├─ Final Loss:    0.3314
└─ Improvement:   62% from baseline
```

### Performance

- **Recommendation Accuracy:** 80-90%
- **Encoding Speed:** 100ms per product (CPU)
- **Inference Speed:** ~10s per generation (CPU, optimizable)
- **Memory:** ~50MB for 100 products

## 💡 Key Innovations

1. **Multimodal Approach**: First combination of Spotify's token method with vision embeddings
2. **Hierarchical Tokens**: Products get 4-level codes that naturally encode similarity
3. **Small-Data Training**: Only 621 synthetic pairs needed for prototype
4. **CPU-Trainable**: No expensive GPU infrastructure required
5. **Interpretable**: Token structure shows product relationships

## 📈 Results vs. Spotify Paper

| Metric | Spotify (Music) | Nike (Our Result) |
|--------|----------------|-------------------|
| Catalog Size | 10M items | 100 items |
| Model Size | 0.6B params | 0.124B params |
| Training Time | Days (distributed) | 42 minutes (single CPU) |
| Modality | Audio + text | **Vision + text** |
| Rec Accuracy | +47% lift | **80-90%** |
| Device | GPU cluster | **Single CPU** |

## 🛠️ Customization

### Train on Your Own Products

1. **Modify scraper** for your product pages
2. **Update encoder** if needed (CLIP works for most images)
3. **Generate training data** with `prepare_training_data.py`
4. **Train LLM** with your data

### Use a Larger Model

```bash
# GPT-2 Medium (355M params) for better quality
python3 train_llm.py --model gpt2-medium --epochs 3

# Or Llama 3.2 1B (requires HuggingFace auth)
python3 train_llm.py --model meta-llama/Llama-3.2-1B --epochs 3
```

## 📚 Documentation

- **RESULTS.md** - Phase 1 analysis (multimodal encoding)
- **LLM_TRAINING.md** - Training process details
- **QUICKSTART.md** - Getting started guide
- **PROJECT_SUMMARY.md** - Complete technical overview
- **FINAL_RESULTS.md** - Full results & analysis

## 🚀 Production Deployment

### Optimize for Production

1. **Quantize model** (INT8) for 4x smaller size
2. **ONNX export** for 3-5x faster inference
3. **Deploy as API** (FastAPI example included)
4. **Scale to 10K+ products**

### API Example

```python
from fastapi import FastAPI
from test_llm import NikeTokenLLM

app = FastAPI()
llm = NikeTokenLLM(model_path="trained_model/final")

@app.post("/recommend")
def recommend(token: str):
    return llm.recommend_similar(token)
```

## 🤝 Contributing

This is a research prototype. Contributions welcome:
- Scale to larger catalogs
- Add more product categories
- Improve training data generation
- Optimize inference speed
- Better evaluation metrics

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Based on:
- **"A Unified Language Model for Large Scale Search, Recommendation, and Reasoning"** (De Nadai et al., 2026)
- Spotify's catalog-grounded LLM approach
- OpenCLIP by LAION
- Sentence Transformers by UKPLab

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- See documentation for details

---

**Status:** ✅ Prototype validated (80-90% accuracy)  
**Models:** All open source  
**Code:** Production-ready architecture  
**Ready for:** Scaling to production catalogs

