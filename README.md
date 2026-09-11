# Vision-RAG Waste Management



University project that classifies a waste image, then answers a handling or recycling question using only retrieved policy and guideline text.

The answer path is **extractive**. Retrieved sentences are selected and cited. No language model invents handling steps.

## Pipeline

```
Image + question
        |
        v
Waste CNN (5 classes + confidence)
        |
        v
FAISS retrieval (all-MiniLM-L6-v2)
        |
        v
Class-aware ranking + extractive answer + citations
```

Example: a photo of a bottle plus *How should glass waste be handled?* → class `Glass` → cited sentence from the EPA recyclables guidance.

## Waste classes

| Class   | Typical items                          |
|---------|----------------------------------------|
| Glass   | bottles, jars, broken glass            |
| Leather | leather / rexine                       |
| Organic | food scraps, kitchen / wet waste       |
| Paper   | paper, cardboard, newspaper            |
| Plastic | plastic bottles, bags, PET             |

## Dataset

Images live under `data/raw/Automatic waste management dataset/`. Splits are stratified 70 / 15 / 15 (`src/split_dataset.py`, seed 42).

| Split      | Images | Glass | Leather | Organic | Paper | Plastic |
|------------|--------|-------|---------|---------|-------|---------|
| Train      | 1,381  | 290   | 204     | 283     | 317   | 287     |
| Validation | 296    | 63    | 44      | 60      | 68    | 61      |
| Test       | 296    | 62    | 43      | 61      | 68    | 62      |
| **Total**  | **1,973** |     |         |         |       |         |

Supported formats: `.jpg`, `.jpeg`, `.png`, `.heic`.

## Image classifier

`WasteCNN` (`src/model.py`) is a small 3-block CNN:

- Input: RGB `128 × 128`
- Conv blocks: 32 → 64 → 128 channels, ReLU, max-pool
- Dense: `128×16×16` → 128 → 5, dropout 0.5
- Train: 20 epochs, batch 32, Adam `lr=0.001`, ImageNet normalization

Checkpoint: `models/best_waste_cnn.pth`.

**Held-out test set (296 images):** accuracy **97.30%**, macro F1 **0.9743**.

| Class        | Precision | Recall | F1     | Support |
|--------------|-----------|--------|--------|---------|
| Glass        | 0.9118    | 1.0000 | 0.9538 | 62      |
| Leather      | 0.9773    | 1.0000 | 0.9885 | 43      |
| Organic      | 0.9839    | 1.0000 | 0.9919 | 61      |
| Paper        | 1.0000    | 0.8824 | 0.9375 | 68      |
| Plastic      | 1.0000    | 1.0000 | 1.0000 | 62      |
| Macro avg    | 0.9746    | 0.9765 | 0.9743 | 296     |
| Accuracy     | 0.9730    | 0.9730 | 0.9730 | 296     |

Most remaining errors are Paper predicted as Glass. Saved under `results/`:

- `dataset_splits.csv` — class counts by split
- `classification_metrics.csv` — accuracy, macro F1, per-class precision / recall / F1
- `classification_report.txt` — sklearn report
- `test_predictions.csv` — image id, true class, predicted class, confidence
- `confusion_matrix.csv` / `confusion_matrix.png`

## Retrieval-augmented answers

### Sources

| ID | Document | Origin |
|----|----------|--------|
| S1 | Solid Waste Management Rules, 2026 | India Gazette |
| S2 | Guidelines for Citizens | Maharashtra Pollution Control Board |
| S3 | How Do I Recycle Common Recyclables | US EPA |
| S4 | Composting | US EPA |
| S5 | Recycling Basics and Benefits | US EPA |

URLs are listed in `rag/documents/source_list.csv`. Text is chunked at 1,000 characters with 200-character overlap (`src/chunk_documents.py`): **559 chunks** (mostly S1).

### Retrieval

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS (`rag/vector_store/waste.index`)
- Search: 30 candidates, then keep the top 5 chunks that match the predicted waste class
- Ranking is intent-aware (`handle` vs `recycle` vs `compost`) and uses a short context window so a class listed in one sentence can be tied to a handling rule in a nearby sentence (needed for leather in S2)

### Grounding rules

- Answers are source sentences, not generated procedures
- Citations `[S2]`, `[S3]`, … are only for sources that contributed selected sentences
- If evidence is too weak: *The available sources do not provide sufficient information to answer this question.*
- `src/test_llm.py` loads Qwen for a separate demo. It is **not** used to write RAG answers

## End-to-end evaluation

`src/end_to_end_evaluation.py` runs **15 cases**: 3 test images per class, each with a class-specific question.

Latest run (`results/end_to_end_evaluation.csv`): **15 / 15** images classified correctly and **15 / 15** answers marked `Supported`. Each row includes `image_id`, `image_path`, classes, confidence, question, retrieved sources, answer, and `support_status`. Answers are stable across the three images of each class:

| Class   | Question | Cited answer (abbrev.) |
|---------|----------|------------------------|
| Glass   | How should glass waste be handled? | Broken glass should not go into the recycling bin. `[S3]` |
| Leather | How should leather waste be handled? | Leather is listed among recyclable waste… sold to the raddiwala or scrap dealer. `[S2]` |
| Organic | How should organic waste be handled? | Dig a compost pit… biodegradable waste… manure. `[S2]` |
| Paper   | How should paper waste be handled? | Most community or office recycling programs accept paper and paper products. `[S3]` |
| Plastic | How should plastic waste be recycled? | Plastic listed among recyclable waste; give to agencies; if segregated at source, they can be recycled. `[S2]` |

## Setup

Run commands from the `vision-rag-waste` directory.

Python 3.10+ and a GPU are useful for training; inference also runs on CPU.

```text
torch
torchvision
pandas
scikit-learn
matplotlib
pillow
pillow-heif
faiss-cpu
sentence-transformers
pypdf
beautifulsoup4
```

Optional, only for `src/test_llm.py`: `transformers` and `Qwen/Qwen2.5-1.5B-Instruct`.

You need locally (these paths are gitignored):

- Dataset under `data/raw/` and CSVs in `data/processed/`
- Classifier weights `models/best_waste_cnn.pth`
- FAISS index `rag/vector_store/waste.index` and `rag/vector_store/metadata.json`

## How to run

**Interactive: image → class → grounded answer**

```bash
python src/end_to_end.py
```

Enter an image path and a question (for example `How should paper waste be handled?`).

**Question only (no image)**

```bash
python src/rag_answer.py
```

**Classify one image**

```bash
python src/classify_image.py
```

**15-case end-to-end evaluation**

```bash
python src/end_to_end_evaluation.py
```

**Classifier test metrics**

```bash
python src/evaluate.py
```

**Train the CNN** (overwrites `models/best_waste_cnn.pth` when validation accuracy improves)

```bash
python src/train.py
```

Rebuild the vector index only if you change the source documents:

```bash
python src/extract_documents.py
python src/chunk_documents.py
python src/build_index.py
```

## Project layout

```text
vision-rag-waste/
├── src/
│   ├── model.py                  # WasteCNN
│   ├── dataset.py                # loaders and 128×128 transforms
│   ├── train.py / evaluate.py
│   ├── classify_image.py
│   ├── retrieve.py               # FAISS search
│   ├── rag_answer.py             # ranking + extractive answer
│   ├── end_to_end.py             # demo
│   └── end_to_end_evaluation.py  # 15-case eval
├── data/processed/               # train / val / test CSVs
├── models/                       # best_waste_cnn.pth
├── rag/documents/                # S1–S5 text + chunks.json
├── rag/vector_store/             # FAISS index
└── results/                      # predictions, confusion matrix, e2e CSV
```
