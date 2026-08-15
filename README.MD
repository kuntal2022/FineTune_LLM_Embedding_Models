# Fine-Tuning My First Embedding Model — BGE-small-en-v1.5

**Author:** Kuntal (Prem) Chakraborty
**Date:** August 2026
**Reference notes:** *How an Embedding Model Gets Trained* (Parts 1–4, study notes)

---

## 1. Objective

To build hands-on, practical understanding of embedding model fine-tuning — going beyond theory (per-token vectors, pooling, [CLS] vs mean pooling, contrastive loss, InfoNCE, Matryoshka Representation Learning) into an actual end-to-end fine-tuning run: data preparation, training, evaluation, debugging, and honest interpretation of results.

This was done as targeted interview prep for ML/AI Engineer roles (RAG, embeddings, fine-tuning), following on from the CampusX Advanced RAG coursework and the theory notes referenced above.

---

## 2. Base Model

**BAAI/bge-small-en-v1.5**

| Property | Value |
|---|---|
| Parameters | ~33M |
| Embedding dimension | 384 |
| Architecture | 12-layer transformer encoder (BERT-base style) |
| Pooling | Mean pooling (per the notes — Part 1 & 2: mean pooling is the standard choice for bi-encoders, since a reusable, general-purpose sentence vector benefits from every token's contribution rather than depending on a single [CLS] bottleneck) |

Chosen because it's lightweight enough to fine-tune and run inference on CPU, while still being a real, production-grade embedding model (used as a backbone in many RAG systems).

---

## 3. Theory Recap (from notes)

The four-part study notes covered the full pipeline this project actually implements:

1. **Part 1 — Architecture & Pooling**: Transformer gives per-token vectors, not one sentence vector. Mean pooling averages all token vectors into a single 384/768-dim representation.
2. **Part 2 — [CLS] vs Mean Pooling**: Bi-encoders prefer mean pooling (generalizable); cross-encoders use [CLS] because the query and document interact directly via self-attention before the classifier head — a narrower, less lossy task.
3. **Part 3 — Contrastive Training**: A randomly initialized encoder is meaningless until trained on (anchor, positive) pairs. Naive distance-minimization collapses trivially, so negatives are introduced — first via **Triplet Loss**, then via **in-batch negatives** and **InfoNCE (Multiple Negatives Ranking Loss)** for a stronger, denser training signal.
4. **Part 4 — Matryoshka Representation Learning**: Nested InfoNCE losses on prefix slices (64/128/256/1024) let front dimensions absorb more gradient signal, enabling later truncation without retraining. *(Not implemented in this run — BGE-small's 384-dim output was used as-is; this remains a natural next experiment.)*

This project puts Part 3 into direct practice — Triplet Loss and InfoNCE were both implemented and compared on the same fine-tuning task.

---

## 4. Dataset

**`sentence-transformers/all-nli`** (triplet configuration) — loaded from Hugging Face.

```python
from datasets import load_dataset
ds = load_dataset("sentence-transformers/all-nli", "triplet")
```

| Split | Rows | Columns |
|---|---|---|
| train | 557,850 | anchor, positive, negative |
| dev | 6,584 | anchor, positive, negative |
| test | 6,609 | anchor, positive, negative |

For practice, a subset (~1,500 rows) was used from `train` to keep CPU training time reasonable (~10–20 minutes per run).

**Example row:**
```json
{
  "anchor": "A person on a horse jumps over a broken down airplane.",
  "positive": "A person is outdoors, on a horse.",
  "negative": "A person is at a diner, ordering an omelette."
}
```

---

## 5. Training Setup

Two loss functions were implemented and compared, both directly from Part 3 of the theory notes.

### 5.1 Triplet Loss

```
Loss = max(0, d(anchor, positive) − d(anchor, negative) + margin)
```

```python
from sentence_transformers.losses import TripletLoss
train_loss = TripletLoss(model=model)
```

- Compares one anchor against **one** negative at a time — weaker signal, slower convergence (as noted in the theory).

### 5.2 InfoNCE (Multiple Negatives Ranking Loss)

```
Loss = −log( exp(sim(a,pos)) / Σ exp(sim(a,neg_i)) )
```

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss
train_loss = MultipleNegativesRankingLoss(model=model, scale=20.0)  # scale = 1/τ
```

- Uses **in-batch negatives** — every other pair's positive in the batch becomes a free negative.
- `scale=20.0` corresponds to temperature τ ≈ 0.05, the standard value used by BGE/E5.

### 5.3 Training Configuration

```python
args = SentenceTransformerTrainingArguments(
    output_dir="ktl-bge-small-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=50,
)
```

Trained on **CPU** using `SentenceTransformerTrainer` (the current sentence-transformers v3+ API — the older `model.fit()` + `InputExample` pattern is deprecated).

---

## 6. Debugging Log — Real Issues Hit During This Run

Documenting these because they were the most valuable part of the exercise — the kind of practical debugging an interviewer would want to hear about.

| Issue | Root Cause | Fix |
|---|---|---|
| `Unsupported input type: list` | Old `InputExample`/`DataLoader` API incompatible with sentence-transformers v3+ | Switched to `Dataset.from_dict()` + `SentenceTransformerTrainer` |
| Base model accuracy = 27% (should be ~90%+) | `positive`/`negative` fields were plain strings, but code did `row["positive"][0]` — this sliced the **first character** of the string, not a list element | Removed `[0]` indexing once field type was confirmed via `type(sample["positive"])` |
| Fine-tuned accuracy = 100% | **Data leakage** — evaluation rows were drawn from the same rows used in training | Switched evaluation to the untouched `ds["dev"]` split, which was never seen during training |
| Fine-tuned ≈ Base accuracy (~97–98%) after fixing leakage | **Ceiling effect** — AllNLI is a generic, "easy" benchmark; base BGE-small is already near-saturated on it, leaving little room for measurable improvement | Understood as an expected outcome, not a training failure — documented below |

---

## 7. Results

Evaluated using `TripletEvaluator` (cosine similarity accuracy: % of triplets where `sim(anchor, positive) > sim(anchor, negative)`), on the clean, non-overlapping `dev` split.

| Model | Cosine Accuracy |
|---|---|
| Base BGE-small (no fine-tuning) | 98.0% |
| Fine-tuned — Triplet Loss | 97.0% |
| Fine-tuned — InfoNCE | 97.0% |

**Manual sanity check** (single triplet, post-bug-fix):
```
anchor–positive similarity: 0.7987
anchor–negative similarity: 0.4749
```
Correct ranking confirmed — the model reliably separates positive from negative pairs.

---

## 8. Interpretation

Fine-tuning on AllNLI produced **no measurable gain over the base model** — and that's an expected, honest result, not a failure of the pipeline:

- AllNLI is a broad, generic sentence-similarity dataset. BGE-small was already pre-trained and fine-tuned (by BAAI) on large-scale pair data including NLI-style corpora, so it starts near the accuracy ceiling on this exact kind of data.
- With only ~1,500 rows and 3 epochs on CPU, there simply wasn't a meaningful "knowledge gap" for the model to close.
- **The real value of fine-tuning shows up on domain-specific data** the base model has never seen — e.g., retail/CVM vocabulary (Shukran points, ATV Booster, Winback, campaign terminology) — where the base model's generic embeddings would genuinely under-perform, and fine-tuning would produce a visible accuracy gap.

This is the natural next step for this project.

---

## 9. How the Fine-Tuned Model Is Used

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("ktl-bge-small-finetuned")

query_embedding = model.encode("How can I return a product?")
doc_embeddings = model.encode([
    "You can return items within 30 days.",
    "Store timings are 10 AM to 9 PM.",
])

scores = cos_sim(query_embedding, doc_embeddings)
# tensor([[0.7211, 0.4921]]) → correctly ranks the return-policy doc highest
```

For RAG integration (LangChain):
```python
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="ktl-bge-small-finetuned")
```

---

## 10. Key Takeaways (Interview-Ready)

1. **Fine-tuning is a form of transfer learning** — starting from a pre-trained encoder and further training it with a contrastive objective (Triplet Loss or InfoNCE) on domain pairs, not training from scratch.
2. **InfoNCE > Triplet Loss** in principle — it compares an anchor against *all* in-batch negatives at once (softmax-style), giving a denser gradient signal per step than triplet loss's one-negative-at-a-time comparison.
3. **Evaluation methodology matters as much as training** — data leakage (evaluating on rows seen during training) silently inflates accuracy to unrealistic levels (100%); a clean, held-out split is essential.
4. **A saturated benchmark hides real gains** — fine-tuning shows its value only when the base model has a genuine knowledge gap to close, i.e., on domain-specific, out-of-distribution data.
5. **Matryoshka Representation Learning** (Part 4 of the notes) is the natural next layer on top of this pipeline — nesting InfoNCE losses across prefix dimensions (64/128/256/384) to enable dimension truncation at inference without retraining.

---

## 11. Next Steps

- [ ] Build a small (500–2,000 row) **domain-specific** query–pair dataset from MAX Fashion / CVM context (campaign names, loyalty program terms, customer analytics vocabulary).
- [ ] Re-run the same Triplet Loss vs InfoNCE comparison on this domain data — expect a visible, non-trivial accuracy gap this time.
- [ ] Extend training with Matryoshka-style nested losses to test dimension truncation (384 → 256 → 128) without retraining.
- [ ] Push the fine-tuned model to Hugging Face Hub for reuse across the `agentic-rag-parent-self-fusion` project.