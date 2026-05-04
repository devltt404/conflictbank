# ConflictBank Survey

This repository contains the code used to reproduce and extend the [CONFLICTBANK benchmark](https://arxiv.org/abs/2408.12076) on Qwen 2.5 (0.5B, 3B, 7B) and SmolLM2 (0.36B). The experiments cover retrieved-knowledge conflicts, embedded-knowledge conflicts, and conflict-description effects.

---

## Requirements

All experiments were run on **Google Colab Pro** with a single NVIDIA A100 GPU (80 GB VRAM). A high-VRAM GPU is required for inference on the 7B model and for continual pre-training.

Install dependencies:

```bash
pip install -r requirements.txt
pip install vllm
```

For Experiment 2, LLaMA Factory is also required:

```bash
pip install "llamafactory[torch,metrics]"
```

---

## Datasets

The experiments use two datasets from the CONFLICTBANK release on Hugging Face:

- **QA pairs:** [`Warrieryes/CB_qa`](https://huggingface.co/datasets/Warrieryes/CB_qa)
- **Claim-evidence pairs:** [`Warrieryes/CB_claim_evidence`](https://huggingface.co/datasets/Warrieryes/CB_claim_evidence)

Both datasets are downloaded automatically by the notebooks.

---

## Experiment 1: Retrieved Knowledge Conflicts and Conflict Descriptions

This experiment evaluates model behavior when conflicting evidence is provided at inference time. It also includes the conflict-description variant, where a temporal or semantic description is appended to the question.

**Notebook:** [ConflictBank_Survey_E1.ipynb](ConflictBank_Survey_E1.ipynb)

**Models evaluated:** Qwen2.5-0.5B, Qwen2.5-3B, Qwen2.5-7B, SmolLM2-360M

Open the notebook and follow the cells in order. The notebook handles dataset download, prompt generation ([prompt.py](prompt.py)), inference ([inference.py](inference.py)), and evaluation ([evaluate.py](evaluate.py)). Run it once per model by setting `MODEL_FULL_PATH` in the configuration cell at the top. After running, the results are zipped and saved to `My Drive/Colab_Output/` on your Google Drive. The zip file contains the `results/` folder with the following `.xlsx` output files and a copy of the notebook.

| File | Provides |
|---|---|
| `results/<model>/context_conflict.xlsx` | Accuracy and Memorization Ratio per conflict type under single conflicting evidence |
| `results/<model>/inter_conflict.xlsx` | Same metrics under two-evidence setting (default + conflicting evidence) |
| `results/<model>/description.xlsx` | Memorization Ratio with and without conflict-context description, for temporal and semantic conflict types |

---

## Experiment 2: Embedded Knowledge Conflicts

This experiment performs continual pre-training with conflicting evidence and measures how it affects the model's internal knowledge.

**Models used:** Qwen2.5-0.5B, Qwen2.5-3B (the 7B model exceeds available GPU memory for training)

**Pre-trained checkpoints** (already available on Hugging Face):

| Checkpoint | Base model | Training ratio |
|---|---|---|
| `introtollm/qwen2.5-0.5B-cb-1_0` | Qwen2.5-0.5B | 1:0 (control) |
| `introtollm/qwen2.5-0.5B-cb-1_1` | Qwen2.5-0.5B | 1:1 (conflict) |
| `introtollm/qwen2.5-3B-cb-1_0`   | Qwen2.5-3B   | 1:0 (control) |
| `introtollm/qwen2.5-3B-cb-1_1`   | Qwen2.5-3B   | 1:1 (conflict) |

If you want to use these checkpoints directly, skip to **Step 3**.

### Steps

**Step 1 — Build the known-correct QA subset**

Run [exp2_build_known_QA_dataset.ipynb](exp2_build_known_QA_dataset.ipynb). This identifies 100 QA pairs that the base models answer correctly both with and without default evidence. The output is saved as `known_subset.json`.

**Step 2 — Run continual pre-training**

Run [exp2_run_continual_pretraining.ipynb](exp2_run_continual_pretraining.ipynb). Set the model size and conflict ratio at the top of the notebook:

```python
MODEL_SIZE     = "0.5B"   # "0.5B" or "3B"
CONFLICT_RATIO = "1:1"    # "1:0" (control) or "1:1" (conflict)
```

The notebook will:
1. Stream 50,000 evidence passages from `Warrieryes/CB_claim_evidence`.
2. Build a balanced JSONL training corpus (equal proportions of misinformation, temporal, and semantic conflict).
3. Write a LLaMA Factory training config and run continual pre-training.

Training hyperparameters (fixed):

| Parameter | Value |
|---|---|
| Batch size | 1 |
| Gradient accumulation | 8 |
| Max sequence length | 1024 tokens |
| Max steps | 2109 |
| Learning rate | 2 × 10⁻⁵ (cosine schedule) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight decay=0.1) |
| Precision | bfloat16 |

Run this notebook four times to produce the four checkpoints (0.5B × {1:0, 1:1} and 3B × {1:0, 1:1}).

**Step 3 — Upload checkpoints** *(optional)*

Run [exp2_upload_model_to_huggingface.ipynb](exp2_upload_model_to_huggingface.ipynb) to push a trained checkpoint to Hugging Face Hub. This step is needed only if you retrained the models.

**Step 4 — Evaluate**

Run [exp2_main_run.ipynb](exp2_main_run.ipynb). Upload `known_subset.json` to `/content/` in Colab. The notebook loads all four checkpoints from Hugging Face, evaluates them on the 100-question subset, and produces:
- OAR without evidence
- OAR with default evidence prepended

Results are saved to `results_exp3_3.json` and plotted as `figure6_replica.pdf`.

---

## Output Files

| File | Description |
|---|---|
| `results/<model>/context_conflict.xlsx` | Accuracy and Memorization Ratio per conflict type under single conflicting evidence. |
| `results/<model>/inter_conflict.xlsx` | Same metrics under two-evidence setting (default + conflicting). |
| `results/<model>/description.xlsx` | Memorization Ratio with and without conflict-context description, for temporal and semantic types. |
| `results_exp3_3.json` | OAR results for embedded conflicts (Exp 2) |
| `figure6_replica.pdf` | Bar chart replicating Figure 6 of original paper (Exp 2) |


