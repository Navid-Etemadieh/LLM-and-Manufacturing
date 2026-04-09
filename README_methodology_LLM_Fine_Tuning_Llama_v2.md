# Methodology README for `LLM_Fine_Tuning_Llama_v2.ipynb`

## 1. High-level goal of the fine-tuning
At a methodological level, this notebook is trying to **adapt a large base Llama model to a manufacturing-focused setting** using a two-stage supervised process. The model starts as a general-purpose pretrained language model (`unsloth/llama-3-70b-bnb-4bit`) and is then adapted to better handle domain material and structured instruction-response behavior.

The adaptation appears to target two different capabilities:
1. **Domain acclimatization (Stage 1):** expose the model to text extracted from local PDF files (`books` folder), so the model’s behavior shifts toward the vocabulary and patterns in that corpus.
2. **Instruction-following specialization (Stage 2):** train on JSON examples with `instruction`, `input`, and `response` fields, using a consistent prompt template (`### Instruction`, `### Input`, `### Response`).

The intended improvement over the base model is therefore not “new pretraining from scratch,” but a **task- and domain-aligned behavior shift**: better responses for the target instruction format and manufacturing-related context, while using efficient adaptation methods that fit constrained hardware.

---

## 2. What type of fine-tuning is being used?
The notebook combines multiple adaptation ideas:

- **PEFT / LoRA-based adaptation** (via `FastLanguageModel.get_peft_model`)  
- **Quantized base model loading** (`load_in_4bit=True`)  
- **Supervised fine-tuning (SFT)** using `trl.SFTTrainer`  
- **Instruction tuning** in Stage 2 through explicit prompt-response formatting  
- **Domain adaptation** in Stage 1 through PDF text chunks

### What this means
- **Not full fine-tuning:** The notebook does not update all base model weights. It applies LoRA adapters to selected modules and trains those adapters.
- **Parameter-efficient adaptation:** Most base parameters stay frozen, while low-rank adapter weights learn task/domain adjustments.
- **SFT framing:** The model is trained to predict training text continuations based on supervised text examples.

### Why someone would use this approach
- Much lower memory footprint than full fine-tuning.
- Faster iteration and cheaper experimentation.
- Practical for very large models (70B-class) on limited resources.
- Ability to keep base model unchanged and store only adapters.

### Tradeoffs
- Usually less expressive than full-model updating for very difficult adaptation tasks.
- Success is sensitive to data formatting and quality.
- Can overfit style/patterns in narrow datasets.

### Why it may have been selected here
Given the notebook settings (`70B`, 4-bit loading, 8-bit optimizer, gradient checkpointing, LoRA), the methodology clearly prioritizes **feasibility + efficiency** while still enabling meaningful domain/task adaptation.

---

## 3. Full methodology pipeline
Conceptually, the pipeline is:

1. **Choose a pretrained base model**  
   A large Llama model is loaded in quantized form for memory efficiency.

2. **Load tokenizer and define sequence limits**  
   `max_seq_length=2048` sets the maximum training/inference context window used here.

3. **Attach LoRA adapters once**  
   The same adapted model object is reused across Stage 1 and Stage 2.

4. **Stage 1 data preparation (PDF domain corpus)**  
   PDF text is extracted, normalized, split into fixed-size word chunks, and wrapped into a text dataset.

5. **Stage 1 SFT training**  
   The model is trained on domain text chunks to nudge internal behavior toward domain language/statistics.

6. **Stage 2 data preparation (instruction JSON)**  
   JSON records are converted into formatted instruction examples with explicit sections.

7. **Stage 2 SFT training (instruction tuning)**  
   The model continues training with prompt-response style supervision.

8. **Basic post-training inference check**  
   A single test prompt is generated to inspect whether response behavior looks aligned.

9. **Export**  
   The resulting model is exported to GGUF with chosen quantization for downstream inference deployment.

This is effectively a **sequential curriculum**: first adapt to domain text distribution, then tune instruction behavior.

---

## 4. LoRA / PEFT methodology in depth
### What LoRA is
LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into selected linear layers. Instead of rewriting the full weight matrix `W`, it learns a small update `ΔW ≈ A·B` where rank(`ΔW`) is constrained by `r`.

### Why LoRA differs from full fine-tuning
- **Full fine-tuning:** update essentially all model weights.
- **LoRA fine-tuning:** freeze base model, update only adapter parameters in chosen layers.

This reduces memory and optimizer state overhead dramatically.

### Intuitive meaning of “low-rank adaptation”
Think of the model’s large weight matrix as an enormous control panel. Full tuning moves every knob. LoRA adds a smaller “overlay control panel” with limited degrees of freedom. You can still steer behavior, but with fewer knobs.

### What updates vs what stays frozen
- **Updated:** LoRA adapter parameters on target modules.
- **Frozen (mostly):** original backbone weights.

### Why this helps efficiency
- Fewer trainable params → less VRAM and faster optimization.
- Compatible with large models and quantized loading.

### Practical limitations
- May struggle if the target task requires very deep/global rewiring.
- Adapter capacity (rank, targets) can bottleneck performance.
- Requires thoughtful module targeting and data quality.

### Specific LoRA setup in this notebook
- **`r = 64`**: relatively high adapter capacity for richer behavioral shift.
- **`lora_alpha = 128`**: scaling factor controlling effective update magnitude.
- **`lora_dropout = 0.0`**: no dropout regularization on LoRA path; may improve fitting speed but can raise overfitting risk.
- **`bias = "none"`**: bias terms are not adapted through LoRA config.
- **`use_gradient_checkpointing = "unsloth"`**: memory-saving mechanism during backprop.
- **`use_rslora = False`** and **`loftq_config = None`**: standard LoRA path, no RS-LoRA or LoftQ variant.

Methodologically, this is a **high-capacity, efficiency-aware LoRA setup** designed to remain practical on constrained hardware while still allowing substantial adaptation.

---

## 5. LoRA target modules: what they are and what each one means
The notebook targets:
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

These cover **both attention projections and MLP projections**, which is broader than attention-only LoRA.

### `q_proj` (query projection)
- **Where:** self-attention block.
- **Function:** maps hidden states into query vectors used to ask “what should I attend to?”
- **Why adapt:** changes what kinds of patterns the model seeks in context.
- **Likely effect:** can alter prompt interpretation and focus behavior.

### `k_proj` (key projection)
- **Where:** self-attention block.
- **Function:** maps states into key vectors that queries match against.
- **Why adapt:** influences how token representations are indexed/retrieved by attention.
- **Likely effect:** shifts which contextual features are considered relevant.

### `v_proj` (value projection)
- **Where:** self-attention block.
- **Function:** projects content that attention actually aggregates.
- **Why adapt:** changes the content payload flowing through attention.
- **Likely effect:** modifies factual/stylistic content composition in generated text.

### `o_proj` (output projection of attention)
- **Where:** end of attention sublayer.
- **Function:** remixes concatenated attention outputs back into model hidden space.
- **Why adapt:** controls how attended information is integrated for downstream layers.
- **Likely effect:** impacts overall expression and coherence of attended information.

### `gate_proj` (MLP gating projection)
- **Where:** feed-forward/MLP block.
- **Function:** contributes to gating behavior (which nonlinear features are emphasized/suppressed).
- **Why adapt:** helps shape internal feature selection beyond attention.
- **Likely effect:** stronger task-style/domain-specific transformation patterns.

### `up_proj` (MLP expansion projection)
- **Where:** feed-forward block.
- **Function:** expands hidden dimension into a larger intermediate representation.
- **Why adapt:** affects how rich candidate features are created.
- **Likely effect:** can improve adaptation of nuanced domain concepts.

### `down_proj` (MLP compression projection)
- **Where:** feed-forward block.
- **Function:** projects expanded intermediate features back to model dimension.
- **Why adapt:** controls which transformed features return to residual stream.
- **Likely effect:** influences final token-level decision shaping.

### Why target both attention and MLP modules?
- **Attention-only adaptation** mostly changes information routing and contextual focus.
- **Attention + MLP adaptation** also changes feature transformation/composition.

So this notebook chooses **broader behavioral control** at higher adaptation capacity/cost than minimal LoRA, while still far cheaper than full fine-tuning.

---

## 6. Training data methodology
### What data is used
Two training sources:
1. **Stage 1:** raw text extracted from PDFs in a local folder.
2. **Stage 2:** structured JSON examples with `instruction`, `input`, `response`.

### How examples are structured
- Stage 1 becomes plain text chunks (50 words each).
- Stage 2 is explicitly formatted into a prompt template:
  - `### Instruction: ...`
  - `### Input: ...`
  - `### Response: ...<|endoftext|>`

### What the model is being taught
- Stage 1: domain language patterns and terminology distribution.
- Stage 2: response behavior for instruction-style prompts and output formatting.

This most closely resembles **instruction tuning with domain adaptation**, not pure classification.

### Why formatting matters
The model learns patterns it sees. If the training prompt schema is consistent, inference with the same schema tends to produce aligned behavior. If schema is inconsistent, model behavior becomes unstable or mixed-style.

### Risks from poor formatting
- Ambiguous boundaries between prompt and target.
- Inconsistent section headers leading to unreliable format compliance.
- Noise in labels/targets causing degraded instruction following.

### Good alignment means
Task objective, example format, and inference prompt style should all match. This notebook partially enforces that by using the same sectioned format during training and test inference.

---

## 7. Tokenization and sequence methodology
- **Tokenization** converts text into subword units the model optimizes over.
- **`max_seq_length=2048`** defines maximum token budget per example.
- **Truncation/padding behavior** (managed by trainer/tokenizer stack) determines whether long samples are cut and how batch shapes are standardized.

Why this matters methodologically:
- If samples exceed sequence length, critical supervision may be truncated.
- If samples are too short/noisy, context-learning signal weakens.
- Inconsistent structure near sequence boundaries can harm learned formatting behavior.

For this notebook, short fixed-size PDF chunks (50 words) reduce truncation risk in Stage 1, but may also reduce long-context learning signal. Stage 2 formatted prompts may vary more and need careful monitoring for cutoff effects.

---

## 8. Quantization and efficiency methodology
The base model is loaded with **4-bit quantization** (`load_in_4bit=True`) and trained with LoRA adapters.

### What quantization means
Quantization stores model weights in lower precision (here, low-bit format), reducing memory and often improving throughput.

### How it differs from fine-tuning
- Quantization is a **representation/computation strategy**.
- Fine-tuning is a **learning/update strategy**.

### LoRA + quantization interaction
This is the standard efficiency pattern often associated with QLoRA-like workflows: keep large backbone in low-bit form, train small adapter parameters (higher precision where needed).

### Advantages
- Makes very large models practical on limited hardware.
- Enables experimentation with 70B-class base models.

### Possible limitations
- Quantized backbones can introduce approximation error.
- Training dynamics may be more sensitive to hyperparameters/data quality.
- Final quality may lag full-precision full fine-tuning for some tasks.

---

## 9. Training hyperparameters: what they mean methodologically
Important settings visible in notebook config:

- **Learning rate (`2e-4` in both stages):** controls update magnitude. Too high can destabilize; too low can under-adapt.
- **Batch size (`per_device_train_batch_size=2`) + gradient accumulation (`4`):** gives effective larger batch while fitting memory constraints.
- **Epochs (Stage1=`1`, Stage2=`3`):** Stage 1 is brief domain shift; Stage 2 emphasizes instruction behavior more heavily.
- **Warmup steps (`10`):** smooths early optimization to reduce instability.
- **Weight decay (`0.01`):** mild regularization to reduce overfitting tendencies.
- **Scheduler (`linear`):** straightforward LR decay strategy.
- **Optimizer (`adamw_8bit`):** memory-efficient optimizer aligned with large-model constraints.
- **Mixed precision (`fp16`/`bf16` based on hardware support):** speed/memory optimization with some numerical tradeoffs.
- **Save strategy (`epoch`):** checkpointing at epoch boundaries, practical but coarse-grained.

Methodological interpretation: the configuration is tuned for **resource-aware stable training** rather than aggressive experimentation with extensive validation control.

---

## 10. What does it mean for the model to be “fine-tuned successfully”?
A successful fine-tuning should show:
- Better adherence to desired instruction-response format.
- More domain-appropriate terminology/logic on relevant prompts.
- Clear improvement relative to base model on the target task distribution.
- Stable outputs across similar prompts.

### Warning signs
- **Underfitting:** minimal behavior change; generic responses; no domain/style improvement.
- **Overfitting:** brittle outputs, memorized phrases, poor generalization to unseen prompts.
- **No meaningful adaptation:** training ran, but before/after outputs are nearly identical on target prompts.

### Interpreting training loss
Lower training loss is necessary but not sufficient. It can reflect memorization or over-specialization. True success requires **generalization checks** and **qualitative task evaluation**.

### Fair comparison method
Use identical prompt sets, decoding settings, and evaluation criteria for:
1. Base model
2. Fine-tuned model

Then compare both quantitative outcomes and human-judged quality.

---

## 11. Evaluation methodology
### Explicitly present in the notebook
- Training execution in two stages and printed trainer stats.
- One post-training qualitative generation test prompt.

### What is missing or weak
- No explicit validation split usage.
- No validation loss tracking.
- No systematic held-out benchmark.
- No formal base-vs-finetuned paired evaluation table.

### Broader evaluation methodology that should be used
- Track training + validation loss curves.
- Hold out task-representative prompts not seen in training.
- Compare base vs fine-tuned outputs side by side.
- Score format compliance and domain correctness.
- Add human judgment rubric (accuracy, usefulness, clarity, safety).
- Test consistency across paraphrases and edge cases.

In short: notebook includes a basic smoke test, but not a full validation methodology.

---

## 12. Possible evaluation metrics for this fine-tuning
Given instruction-style generation with domain grounding, useful metrics include:

- **Loss / perplexity (validation):** language modeling fit; useful trend indicator, not direct task utility.
- **Format compliance rate:** percent outputs following required schema; important for structured workflows.
- **Task success rate:** proportion of prompts meeting expected intent-specific criteria.
- **Human rubric scores:** factual correctness, relevance, specificity, helpfulness.
- **Pairwise preference vs base model:** head-to-head judgments on same prompts.
- **Domain correctness checks:** accuracy of manufacturing interpretations/terminology.

Potential but limited depending on task:
- **Exact Match / Accuracy / F1:** good for classification or tightly constrained QA, less suitable for open-ended outputs.
- **BLEU/ROUGE:** n-gram overlap metrics can miss semantic quality and can be misleading for generative tasks.

Methodological rule: use multiple metrics; no single metric fully captures instruction-tuned model quality.

---

## 13. How to verify that the fine-tuning changed the model behavior
Practical verification steps:

1. Build a fixed prompt suite (in-domain, out-of-domain, edge cases).
2. Run both base and fine-tuned models with identical decoding params.
3. Compare:
   - domain terminology use,
   - instruction adherence,
   - response structure consistency,
   - error/hallucination patterns,
   - usefulness for target application.
4. Quantify with rubric + compliance metrics.

### Convincing evidence
- Repeated improvement across many unseen prompts.
- Better format adherence with equal or better factual quality.
- Reduced failure modes relevant to the intended task.

### Weak evidence
- One or two cherry-picked examples.
- Improvement only on near-duplicate training prompts.
- Claiming success solely from reduced training loss.

---

## 14. Common methodological mistakes and risks
Typical risks for this methodology:
- Inconsistent prompt template between train and inference.
- Low-quality or noisy `response` targets.
- Too little data for desired behavior shift.
- Too many epochs causing overfitting.
- Adapting modules without enough evaluation to justify choices.
- Relying on training loss only.
- Confusing memorized phrasing with robust improvement.
- Saving/exporting model but not validating adapter effect rigorously.

Notebook-specific risk signals:
- Minimal explicit validation framework.
- Single qualitative test prompt is insufficient evidence.

---

## 15. Strengths and limitations of the methodology in this notebook
### Methodological strengths
- Clear two-stage adaptation logic (domain then instruction).
- Efficient large-model strategy (4-bit + LoRA + 8-bit optimizer + checkpointing).
- Broad LoRA targeting across attention and MLP for richer adaptation capacity.
- Consistent prompt schema in Stage 2 training and inference test.

### Methodological limitations
- No robust validation pipeline shown.
- No explicit held-out evaluation set.
- No formal base-vs-finetuned comparative analysis.
- Stage 1 chunking method may simplify/alter text structure in ways that could affect quality.

### Improvements to make claims more convincing
- Add train/validation split and loss tracking.
- Add systematic evaluation prompts and rubric scoring.
- Report before/after comparisons quantitatively.
- Add ablation checks (e.g., Stage 2 only vs Stage 1+2).
- Measure format compliance and domain correctness explicitly.

---

## 16. Teacher-style summary
If you teach this notebook as a method, here is the core idea:

> We are not rebuilding a language model from scratch. We take a strong pretrained Llama, keep its core intelligence mostly fixed, and teach it a new “specialization layer” through LoRA adapters. First, we immerse it in domain text so it speaks the domain language more naturally. Then we train it on instruction-response examples so it behaves in the format and style we want. We do this with quantization and parameter-efficient learning so the process is feasible on limited hardware.

The method is practical and modern: **efficient adaptation of a huge model** using staged supervision. But a good teacher would also stress this: training is only half the story. Without disciplined evaluation (validation sets, before/after comparisons, human review), you cannot confidently claim true improvement.

---

## 17. Observed vs inferred
### Directly observed in the notebook
- Base model: `unsloth/llama-3-70b-bnb-4bit` with `load_in_4bit=True`.
- LoRA attachment via `FastLanguageModel.get_peft_model`.
- LoRA settings: `r=64`, `alpha=128`, `dropout=0.0`, `bias="none"`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Two-stage training: Stage 1 on PDF-derived text, Stage 2 on instruction JSON.
- Stage 2 prompt template with `Instruction/Input/Response` markers.
- Training via `SFTTrainer` and `TrainingArguments`.
- Basic single-prompt post-training generation test.
- GGUF export with `q4_k_m` quantization setting.

### Reasonable methodological inferences
- The overall approach is a PEFT + SFT instruction/domain adaptation pipeline.
- Stage ordering suggests intended curriculum (domain acclimation before behavior shaping).
- Broader module targeting indicates intent for stronger behavioral flexibility than attention-only LoRA.
- Evaluation methodology is currently limited; stronger validation is needed to verify generalization.

