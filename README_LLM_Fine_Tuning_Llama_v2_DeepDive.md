# README for `LLM_Fine_Tuning_Llama_v2.ipynb`

## 1. High-level overview
This notebook builds a **two-stage supervised fine-tuning pipeline** for a Llama-family model (loaded through Unsloth in 4-bit mode) and applies LoRA adapters to keep training memory-efficient.

At a high level, the notebook:
1. Installs dependencies in the active Jupyter environment.
2. Defines all paths and hyperparameters in one configuration dataclass (`FineTuneCfg`).
3. Sets up environment variables, Hugging Face cache behavior, optional offline behavior, and token handling.
4. Loads a quantized Llama model through `FastLanguageModel.from_pretrained(...)`.
5. Attaches LoRA adapters once and reuses that adapted model across two training stages.
6. Runs **Stage 1 domain adaptation** on text extracted from PDFs.
7. Runs **Stage 2 instruction tuning** from a JSON dataset with `instruction/input/response` fields.
8. Tests generation on one prompt.
9. Optionally exports the final model to GGUF format and zips it.

So this is not just a single training script; it is an end-to-end notebook pipeline from setup through export, with explicit support for local caches, optional local model snapshots, and optional offline mode.

## 2. Main purpose of the notebook
The practical purpose appears to be:
- First, teach the model domain content from PDF documents (Stage 1).
- Then shape model behavior into instruction-following for a target task format (Stage 2).

This suggests an engineering workflow where raw domain knowledge and instruction-style behavior are separated into two consecutive phases. The notebook’s default test prompt (`"What is my 3D printer doing? Be specific"`) indicates intended usage in a manufacturing/monitoring context, likely using structured sensor-like input in the `### Input:` field.

In short: this notebook is designed to produce a task-adapted, instruction-tuned model that can later be deployed (including GGUF-compatible environments).

## 3. Methodology
### Overall approach
The notebook uses **parameter-efficient fine-tuning (LoRA) on a 4-bit base model** loaded via Unsloth:
- Quantized model loading (`load_in_4bit=True`).
- LoRA adapters attached to attention and MLP projection layers.
- SFT-style training (TRL `SFTTrainer`) in two phases.

### Step-by-step methodology
1. **Environment and package setup**
   - Installs PyTorch companion packages and training stack packages (`unsloth`, `trl`, `peft`, `datasets`, `bitsandbytes`, etc.).
2. **Centralized configuration (`FineTuneCfg`)**
   - Defines project root, input file locations, model ID/local model options, LoRA params, stage-specific training args, and export settings.
3. **Runtime environment checks**
   - Creates required directories.
   - Configures HF cache directories.
   - Optionally forces offline mode.
   - Resolves Hugging Face token from config or environment.
   - Validates required inputs (`books/*.pdf`, `BigDataset.json`).
4. **Model access pre-check**
   - Uses `whoami(token=...)` and downloads a test file from gated Meta-Llama repo.
5. **Optional predownload**
   - Can snapshot-download the full model repo into local storage.
6. **Model load**
   - Loads model + tokenizer through `FastLanguageModel.from_pretrained(...)` from either Hub or local dir.
7. **Attach LoRA once**
   - Applies LoRA with target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
8. **Stage 1 data preparation (PDFs)**
   - Reads each PDF with PyMuPDF.
   - Applies custom text clean/chunk logic.
   - Creates a Hugging Face `Dataset` with one `text` column.
9. **Stage 1 training**
   - Runs SFTTrainer with `dataset_text_field='text'` and stage-1 `TrainingArguments`.
10. **Stage 2 data preparation (JSON instructions)**
    - Expects list of objects with `instruction`, `input`, `response`.
    - Cleans fields (`strip('# ')`).
    - Converts to `Dataset`.
11. **Stage 2 training**
    - Formats each sample into:
      - `### Instruction: ...`
      - `### Input: ...`
      - `### Response: ...<|endoftext|>`
    - Runs SFTTrainer with stage-2 arguments.
12. **Inference test**
    - Builds prompt in same stage-2 format and runs `model.generate`.
13. **Export artifacts**
    - Saves GGUF (`save_pretrained_gguf`) and zips output directory.

## 4. Notebook workflow, section by section
### Section A: Package installation and CUDA sanity check
**Purpose**
- Ensure notebook kernel has required dependencies.

**Important code**
- `subprocess.check_call([... pip install ...])`
- Torch version checks and CUDA availability print.

**Inputs / outputs**
- Input: active Python env.
- Output: installed packages and printed hardware/software info.

**Why it matters**
- The rest of the notebook depends on these packages and CUDA compatibility.

---

### Section B: `FineTuneCfg` dataclass configuration
**Purpose**
- Centralize all user-tunable values.

**Important code and variables**
- Paths: `project_root`, `pdf_subdir`, `instruction_json_name`.
- Model controls: `model_name`, `use_local_model_dir`, `predownload_model_snapshot`, `offline_mode`.
- LoRA: `lora_r=64`, `lora_alpha=128`, `lora_dropout=0.0`, `target_modules=(...)`.
- Stage 1 args: batch size, grad accumulation, epochs=1, LR, scheduler, output dir.
- Stage 2 args: epochs=3, checkpoint controls, etc.
- Export: `save_gguf`, `gguf_quantization_method='q4_k_m'`.

**Inputs / outputs**
- Input: defaults in class (or user edits).
- Output: `cfg` object used throughout all cells.

**Why it matters**
- Makes the notebook easier to retarget without editing training logic cells.

---

### Section C: Environment setup, token resolution, and input checks
**Purpose**
- Prepare directories and environment state for reproducible model/data behavior.

**Important code**
- Sets `HF_HOME`, `HF_HUB_CACHE`, `TOKENIZERS_PARALLELISM`.
- Optional offline vars: `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`.
- Token resolution priority:
  1. `cfg.hf_token_value`
  2. environment variable `cfg.hf_token_env_var`
  3. no token
- Validates `cfg.instruction_json_path` exists and warns if no PDFs.

**Inputs / outputs**
- Input: filesystem + environment vars.
- Output: created folders, resolved token, sanity prints/errors.

**Why it matters**
- Avoids hidden cache paths and catches missing data early.

---

### Section D: Hugging Face access verification
**Purpose**
- Confirm token and gated model access before heavy jobs.

**Important code**
- `whoami(token=hf_token)`
- `hf_hub_download(repo_id='meta-llama/Meta-Llama-3-70B', filename='model.safetensors.index.json', token=hf_token)`

**Inputs / outputs**
- Input: valid token.
- Output: account name + downloaded test file path.

**Why it matters**
- GGUF export path appears to require access to Meta-Llama assets.

---

### Section E: Optional model snapshot predownload
**Purpose**
- Support local/offline workflow.

**Important code**
- `snapshot_download(repo_id=cfg.model_name, local_dir=cfg.local_model_dir, token=...)`

**Inputs / outputs**
- Input: `cfg.predownload_model_snapshot` flag.
- Output: local model repository copy when enabled.

**Why it matters**
- Useful for unstable networks or repeat runs.

---

### Section F: Model and tokenizer loading
**Purpose**
- Instantiate base model in memory-efficient configuration.

**Important code**
- `FastLanguageModel.from_pretrained(...)`
- Uses either `cfg.model_name` or local snapshot directory.

**Inputs / outputs**
- Input: model source + token + max seq len + quantization flags.
- Output: `(model, tokenizer)`.

**Why it matters**
- Core model object used by both training stages.

---

### Section G: LoRA adapter attachment
**Purpose**
- Add trainable adapters once and reuse across stages.

**Important code**
- `attach_lora_once(model, cfg)`
- Checks existing `peft_config` to avoid double wrapping.
- Calls `FastLanguageModel.get_peft_model(...)` with LoRA settings.

**Inputs / outputs**
- Input: base model.
- Output: PEFT-wrapped model.

**Why it matters**
- Preserves memory and allows two-stage training on same adapter set.

---

### Section H: Stage 1 PDF ingestion and chunking
**Purpose**
- Turn PDF corpus into trainable text samples for domain adaptation.

**Important code**
- `extract_text_from_pdf(pdf_path)` with `fitz`.
- `split_paragraphs(...)` custom cleanup/chunking:
  - replace repeated whitespace with placeholder
  - remove all single spaces
  - restore placeholder to space
  - split into words and fixed-size chunks (`cfg.pdf_chunk_size`, default 50)
- Builds `Dataset.from_dict({"text": all_pdf_chunks})`.

**Inputs / outputs**
- Input: all `*.pdf` files in `cfg.pdf_dir`.
- Output: HF `Dataset` (`domain_dataset`) with text chunks.

**Why it matters**
- Provides domain-specific training material before instruction tuning.

**Important caution**
- The text-cleaning logic is unusual and potentially lossy; it may merge tokens incorrectly depending on PDF extraction spacing patterns.

---

### Section I: Stage 1 trainer and training execution
**Purpose**
- Run first fine-tuning stage on PDF chunk dataset.

**Important code**
- `TrainingArguments(...)` with stage-1 config.
- `SFTTrainer(... dataset_text_field='text' ...)`
- Removes `ProgressCallback`, disables tqdm, then `domain_trainer.train()`.

**Inputs / outputs**
- Input: LoRA model + tokenizer + `domain_dataset`.
- Output: trained adapter weights/checkpoints in `cfg.stage1_output_dir`.

**Why it matters**
- Injects domain corpus information into LoRA parameters.

---

### Section J: Stage 2 instruction dataset loading
**Purpose**
- Prepare structured instruction-response dataset for behavior shaping.

**Important code**
- `json.load(...)` from `cfg.instruction_json_path`.
- `clean_text` uses `text.strip('# ')`.
- Expects keys: `instruction`, `input`, `response` for each item.
- Builds dataset via `Dataset.from_dict(separated_data)`.

**Inputs / outputs**
- Input: JSON file (default `BigDataset.json`).
- Output: HF dataset with 3 columns.

**Why it matters**
- Stage 2 aligns the model with the intended prompt/response protocol.

---

### Section K: Stage 2 trainer and training execution
**Purpose**
- Fine-tune the same LoRA-adapted model in instruction format.

**Important code**
- `formatting_prompts_func(examples)` composes prompt+target with `###` labels + `<|endoftext|>`.
- `SFTTrainer(... formatting_func=formatting_prompts_func ...)`
- Stage 2 `TrainingArguments` (e.g., epochs=3, `save_total_limit=2`).
- `trainer.train()`.

**Inputs / outputs**
- Input: stage-2 dataset and reused model.
- Output: instruction-tuned adapter state and checkpoints in `cfg.stage2_output_dir`.

**Why it matters**
- Converts domain-adapted model into instruction-following model for downstream inference.

---

### Section L: Inference test
**Purpose**
- Quick functional validation after training.

**Important code**
- `FastLanguageModel.for_inference(model)`.
- Prompt pattern mirrors training format.
- `model.generate(... max_new_tokens=cfg.inference_max_new_tokens, use_cache=False, pad_token_id=...)`.
- Decodes only new tokens after prompt length.

**Inputs / outputs**
- Input: one configured instruction + input string.
- Output: printed model response text.

**Why it matters**
- Checks that model can respond in expected format.

---

### Section M: GGUF export and zip packaging
**Purpose**
- Produce deployment-ready artifact.

**Important code**
- `model.save_pretrained_gguf(output_dir, tokenizer, quantization_method='q4_k_m', token=...)`.
- Creates zip with `shutil.make_archive(...)`.

**Inputs / outputs**
- Input: trained model and tokenizer.
- Output: GGUF folder + zip archive path under `project_root`.

**Why it matters**
- Enables easier transfer/deployment to GGUF-compatible runtimes.

## 5. Libraries, tools, and dependencies
Major libraries and their roles:
- **torch / torchvision / torchaudio**: core tensor and CUDA backend.
- **unsloth** (`FastLanguageModel`): model loading, LoRA wrapping helper, inference mode, GGUF save helper.
- **trl** (`SFTTrainer`): supervised fine-tuning trainer abstraction.
- **transformers** (`TrainingArguments`, callbacks): training config and trainer internals.
- **peft** (indirectly via Unsloth): LoRA/PEFT integration.
- **bitsandbytes**: 4-bit/8-bit optimization support (`load_in_4bit`, `adamw_8bit`).
- **datasets**: HF Dataset objects for training inputs.
- **PyMuPDF (`fitz`)**: PDF text extraction.
- **huggingface_hub**: auth verification, file download check, optional snapshot download.
- **json / glob / re / shutil / pathlib / os**: filesystem, parsing, cleanup, artifact packaging.

## 6. Data and input format
### Expected inputs
1. **PDF directory**
   - Path: `project_root / pdf_subdir` (default `/home/n0etem01/LLM/books`).
   - Content: one or more `.pdf` files for stage-1 training.
2. **Instruction JSON file**
   - Path: `project_root / instruction_json_name` (default `/home/n0etem01/LLM/BigDataset.json`).
   - Format: list of objects each with:
     - `instruction`
     - `input`
     - `response`

### Loading and preprocessing
- PDFs:
  - Extract full text page-by-page.
  - Run custom whitespace transformation and fixed-word chunking.
  - Build one-column dataset: `{"text": ...}`.
- JSON:
  - Load entire file with `json.load`.
  - Strip leading/trailing `#` and spaces from each text field.
  - Build dataset columns: `instruction`, `input`, `response`.

### Prompt formatting
- Stage 2 turns each example into:
  - `### Instruction: ...`
  - `### Input: ...`
  - `### Response: ...<|endoftext|>`
- Inference prompt mirrors this exactly (without target answer).

### Tokenization / transformation behavior
- Tokenization is handled internally by `SFTTrainer` + tokenizer.
- `max_seq_length=2048` controls truncation/packing behavior boundary.
- `dataset_num_proc=2` enables parallel preprocessing operations where supported.

## 7. Model and fine-tuning setup
### Model
- Default model: `unsloth/llama-3-70b-bnb-4bit`.
- Loaded in 4-bit mode with Unsloth.
- Optional local model dir for offline or controlled runs.

### Fine-tuning strategy
- **PEFT LoRA** on selected projection modules in transformer blocks.
- Single adapter attachment reused across stage-1 and stage-2.
- Two-phase SFT sequence:
  1. Domain adaptation on unlabeled text chunks.
  2. Instruction tuning on formatted instruction-response data.

### LoRA / PEFT config highlights
- `r=64`, `lora_alpha=128`, `lora_dropout=0.0`, `bias='none'`.
- `target_modules`: q/k/v/o projections + gate/up/down projections.
- `use_gradient_checkpointing='unsloth'`.
- `use_rslora=False`, `loftq_config=None`.

### Training settings (selected)
- Optimizer: `adamw_8bit` for both stages.
- Scheduler: linear.
- Stage 1: epochs=1, batch=2, grad accumulation=4.
- Stage 2: epochs=3, batch=2, grad accumulation=4.
- Precision: BF16 when supported, otherwise FP16.

## 8. Training logic
### Batching and effective step size
- Both stages use `per_device_train_batch_size=2` and `gradient_accumulation_steps=4`.
- Effective microbatch accumulation before optimizer step is 8 samples per device-equivalent (subject to trainer semantics).

### Objective/loss
- Explicit custom loss is not defined in notebook.
- With `SFTTrainer`, the effective objective is standard causal LM next-token prediction over formatted text sequences.

### Checkpointing/saving
- Stage 1: `save_strategy='epoch'` to `domain_pretrain_outputs`.
- Stage 2: `save_strategy='epoch'`, `save_total_limit=2` to `outputs`.

### Logging
- Stage 1 logging every 10 steps.
- Stage 2 logging every 25 steps.
- `report_to='none'` disables external trackers.
- Progress bars suppressed in Stage 1 (`disable_tqdm=True`, removes `ProgressCallback`).

### Evaluation during training
- No validation dataset or periodic eval is configured.
- Therefore, no explicit train/val metrics split is shown.

## 9. Outputs and generated artifacts
Potential artifacts produced:
- **Stage 1 outputs**: checkpoints/logs in `project_root/domain_pretrain_outputs`.
- **Stage 2 outputs**: checkpoints/logs in `project_root/outputs`.
- **GGUF export** (if enabled):
  - folder: `project_root/gguf_model`
  - zip: `project_root/llama3_model_folder.zip`
- **Console outputs**:
  - training stats from `.train()` return values,
  - one inference response string.

## 10. Practical interpretation
If this notebook runs successfully, a user gets:
- A LoRA-adapted Llama model tuned to their PDF domain.
- Additional instruction-following tuning on their JSON dataset.
- A tested generation path using the same prompt format as training.
- A GGUF artifact suitable for downstream deployment environments.

Practically, this can serve as a prototype pipeline for turning enterprise documents + task examples into a deployable specialist assistant.

## 11. Limitations and things to watch out for
1. **Hard-coded paths and defaults**
   - `project_root` defaults to `/home/n0etem01/LLM`; this will fail unless changed on other systems.
2. **Token handling risk**
   - `hf_token_value` field is present in config; hardcoded secrets in notebooks are risky.
3. **Potentially fragile PDF cleaning**
   - The whitespace logic removes all single spaces before restoring placeholders; can distort text.
4. **No train/val split or evaluation metrics**
   - Makes model quality hard to assess rigorously.
5. **No exception handling around data schema**
   - Missing keys in JSON samples would raise errors.
6. **Resource intensity**
   - Even 4-bit 70B with LoRA is heavy; environment/GPU constraints may still block training.
7. **No deterministic full reproducibility controls**
   - Seeds exist, but complete deterministic setup (all libs + dataloader workers + CUDA determinism flags) is not shown.
8. **Prompt contract dependency**
   - Inference prompt should match training format; changing format may reduce output quality.
9. **Export assumptions**
   - GGUF export may require additional tooling/model access; notebook checks gated repo access but does not fully validate export environment compatibility beforehand.
10. **Single sample test only**
   - The notebook performs one qualitative inference check, not systematic benchmarking.

## 12. Beginner-friendly summary
This notebook teaches a Llama model in two steps:
1. read your PDF documents so it learns your domain language,
2. train on instruction/answer examples so it responds in a useful format.

It then asks the trained model one test question and saves the final model in GGUF format so it can be used in more deployment tools.

## 13. Technical summary
A 4-bit Unsloth Llama checkpoint is loaded and wrapped with LoRA adapters over attention/MLP projection modules. The same PEFT model is trained sequentially with TRL `SFTTrainer`: (i) domain text chunks from PDFs via `dataset_text_field='text'`, then (ii) formatted instruction-input-response tuples with `<|endoftext|>` termination. Training uses 8-bit AdamW, linear LR schedule, BF16/FP16 auto-selection, epoch checkpointing, and no eval set. Post-training inference is run on the same prompt schema, then optional GGUF export and ZIP packaging are produced.

## 14. Suggested improvements
### Reproducibility
- Move config to a separate YAML/TOML file and log a frozen run config to output.
- Add explicit version pinning for all critical packages (not just some).
- Add deterministic settings and record CUDA/GPU metadata in artifacts.

### Data quality and robustness
- Replace custom PDF spacing cleanup with safer normalization and unit tests.
- Validate JSON schema before training (required keys, non-empty strings).
- Add dataset statistics (token lengths, truncation rates, sample previews).

### Training quality
- Add train/validation split and periodic evaluation.
- Track metrics (loss curves, optionally WandB/TensorBoard).
- Consider early stopping or best-checkpoint selection.

### Modularity
- Refactor notebook cells into reusable Python modules/scripts.
- Separate stage-1 and stage-2 as independent CLI commands.

### Security and operations
- Remove hardcoded token field entirely; require env-var or secret manager.
- Add explicit checks for available VRAM and estimated memory before training.

### Documentation
- Document expected JSON schema with concrete example records.
- Explain intended deployment runtime for GGUF and compatibility caveats.

## Observed vs Inferred
### Observed directly in code
- Two-stage training exists (PDF stage then instruction stage).
- LoRA is attached once and reused.
- Model defaults to `unsloth/llama-3-70b-bnb-4bit` with `load_in_4bit=True`.
- Instruction format uses `### Instruction`, `### Input`, `### Response` and `<|endoftext|>`.
- Optional GGUF export and zip creation are implemented.

### Reasonable inferences from code
- Stage 1 is intended as domain adaptation and Stage 2 as instruction alignment.
- The target application likely includes machine/3D-printer status interpretation due to test prompt text.
- GGUF path is intended for downstream lightweight deployment.

### Unclear / not explicitly shown
- Exact quality of final model (no benchmark section).
- Exact schema guarantees of `BigDataset.json` beyond expected keys.
- Whether exported GGUF was verified in an external runtime.
