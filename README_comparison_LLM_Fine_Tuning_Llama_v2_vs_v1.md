# Comparison of `LLM_Fine_Tuning_Llama_v2.ipynb` and `LLM_Fine_Tuning_Llama.ipynb`

## 1. Executive comparison
Both notebooks implement the **same core idea**: two-stage fine-tuning of a Llama-family model with Unsloth + LoRA + TRL SFT, where Stage 1 uses PDF-derived text and Stage 2 uses instruction JSON data.

The biggest differences are in **engineering quality and robustness**, not in core training concept:
- `v2` introduces a centralized `FineTuneCfg` dataclass and structured path management.
- `v2` adds explicit environment setup, cache routing, offline mode support, token resolution logic, and preflight checks.
- `v2` avoids re-attaching LoRA in Stage 2 by adding an `attach_lora_once(...)` guard.
- `v2` removes Colab-only logic and replaces Drive/browser download steps with local GGUF export + local zipping.
- `v2` preserves most major hyperparameters and the stage-2 prompt template from `v1`.

So the methodology is mostly the same, but `v2` is significantly more organized and reproducible.

---

## 2. High-level purpose comparison
### Shared purpose
Both notebooks appear intended to:
1. Load a 4-bit Unsloth Llama model.
2. Perform domain adaptation from PDFs.
3. Perform instruction tuning from JSON records (`instruction`, `input`, `response`).
4. Test generation on a sample manufacturing/3D-printer prompt.
5. Export a GGUF artifact.

### Purpose shifts in `v2`
`v2` keeps the same model-training purpose but adds a strong operational goal:
- make the pipeline less Colab-dependent,
- more configurable,
- safer to rerun,
- easier to move between online/offline/local model workflows.

Inference: `v2` is a refactor-and-hardening update rather than a methodological replacement.

---

## 3. Methodology comparison

### Data preparation
- **v1**:
  - PDF extraction/chunking split across separate cells.
  - hard-coded PDF path (`/content/books`).
  - JSON loaded from relative file `BigDataset.json` in runtime working dir.
- **v2**:
  - Same extraction/chunking logic preserved.
  - paths parameterized through `cfg` (`project_root`, `pdf_subdir`, `instruction_json_name`).
  - explicit file existence checks before training.

### Prompt construction
- **v1** and **v2** both use Stage-2 template:
  - `### Instruction: ...`
  - `### Input: ...`
  - `### Response: ...<|endoftext|>`
- Inference prompt in both mirrors training format.

### Tokenizer/model setup
- **v1**:
  - direct model load from hardcoded model name.
- **v2**:
  - same base model default, plus optional local model directory and optional predownload snapshot.

### Fine-tuning method
- Same conceptual strategy in both:
  - Unsloth load + LoRA + TRL `SFTTrainer`.
  - Stage 1 on plain text chunks.
  - Stage 2 with formatting function.

### Trainer configuration
- Most hyperparameters are unchanged.
- `v2` exposes them as named config fields; easier to audit and modify.

### Evaluation/inference
- Both do a single sample generation.
- `v2` adds CPU fallback for inference device selection (`cuda` else `cpu`).

### Saving/exporting
- **v1**: Google Drive mount + Colab file download + shell copy.
- **v2**: local GGUF folder + local zip archive with `shutil.make_archive`.

### Methodology verdict
- **Conceptual pipeline:** mostly unchanged.
- **Implementation quality:** materially improved in `v2`.

---

## 4. Structure comparison

| Aspect | `LLM_Fine_Tuning_Llama.ipynb` (v1) | `LLM_Fine_Tuning_Llama_v2.ipynb` (v2) | Impact |
|---|---|---|---|
| Cell count | 34 cells (mixed markdown/code, includes empties) | 17 code cells | v2 is more compact and linear |
| Config style | scattered constants in multiple cells | single `FineTuneCfg` dataclass | higher maintainability in v2 |
| Environment setup | minimal pip commands; Colab assumptions | explicit package install + env vars + checks | v2 is more reproducible |
| Token handling | hardcoded token cell near end | token resolution strategy (config/env/disabled) | v2 clearer/safer workflow |
| Path handling | hard-coded `/content/...` and relative JSON | path properties under project root | v2 easier to port |
| Flow clarity | interleaves training and Colab download ops | staged and commented sections | v2 clearer execution story |
| Robustness guards | limited | file checks, local-dir checks, offline toggle | v2 more robust |

### Added/removed/reordered highlights
- `v2` adds:
  - global config cell,
  - environment sanity + directory creation,
  - HF auth test cell,
  - optional model predownload cell,
  - LoRA idempotency helper.
- `v2` removes/replaces:
  - Colab drive mounts,
  - browser download steps,
  - duplicate LoRA attachment cell before stage 2,
  - empty placeholder cells.

---

## 5. Detailed code differences by section

### A) Install/setup section
**v1**
- Uses shell magics:
  - `!pip install PyMuPDF`
  - `!pip install unsloth trl peft accelerate bitsandbytes`
- Has simple GPU print cell.

**v2**
- Uses `subprocess.check_call` with `sys.executable -m pip`.
- Pins/reinstalls `torchvision==0.25.0` and `torchaudio==2.10.0` from CUDA 12.8 index.
- Installs `datasets` and `huggingface_hub` explicitly.
- Adds a separate CUDA sanity cell.

**Likely effect**
- v2 installation is more deterministic across environments, though more aggressive due to force-reinstall behavior.

---

### B) Configuration section
**v1**
- Values like model name, max length, output dirs, and dataset paths live inline across cells.

**v2**
- `FineTuneCfg` holds:
  - project/data/output paths,
  - model flags,
  - LoRA settings,
  - stage1/stage2 training args,
  - inference test values,
  - GGUF export settings.

**Likely effect**
- Easier experiment changes and fewer hidden hardcoded values.

---

### C) Environment and token management
**v1**
- No centralized env-variable setup for HF caches/offline behavior.
- Token appears in late cell with `hf_token = "**************"` and Colab comment.

**v2**
- Creates required directories proactively.
- Sets `HF_HOME`, `HF_HUB_CACHE`, `TOKENIZERS_PARALLELISM`.
- Adds optional offline mode via `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`.
- Token resolution priority:
  1. `cfg.hf_token_value`
  2. environment variable (`HF_TOKEN` by default)
  3. no token.

**Likely effect**
- Better cache control and portability, clearer auth behavior.

---

### D) Model access checks and model source
**v1**
- No explicit preflight token/access check.
- Directly calls `FastLanguageModel.from_pretrained(model_name=...)`.

**v2**
- Adds `whoami(token=hf_token)` and `hf_hub_download(...)` access test to gated Meta Llama repo.
- Adds optional `snapshot_download(...)` model predownload.
- Supports loading from either HF repo ID or local model directory.

**Likely effect**
- Faster failure detection when auth/access/network is wrong.
- Supports offline/local workflows more cleanly.

---

### E) LoRA attachment behavior
**v1**
- Attaches LoRA in stage-1 setup cell.
- Re-attaches LoRA again before stage-2 (`FastLanguageModel.get_peft_model(...)` appears twice).

**v2**
- Defines `attach_lora_once(model, cfg)`.
- Checks `model.peft_config` before wrapping.
- Reuses same adapted model across both stages.

**Likely effect**
- Reduces risk of accidental double wrapping and state confusion.

---

### F) Stage-1 data handling (PDF)
**Same in both**
- `extract_text_from_pdf(...)` with `fitz`.
- `split_paragraphs(...)` with the same whitespace placeholder approach.
- chunk size default 50 words.
- create `Dataset.from_dict({"text": all_pdf_chunks})`.

**Differences**
- v1 path fixed as `/content/books`.
- v2 path from `cfg.pdf_dir` and checks PDF folder state earlier.

**Likely effect**
- Data logic same; operational portability better in v2.

---

### G) Stage-1 trainer setup/execution
**Same core settings**
- batch size 2, grad accumulation 4, warmup 10, epochs 1, LR 2e-4, `adamw_8bit`, linear scheduler, seed 3407.
- BF16/FP16 switch based on GPU capability.
- output dir: `domain_pretrain_outputs`.

**Difference**
- v2 adds `average_tokens_across_devices` setting from config (default false).
- v2 keeps all values in config.

**Likely effect**
- Minimal methodological change; mostly configurability.

---

### H) Stage-2 data loading and cleaning
**Same core behavior**
- load JSON with records containing `instruction`, `input`, `response`.
- clean text with `text.strip('# ')`.
- build dataset with three columns.

**Differences**
- v1 uses `json.load(open("BigDataset.json"))` inline.
- v2 uses configured path with explicit encoding and prior existence checks.

**Likely effect**
- v2 improves path clarity and predictable file handling.

---

### I) Stage-2 trainer setup/execution
**Shared core**
- same prompt format logic.
- same trainer class and near-identical hyperparameters.

**Differences**
- v1 includes `dataset_text_field="text"` while also passing `formatting_func`; this may be redundant/inconsistent depending on TRL version.
- v2 removes `dataset_text_field` and relies on `formatting_func` only.
- v2 adds `average_tokens_across_devices` config field.

**Likely effect**
- v2 is cleaner and likely less error-prone across TRL versions.

---

### J) Inference cell
**Shared behavior**
- `FastLanguageModel.for_inference(model)`.
- prompt built with same structure.
- `model.generate(..., max_new_tokens=256, use_cache=False, pad_token_id=eos)`.

**Differences**
- v1 always sends tensors to CUDA (`to("cuda")`).
- v2 chooses device dynamically (`"cuda" if available else "cpu"`).
- v2 externalizes prompt values into config fields.

**Likely effect**
- v2 safer on systems without CUDA (though training still likely GPU-bound).

---

### K) Export and artifact handling
**v1**
- exports GGUF to Google Drive path.
- uses Colab file operations (`files.download`, `!cp`, drive mount/unmount).

**v2**
- exports GGUF locally to configured directory.
- zips folder locally with `shutil.make_archive`.
- no Colab-specific dependencies.

**Likely effect**
- v2 works outside Colab and is cleaner for local/cluster workflows.

---

### L) Notebook integrity issue in v1
Observed issue:
- v1 file contains malformed JSON around token line:
  - `"hf_token = \"**************"\n"` (missing escaped quote before line break).

Likely effect:
- Some strict notebook parsers/tools may fail to parse v1 as valid JSON without manual repair.

---

## 6. Hyperparameter and configuration comparison

| Parameter / Setting | v1 | v2 | Difference type |
|---|---|---|---|
| Base model | `unsloth/llama-3-70b-bnb-4bit` | same default | same |
| Alt model comment | commented 8B option present | not present | cosmetic/cleanup |
| `max_seq_length` | 2048 | 2048 | same |
| `dtype` | `None` | `None` | same |
| `load_in_4bit` | True (inline) | True (config field) | organization |
| LoRA rank `r` | 64 | 64 | same |
| LoRA alpha | 128 | 128 | same |
| LoRA dropout | 0 | 0.0 | same |
| LoRA target modules | q/k/v/o + gate/up/down | same | same |
| `use_gradient_checkpointing` | `"unsloth"` | same | same |
| Stage1 batch | 2 | 2 | same |
| Stage1 grad accum | 4 | 4 | same |
| Stage1 epochs | 1 | 1 | same |
| Stage1 LR | 2e-4 | 2e-4 | same |
| Stage1 logging | 10 | 10 | same |
| Stage1 output dir | `domain_pretrain_outputs` | same (config property path) | organization |
| Stage2 batch | 2 | 2 | same |
| Stage2 grad accum | 4 | 4 | same |
| Stage2 epochs | 3 | 3 | same |
| Stage2 LR | 2e-4 | 2e-4 | same |
| Stage2 logging | 25 | 25 | same |
| Stage2 save strategy | epoch | epoch | same |
| Stage2 save total limit | 2 | 2 | same |
| Dataset processing procs | 2 (inline) | 2 (config) | organization |
| `average_tokens_across_devices` | not set | set (false default) | new explicit arg |
| `dataloader_pin_memory` | False | False | same |
| report_to | none | none | same |
| Test prompt text | hardcoded in cell | moved to config | organization |
| GGUF quant method | `q4_k_m` | `q4_k_m` | same |
| Output zip name | `llama3_model_folder.zip` | config-driven same base | organization |
| Local/offline flags | absent | present (`offline_mode`, local model flags) | new capability |
| HF cache routing | absent | present (`HF_HOME`, `HF_HUB_CACHE`) | new capability |

If a setting is “same,” the key improvement in v2 is usually centralization and explicitness rather than numerical change.

---

## 7. Data and prompt formatting comparison

### Data source comparison
- **Stage 1 PDFs**
  - v1: `/content/books` hardcoded.
  - v2: `cfg.pdf_dir` derived from configurable root.
- **Stage 2 JSON**
  - v1: `BigDataset.json` in current working dir.
  - v2: `cfg.instruction_json_path` with explicit check.

### Data schema comparison
- Same expected schema for stage 2:
  - `instruction`, `input`, `response` per record.

### Preprocessing comparison
- `clean_text(text.strip('# '))` is unchanged.
- PDF chunking logic is unchanged (same strengths and same fragility).

### Formatting function comparison
- Both versions format as:
  - `### Instruction: ...`
  - `### Input: ...`
  - `### Response: ...<|endoftext|>`

### Tokenization comparison
- Training tokenization remains inside `SFTTrainer` flow.
- Inference tokenization remains explicit via tokenizer call.
- v2 inference adds dynamic device choice.

### Impact on training behavior
- Because prompt template and major hyperparameters are unchanged, core learned behavior should be broadly comparable.
- Most differences affect **reliability and reproducibility**, not the fundamental supervised objective.

---

## 8. Model and training pipeline comparison

### Model loading logic
- v1: direct HF model load, no local/offline branch.
- v2: supports HF or local dir, plus optional predownload and auth precheck.

### Adapter setup
- v1: LoRA applied twice (before each stage).
- v2: LoRA applied once and guarded against re-application.

### Training arguments and trainer creation
- Core values same.
- v2 wraps values in config and removes potential stage-2 `dataset_text_field` ambiguity.

### Checkpoint behavior
- Stage 1: epoch saves in both.
- Stage 2: epoch saves with total-limit=2 in both.

### Save/export behavior
- v1: Colab/Drive-centric export flow.
- v2: local filesystem-centric export flow.

### Pipeline change verdict
- **Conceptually same pipeline** (2-stage SFT + inference + GGUF export).
- **Material implementation improvements** in environment handling, portability, and notebook hygiene.

---

## 9. Output and artifact comparison

| Artifact type | v1 | v2 |
|---|---|---|
| Stage1 training stats | printed | printed |
| Stage2 training stats | produced (not always printed explicitly) | printed |
| Stage1 checkpoints | `domain_pretrain_outputs` | `cfg.stage1_output_dir` (same default name) |
| Stage2 checkpoints | `outputs` | `cfg.stage2_output_dir` (same default name) |
| Inference text output | yes | yes |
| GGUF folder | yes | yes |
| Zip artifact | yes | yes |
| Google Drive copies/download | yes | no |
| Local zip via Python stdlib | no | yes |

No metric plots, benchmark tables, or validation scores are explicitly produced in either notebook.

---

## 10. Practical impact of the differences

### Ease of use
- v2 is easier for non-Colab environments due to configurable paths and local export.

### Reproducibility
- v2 improves reproducibility via centralized config, explicit env vars, and startup checks.

### GPU memory usage/runtime
- Mostly unchanged: same 70B 4-bit base + LoRA + similar hyperparameters.
- Minor differences may come from runtime/env configuration, not core training settings.

### Training stability
- v2 likely more stable operationally due to:
  - preflight auth checks,
  - file existence checks,
  - no duplicate LoRA wrapping,
  - less environment ambiguity.

### Output quality
- No direct code evidence that model quality should differ significantly if data and seeds are same.
- However, avoiding setup errors (v2) can indirectly improve practical outcomes.

### Maintainability
- Strong improvement in v2 due to config dataclass + cleaner stage boundaries.

---

## 11. Which version is stronger and why
Based only on observed code, **`v2` is stronger overall**.

### Why v2 appears better
1. Better configuration architecture (`FineTuneCfg`).
2. Better environment hygiene (cache routing, optional offline mode).
3. Better failure transparency (sanity checks and auth test).
4. Better pipeline correctness (LoRA attached once).
5. Better portability (not tied to Colab Drive).

### Tradeoffs / mixed points
- v2 is more verbose and configuration-heavy, which can feel complex for quick Colab demos.
- Both versions still share some fragile areas (e.g., PDF text reconstruction heuristic, no validation/eval dataset).

Net judgment: v2 is more maintainable and production-friendly while preserving v1’s core training recipe.

---

## 12. Change log summary
- Added centralized config dataclass and path properties in v2.
- Added environment setup cell with directory creation and cache env vars.
- Added token resolution strategy and optional offline mode.
- Added HF auth and gated access verification step.
- Added optional model predownload and local-model loading branch.
- Refactored LoRA setup into idempotent helper; removed duplicate stage-2 LoRA wrap.
- Preserved core stage-1/stage-2 training hyperparameters and prompt format.
- Replaced Colab Drive export/download flow with local GGUF+zip export.
- Added CPU fallback in inference device selection.
- Reduced notebook clutter (fewer cells, fewer empty/Colab utility cells).

---

## 13. Observed vs inferred

### Directly observed in code
- Same base model default (`unsloth/llama-3-70b-bnb-4bit`) and same 2-stage training logic.
- Same stage-2 prompt template and nearly identical key hyperparameters.
- v2 has explicit config class, environment setup, token logic, and local export workflow.
- v1 includes Colab-specific cells (`google.colab.drive`, `files.download`, `!cp`).
- v1 contains a malformed JSON line in notebook source around `hf_token` assignment.

### Inferred intent behind changes
- Refactor from demo-style Colab notebook toward reusable local workflow.
- Improve reliability and reduce hidden state issues when rerunning cells.
- Preserve training behavior while improving maintainability.

### Ambiguous points
- Whether v2 intentionally changes final model quality is not clear from code alone.
- Some trainer argument compatibility differences depend on library versions (e.g., handling of `dataset_text_field` with `formatting_func`).
