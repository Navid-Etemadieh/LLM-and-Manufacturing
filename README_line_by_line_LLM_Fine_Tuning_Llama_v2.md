# README Line-by-Line for `LLM_Fine_Tuning_Llama_v2.ipynb`

## 1. Purpose of this notebook
This notebook builds a **two-stage Llama fine-tuning pipeline** using Unsloth + TRL + LoRA. Its goal is to transform a base Llama model into a domain-adapted and instruction-following model.

The workflow is staged intentionally:
1. **Stage 1 (Domain adaptation):** train on text extracted from PDFs.
2. **Stage 2 (Instruction tuning):** train on a structured JSON dataset with `instruction`, `input`, and `response`.

It also includes:
- environment setup and package installation,
- Hugging Face token/access checks,
- optional local snapshot download for offline use,
- final inference test,
- optional GGUF export and zip packaging.

The notebook is designed as an end-to-end workflow rather than a minimal training script.

---

## 2. Big-picture methodology map

1. **Environment setup**
   - Install required packages in current Jupyter kernel env.
   - Validate CUDA + PyTorch state.

2. **Configuration setup**
   - Define one dataclass (`FineTuneCfg`) containing all paths, model names, training hyperparameters, and export flags.

3. **Infrastructure preparation**
   - Create required folders.
   - Configure Hugging Face cache location.
   - Resolve Hugging Face token and optional offline mode.
   - Validate required inputs exist.

4. **Model access + loading**
   - Verify HF token and gated repo access.
   - Optionally predownload model snapshot.
   - Load model/tokenizer via Unsloth in 4-bit mode.

5. **PEFT adaptation**
   - Attach LoRA adapters once.

6. **Stage 1 data prep (PDFs)**
   - Extract PDF text with PyMuPDF.
   - Apply whitespace cleanup + fixed-size chunking.
   - Create HF dataset with `text` field.

7. **Stage 1 training**
   - Configure `TrainingArguments` and `SFTTrainer`.
   - Train LoRA adapters on domain text.

8. **Stage 2 data prep (instructions)**
   - Load JSON records.
   - Clean strings and build dataset columns.
   - Define prompt-format function.

9. **Stage 2 training**
   - Reuse same LoRA model.
   - Train using formatted instruction-input-response prompts.

10. **Inference + export**
    - Test generated response on one sample prompt.
    - Optionally export GGUF and create zip archive.

---

## 3. Cell-by-cell and line-by-line explanation

> Notes before we begin:
> - The notebook contains **17 code cells** (Cell 0–Cell 16).
> - Cell 16 is empty.
> - Explanations below follow exact execution order.

---

### Cell 0: Package installation in active kernel

**Purpose of this cell**
- Install required Python packages into the same environment used by Jupyter kernel.

**Inputs**
- Current Python executable (`sys.executable`), internet access.

**Outputs / state changes**
- Installs or updates packages in current environment.

**Why this cell matters**
- All downstream cells depend on these libraries.

#### Code
```python
import sys
import subprocess
```

#### Explanation
- `import sys`
  - Literal action: imports Python system module.
  - Purpose in notebook: gets the exact interpreter path used by the notebook.
  - Methodology connection: avoids installing packages into the wrong environment.
  - Notes/cautions: important when multiple conda/venv envs exist.

- `import subprocess`
  - Literal action: imports process-execution module.
  - Purpose in notebook: runs pip commands programmatically.
  - Methodology connection: reproducible setup in notebook.
  - Notes/cautions: failures will stop cell because `check_call` raises.

#### Code
```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--force-reinstall",
    "--no-deps",
    "torchvision==0.25.0",
    "torchaudio==2.10.0",
    "--index-url", "https://download.pytorch.org/whl/cu128",
])
```

#### Explanation
- `subprocess.check_call([...])`
  - Literal action: executes pip install command and raises if non-zero exit.
  - Purpose in notebook: installs exact CUDA-matched torchvision/torchaudio versions.
  - Methodology connection: ensures runtime compatibility for deep learning stack.
  - Notes/cautions: uses `--no-deps`; assumes torch and dependencies already correct.

- `sys.executable, "-m", "pip", "install"`
  - Literal action: runs pip from current Python interpreter.
  - Purpose in notebook: environment consistency.
  - Methodology connection: infrastructure reliability.

- `"--force-reinstall"`
  - Literal action: reinstalls package even if present.
  - Purpose: enforce expected package versions.
  - Notes: can override previously working installations.

- `"--no-deps"`
  - Literal action: skip dependency resolution.
  - Purpose: prevent pip from modifying torch unexpectedly.
  - Notes: can break if required deps missing.

- `"torchvision==0.25.0", "torchaudio==2.10.0"`
  - Literal action: version-pinned installs.
  - Purpose: pair with fixed torch version.

- `"--index-url", "https://download.pytorch.org/whl/cu128"`
  - Literal action: uses PyTorch wheel index for CUDA 12.8 builds.
  - Purpose: ensure GPU-enabled wheels.
  - Notes: requires network and matching driver/runtime.

#### Code
```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-U",
    "PyMuPDF",
    "unsloth",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface_hub",
])
```

#### Explanation
- Installs/updates core packages for:
  - PDF parsing (`PyMuPDF`),
  - model loading + PEFT helpers (`unsloth`),
  - supervised fine-tuning (`trl`),
  - LoRA (`peft`),
  - distributed acceleration (`accelerate`),
  - quant/optimizer stack (`bitsandbytes`),
  - dataset management (`datasets`),
  - HF auth/download utilities (`huggingface_hub`).

#### Code
```python
print("✅ Installation finished.")
print("Python executable:", sys.executable)
```

#### Explanation
- Prints completion indicator and interpreter path.
- Helps user verify where packages were installed.

---

### Cell 1: Torch/CUDA sanity print

**Purpose**
- Quick runtime verification of torch and CUDA status.

**Inputs**
- Installed torch package + visible GPU runtime.

**Outputs/state changes**
- Prints versions and CUDA availability.

#### Code
```python
import torch
print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("CUDA available =", torch.cuda.is_available())
```

#### Explanation
- `import torch`: loads PyTorch backend.
- `torch.__version__`: confirms torch version.
- `torch.version.cuda`: confirms CUDA toolkit version torch expects.
- `torch.cuda.is_available()`: checks if a CUDA-capable GPU is usable.

Methodology link: confirms infrastructure before expensive model load/training.

---

### Cell 2: Global configuration (`FineTuneCfg`)

**Purpose**
- Define all tunable parameters in one dataclass.

**Inputs**
- None external; static defaults.

**Outputs/state changes**
- Creates class `FineTuneCfg`; instantiates `cfg`.

**Why important**
- Central configuration source for entire notebook.

#### Code
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any
```

#### Explanation
- Imports utilities for structured config and path typing.
- `field` and `Optional` are imported but not used directly (harmless).

#### Code
```python
@dataclass
class FineTuneCfg:
    project_root: str = "/home/n0etem01/LLM"
    pdf_subdir: str = "books"
    instruction_json_name: str = "BigDataset.json"
```

#### Explanation
- `@dataclass`: auto-generates init/repr methods.
- `project_root`: root for data, caches, outputs.
- `pdf_subdir`: stage-1 PDF location under root.
- `instruction_json_name`: stage-2 JSON file name.

#### Code
```python
    model_name: str = "unsloth/llama-3-70b-bnb-4bit"
    use_local_model_dir: bool = False
    local_model_subdir: str = "local_models/unsloth_llama_3_70b_bnb_4bit"
    predownload_model_snapshot: bool = False
    offline_mode: bool = False
```

#### Explanation
- Controls where model comes from and whether offline/local workflow is used.
- Default behavior: load from HF repo in online mode.

#### Code
```python
    hf_token_env_var: str = "HF_TOKEN"
    use_hf_token: bool = True
    hf_token_value: str = "***************"
```

#### Explanation
- Token config:
  - env var name,
  - enable/disable token usage,
  - optional hardcoded token string.
- Caution: hardcoded token placeholder field suggests secret handling risk if real token inserted.

#### Code
```python
    hf_cache_subdir: str = "hf_cache"
    max_seq_length: int = 2048
    dtype: Any = None
    load_in_4bit: bool = True
```

#### Explanation
- HF cache location under project root.
- sequence length for tokenizer/model context during training.
- `dtype=None`: delegate precision choice to library/runtime defaults.
- `load_in_4bit=True`: memory-efficient quantized loading.

#### Code
```python
    pdf_chunk_size: int = 50
```

#### Explanation
- Number of words per chunk in stage-1 PDF corpus.
- Directly affects sample count and sequence structure.

#### Code
```python
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Any = None
```

#### Explanation
- Core PEFT/LoRA configuration.
- `r` and `alpha` control adapter capacity/scaling.
- `dropout=0.0` means no LoRA dropout regularization.
- checkpointing mode saves memory at speed cost.
- `use_rslora`/`loftq_config` are optional advanced features disabled here.

#### Code
```python
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
```

#### Explanation
- Specifies transformer submodules to receive LoRA adapters.
- Includes attention projections and feedforward projections.
- Methodology: parameter-efficient adaptation at critical linear layers.

#### Code
```python
    stage1_per_device_train_batch_size: int = 2
    stage1_gradient_accumulation_steps: int = 4
    stage1_warmup_steps: int = 10
    stage1_num_train_epochs: int = 1
    stage1_learning_rate: float = 2e-4
    ...
```

#### Explanation
- Stage-1 hyperparameters for domain adaptation.
- Effective batch scaling uses accumulation.
- Epoch count is short (1), likely intended as light domain adaptation.

#### Code
```python
    stage2_per_device_train_batch_size: int = 2
    stage2_gradient_accumulation_steps: int = 4
    stage2_warmup_steps: int = 10
    stage2_num_train_epochs: int = 3
    stage2_learning_rate: float = 2e-4
    ...
```

#### Explanation
- Stage-2 hyperparameters for instruction tuning.
- More epochs than stage-1, indicating stronger instruction alignment phase.

#### Code
```python
    dataset_num_proc: int = 2
    test_instruction: str = "What is my 3D printer doing? Be specific"
    test_input_data: str = "[170.0, 60.5, 1.4, -1.4, 0.0, 1.1, 1.1, 0.5, 0.9, 0]"
    inference_max_new_tokens: int = 256
```

#### Explanation
- Dataset processing parallelism.
- Fixed test prompt+input for post-training generation.
- Max generation length control.

#### Code
```python
    save_gguf: bool = True
    gguf_output_subdir: str = "gguf_model"
    gguf_quantization_method: str = "q4_k_m"
    gguf_zip_base_name: str = "llama3_model_folder"
```

#### Explanation
- Enables GGUF export by default and sets quantization style.
- Adds zip packaging name.

#### Code
```python
    @property
    def root(self) -> Path:
        return Path(self.project_root)
    ...
```

#### Explanation
- Property helpers derive full paths (`pdf_dir`, `instruction_json_path`, outputs, gguf zip path).
- Keeps path construction consistent and centralized.

#### Code
```python
cfg = FineTuneCfg()
print(cfg)
```

#### Explanation
- Instantiates configuration with defaults.
- Prints full config for transparency/debugging.

---

### Cell 3: Environment setup and sanity checks

**Purpose**
- Prepare filesystem and environment variables, resolve token, validate inputs.

**Inputs**
- `cfg` from Cell 2, local filesystem, environment vars.

**Outputs/state changes**
- Creates directories; sets env vars; defines `hf_token`.

#### Code
```python
import os
import json
import glob
import re
import shutil
import torch
```

#### Explanation
- Imports utility modules used in later cells.
- These imports are shared infra dependencies.

#### Code
```python
cfg.root.mkdir(parents=True, exist_ok=True)
cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
cfg.hf_cache_dir.mkdir(parents=True, exist_ok=True)
cfg.stage1_output_dir.mkdir(parents=True, exist_ok=True)
cfg.stage2_output_dir.mkdir(parents=True, exist_ok=True)
```

#### Explanation
- Ensures required directory tree exists.
- `exist_ok=True` avoids errors if rerun.
- Infrastructure step supporting data IO and checkpoint saves.

#### Code
```python
os.environ["HF_HOME"] = str(cfg.hf_cache_dir)
os.environ["HF_HUB_CACHE"] = str(cfg.hf_cache_dir / "hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

#### Explanation
- Redirects HF caches into project-local folder.
- Disables tokenizer parallelism warnings/race behavior.

#### Code
```python
if cfg.offline_mode:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
```

#### Explanation
- Toggles strict offline behavior.
- In online mode, explicitly unsets stale offline flags from previous sessions.

#### Code
```python
if cfg.use_hf_token:
    if hasattr(cfg, "hf_token_value") and cfg.hf_token_value.strip():
        hf_token = cfg.hf_token_value.strip()
        token_source = "config"
    else:
        hf_token = os.environ.get(cfg.hf_token_env_var, "").strip()
        token_source = f"environment variable '{cfg.hf_token_env_var}'" if hf_token else "not found"
else:
    hf_token = ""
    token_source = "disabled"
```

#### Explanation
- Resolves token by priority:
  1. hardcoded config token,
  2. env var token,
  3. empty token.
- Creates `token_source` string for debug print.

#### Code
```python
print(f"Project root: {cfg.root}")
...
print(f"Offline mode: {cfg.offline_mode}")
```

#### Explanation
- Emits resolved environment state.
- Helps detect misconfigured paths/tokens early.

#### Code
```python
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"BF16 supported: {torch.cuda.is_available() and torch.cuda.is_bf16_supported()}")
```

#### Explanation
- Captures device name and BF16 capability.
- Later cells use BF16/FP16 switch in `TrainingArguments`.

#### Code
```python
if not cfg.pdf_dir.exists():
    raise FileNotFoundError(...)

if not cfg.instruction_json_path.exists():
    raise FileNotFoundError(...)
```

#### Explanation
- Hard fails if stage-2 JSON missing.
- Note: `cfg.pdf_dir` was just created, so first check is mostly defensive; second check is critical.

#### Code
```python
pdf_count = len(list(cfg.pdf_dir.glob("*.pdf")))
print(f"Found {pdf_count} PDF file(s) in {cfg.pdf_dir}")
if pdf_count == 0:
    print("⚠️ No PDFs found yet....")
```

#### Explanation
- Warns (not fail) when stage-1 source PDFs are missing.
- So stage-1 may run on empty dataset unless user provides PDFs.

---

### Cell 4: Hugging Face auth + gated repo access test

**Purpose**
- Validate token and access rights before training/export.

**Inputs**
- `hf_token` from Cell 3; internet access.

**Outputs**
- prints account name and local path of downloaded test file.

#### Code
```python
from huggingface_hub import whoami, hf_hub_download
print("Token present:", bool(hf_token))
info = whoami(token=hf_token)
print("Logged in as:", info["name"])
```

#### Explanation
- `whoami` verifies token validity.
- If invalid/empty token for private gated access, likely fails here.

#### Code
```python
test_file = hf_hub_download(
    repo_id="meta-llama/Meta-Llama-3-70B",
    filename="model.safetensors.index.json",
    token=hf_token,
)
print("✅ Access test passed:", test_file)
```

#### Explanation
- Downloads one known file from gated Meta Llama repo.
- Confirms both network and authorization.
- Methodology link: pre-flight check before later operations depending on model access.

---

### Cell 5: Optional model pre-download

**Purpose**
- Support local snapshot workflow.

**Inputs**
- `cfg.predownload_model_snapshot`, `cfg.model_name`, `hf_token`.

**Outputs/state changes**
- Downloads model files under local directory if enabled.

#### Code
```python
from huggingface_hub import snapshot_download

if cfg.predownload_model_snapshot:
    cfg.local_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model snapshot to: {cfg.local_model_dir}")

    snapshot_download(
        repo_id=cfg.model_name,
        local_dir=str(cfg.local_model_dir),
        token=hf_token if hf_token else None,
    )

    print("✅ Model snapshot download finished.")
else:
    print("Skipped model pre-download.")
```

#### Explanation
- Conditional branch controls one-time full snapshot pull.
- `token=... if hf_token else None` gracefully handles public models.
- If flag is false (default), cell is no-op except print.

---

### Cell 6: Load model and tokenizer

**Purpose**
- Instantiate base model/tokenizer via Unsloth.

**Inputs**
- `cfg` flags controlling model source.

**Outputs**
- `model`, `tokenizer` objects in memory.

#### Code
```python
from unsloth import FastLanguageModel

if cfg.use_local_model_dir:
    if not cfg.local_model_dir.exists():
        raise FileNotFoundError(...)
    model_source = str(cfg.local_model_dir)
else:
    model_source = cfg.model_name

print(f"Loading model from: {model_source}")
```

#### Explanation
- Chooses local directory or HF model ID as source.
- Includes fail-fast check if local mode enabled but path absent.

#### Code
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_source,
    max_seq_length=cfg.max_seq_length,
    dtype=cfg.dtype,
    load_in_4bit=cfg.load_in_4bit,
    token=hf_token if hf_token else None,
)

print("✅ Model and tokenizer loaded.")
```

#### Explanation
- Unsloth convenience loader returns both model and tokenizer.
- Applies sequence length and 4-bit config at load time.
- Token is provided when available.

Methodology connection: this is the base model foundation for both training stages.

---

### Cell 7: Attach LoRA adapters once

**Purpose**
- Turn base model into PEFT model with trainable adapters.

**Inputs**
- `model` from Cell 6; LoRA config from `cfg`.

**Outputs/state changes**
- `model` replaced by LoRA-augmented model.

#### Code
```python
def attach_lora_once(model, cfg: FineTuneCfg):
    if hasattr(model, "peft_config") and model.peft_config:
        print("LoRA adapters already attached. Reusing the same adapted model.")
        return model
```

#### Explanation
- Defines helper to avoid double-wrapping.
- Check for existing `peft_config` supports idempotent reruns.

#### Code
```python
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=list(cfg.target_modules),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
        use_rslora=cfg.use_rslora,
        loftq_config=cfg.loftq_config,
    )
    print("✅ LoRA adapters attached.")
    return model

model = attach_lora_once(model, cfg)
```

#### Explanation
- Calls Unsloth PEFT wrapper with config parameters.
- Applies LoRA to target projection modules.
- Returns adapted model and binds back to `model`.

Methodology connection: this is the PEFT mechanism that makes 70B fine-tuning feasible.

---

### Cell 8: Stage 1 PDF loading and chunk processing

**Purpose**
- Build stage-1 dataset from PDF text.

**Inputs**
- PDFs in `cfg.pdf_dir`.

**Outputs**
- `domain_dataset` with one column: `text`.

#### Code
```python
import fitz
from datasets import Dataset
```

#### Explanation
- `fitz` (PyMuPDF) for PDF extraction.
- HF `Dataset` for trainer-compatible data object.

#### Code
```python
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

#### Explanation
- Opens PDF and concatenates text from each page.
- No explicit close call (PyMuPDF usually cleans up on object release, but explicit close is safer style).

#### Code
```python
def split_paragraphs(pages, chunk_size=50):
    full_text = " ".join(pages)
    full_text = re.sub(r'\s{2,}', '___BOUNDARY___', full_text)
    full_text = full_text.replace(' ', '')
    full_text = full_text.replace('___BOUNDARY___', ' ')
    clean_text = re.sub(r'\s+', ' ', full_text).strip()
    words = clean_text.split()
    paragraphs = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        paragraphs.append(chunk)
    return paragraphs
```

#### Explanation
- Important nuance:
  - Parameter name `pages` suggests list input.
  - Actual call passes a string (`raw_text`), so `" ".join(pages)` initially inserts spaces between characters.
- Cleanup steps then attempt to reconstruct word boundaries and normalize whitespace.
- Final result is chunk list of `chunk_size` words.
- Caution: unconventional text reconstruction can be lossy/fragile.

#### Code
```python
pdf_directory = str(cfg.pdf_dir)
pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
all_pdf_chunks = []

for pdf_path in pdf_files:
    print(f"Processing: {os.path.basename(pdf_path)}...")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_paragraphs(raw_text, chunk_size=cfg.pdf_chunk_size)
    all_pdf_chunks.extend(chunks)
```

#### Explanation
- Locates all PDFs.
- For each file: extract text → split/chunk → append to global list.

#### Code
```python
print("-" * 30)
print(f"Successfully processed {len(pdf_files)} PDF(s).")
print(f"Total chunks combined: {len(all_pdf_chunks)}")

if all_pdf_chunks:
    print(f"First chunk overall: {all_pdf_chunks[0]}")
else:
    print("No chunks were generated...")

domain_dataset = Dataset.from_dict({"text": all_pdf_chunks})

print(domain_dataset)
if len(domain_dataset) > 0:
    print(domain_dataset[-1])
```

#### Explanation
- Prints aggregate processing summary.
- Creates training dataset with single `text` column.
- Prints dataset object + sample for quick validation.

---

### Cell 9: Stage 1 trainer setup

**Purpose**
- Configure and instantiate training for PDF domain adaptation.

**Inputs**
- `model`, `tokenizer`, `domain_dataset`, stage-1 config values.

**Outputs**
- `domain_training_args`, `domain_trainer`.

#### Code
```python
from trl import SFTTrainer
from transformers import TrainingArguments
```

#### Explanation
- Imports training classes.

#### Code
```python
domain_training_args = TrainingArguments(
    per_device_train_batch_size=cfg.stage1_per_device_train_batch_size,
    gradient_accumulation_steps=cfg.stage1_gradient_accumulation_steps,
    warmup_steps=cfg.stage1_warmup_steps,
    num_train_epochs=cfg.stage1_num_train_epochs,
    learning_rate=cfg.stage1_learning_rate,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=cfg.stage1_logging_steps,
    optim=cfg.stage1_optim,
    weight_decay=cfg.stage1_weight_decay,
    lr_scheduler_type=cfg.stage1_lr_scheduler_type,
    seed=cfg.stage1_seed,
    output_dir=str(cfg.stage1_output_dir),
    save_strategy=cfg.stage1_save_strategy,
    report_to=cfg.stage1_report_to,
    average_tokens_across_devices=cfg.stage1_average_tokens_across_devices,
)
```

#### Explanation
- Defines stage-1 optimization behavior.
- Precision is dynamically selected from BF16 support.
- Uses 8-bit AdamW and epoch-based save strategy.

#### Code
```python
domain_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=domain_dataset,
    dataset_text_field="text",
    max_seq_length=cfg.max_seq_length,
    dataset_num_proc=cfg.dataset_num_proc,
    args=domain_training_args,
)

print("✅ Stage 1 trainer ready.")
```

#### Explanation
- Builds SFT trainer consuming raw text field `text`.
- This object encapsulates tokenization + training loop configuration.

---

### Cell 10: Execute Stage 1 training

**Purpose**
- Run first fine-tuning stage.

**Inputs**
- `domain_trainer`.

**Outputs/state changes**
- Trained LoRA adapter state and stage-1 checkpoints.

#### Code
```python
from transformers.trainer_callback import ProgressCallback

domain_trainer.remove_callback(ProgressCallback)
domain_trainer.args.disable_tqdm = True

domain_train_stats = domain_trainer.train()
print(domain_train_stats)
```

#### Explanation
- Removes progress callback and disables tqdm display.
- Executes `train()`.
- Captures and prints training stats.

---

### Cell 11: Stage 2 JSON dataset loading

**Purpose**
- Load instruction dataset and convert to HF Dataset.

**Inputs**
- `cfg.instruction_json_path` JSON file.

**Outputs**
- `dataset` with required columns.

#### Code
```python
from datasets import Dataset

with open(cfg.instruction_json_path, "r", encoding="utf-8") as f:
    file = json.load(f)

print(f"Number of samples: {len(file)}")
if len(file) > 1:
    print(file[1])
```

#### Explanation
- Reads full JSON list into memory.
- Prints dataset size and one sample.

#### Code
```python
def clean_text(text):
    return text.strip('# ')
```

#### Explanation
- Minimal normalizer that removes edge `#` and spaces.
- Could fail if non-string values appear.

#### Code
```python
separated_data = {
    "instruction": [clean_text(item["instruction"]) for item in file],
    "input": [clean_text(item["input"]) for item in file],
    "response": [clean_text(item["response"]) for item in file]
}

dataset = Dataset.from_dict(separated_data)

if len(dataset) > 1:
    print(dataset[1])
```

#### Explanation
- Extracts required keys from each JSON record.
- Creates column-oriented dataset consumed in stage-2 trainer.

---

### Cell 12: Stage 2 trainer setup and prompt template

**Purpose**
- Build formatting function + training object for instruction stage.

**Inputs**
- `dataset`, `model`, `tokenizer`, stage-2 config.

**Outputs**
- `trainer` object.

#### Code
```python
from trl import SFTTrainer
from transformers import TrainingArguments
```

#### Explanation
- Re-imports trainer classes.

#### Code
```python
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["response"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = (
            f"### Instruction: {instruction}\n"
            f"### Input: {input_text}\n"
            f"### Response: {output}<|endoftext|>"
        )
        texts.append(text)
    return texts
```

#### Explanation
- Converts structured columns to flat training text sequences.
- Embeds explicit section markers; appends EOS-like terminator token text.
- This function defines the instruction-training data representation.

#### Code
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=cfg.max_seq_length,
    dataset_num_proc=cfg.dataset_num_proc,
    formatting_func=formatting_prompts_func,
    args=TrainingArguments(
        per_device_train_batch_size=cfg.stage2_per_device_train_batch_size,
        gradient_accumulation_steps=cfg.stage2_gradient_accumulation_steps,
        warmup_steps=cfg.stage2_warmup_steps,
        num_train_epochs=cfg.stage2_num_train_epochs,
        learning_rate=cfg.stage2_learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.stage2_logging_steps,
        optim=cfg.stage2_optim,
        weight_decay=cfg.stage2_weight_decay,
        lr_scheduler_type=cfg.stage2_lr_scheduler_type,
        seed=cfg.stage2_seed,
        output_dir=str(cfg.stage2_output_dir),
        save_strategy=cfg.stage2_save_strategy,
        save_total_limit=cfg.stage2_save_total_limit,
        dataloader_pin_memory=cfg.stage2_dataloader_pin_memory,
        report_to=cfg.stage2_report_to,
        average_tokens_across_devices=cfg.stage2_average_tokens_across_devices,
    ),
)

print("✅ Stage 2 trainer ready.")
```

#### Explanation
- Creates stage-2 trainer on same model (already stage-1 adapted).
- Adds save limit to keep at most 2 checkpoints.
- Logs and optim behavior configured via cfg.

---

### Cell 13: Execute Stage 2 training

**Purpose**
- Run instruction tuning stage.

**Inputs**
- `trainer`.

**Outputs**
- Updated LoRA weights + stage-2 checkpoints.

#### Code
```python
trainer_stats = trainer.train()
print(trainer_stats)
```

#### Explanation
- Executes stage-2 training loop.
- Prints returned training metrics summary.

---

### Cell 14: Inference test

**Purpose**
- Validate model response generation after fine-tuning.

**Inputs**
- `model`, `tokenizer`, configured test prompt/input.

**Outputs**
- Printed generated response.

#### Code
```python
FastLanguageModel.for_inference(model)

instruction = cfg.test_instruction
input_data = cfg.test_input_data
```

#### Explanation
- Switches model to inference mode helper path.
- Retrieves test content from config.

#### Code
```python
prompt = (
    f"### Instruction: {instruction}\n"
    f"### Input: {input_data}\n"
    f"### Response: "
)
```

#### Explanation
- Builds inference prompt using training-compatible template.

#### Code
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
```

#### Explanation
- Tokenizes prompt and places tensors on selected device.

#### Code
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=cfg.inference_max_new_tokens,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
)
```

#### Explanation
- Generates continuation tokens with max-length cap.
- Sets pad token id to EOS for compatibility.

#### Code
```python
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("Model response:")
print(response)
```

#### Explanation
- Removes prompt prefix tokens from output.
- Decodes only model-generated continuation.
- Prints final text output.

---

### Cell 15: GGUF export + zip packaging

**Purpose**
- Save deployable model artifacts.

**Inputs**
- `model`, `tokenizer`, `cfg`, `hf_token`.

**Outputs**
- GGUF directory and zip archive.

#### Code
```python
if cfg.save_gguf:
    cfg.gguf_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving GGUF model to: {cfg.gguf_output_dir}")
    model.save_pretrained_gguf(
        str(cfg.gguf_output_dir),
        tokenizer,
        quantization_method=cfg.gguf_quantization_method,
        token=hf_token if hf_token else None,
    )
```

#### Explanation
- Runs only if export enabled.
- Creates output folder and writes GGUF files with selected quantization.

#### Code
```python
    if cfg.gguf_zip_path.exists():
        cfg.gguf_zip_path.unlink()

    shutil.make_archive(
        base_name=str(cfg.gguf_zip_path.with_suffix("")),
        format="zip",
        root_dir=str(cfg.gguf_output_dir),
    )
```

#### Explanation
- Removes old zip file if present.
- Archives entire GGUF folder into zip.

#### Code
```python
    print(f"✅ GGUF folder saved at: {cfg.gguf_output_dir}")
    print(f"✅ ZIP archive saved at: {cfg.gguf_zip_path}")
else:
    print("Skipped GGUF export.")
```

#### Explanation
- Prints final artifact locations or skip notice.

---

### Cell 16: Empty cell

**Purpose**
- No executable logic.

**Notes**
- Likely placeholder/future extension cell.

---

## 4. Variable and object tracking

| Variable/Object | Created in | Contains | Modified? | Used later in |
|---|---|---|---|---|
| `cfg` | Cell 2 | All configuration values + path properties | No (usually static) | Cells 3–15 |
| `hf_token` | Cell 3 | Resolved HF token string | No | Cells 4, 5, 6, 15 |
| `model` | Cell 6 | Base model (then LoRA-adapted) | Yes (Cell 7, trained in 10/13) | Cells 7–15 |
| `tokenizer` | Cell 6 | Tokenizer for model | No | Cells 9, 12, 14, 15 |
| `domain_dataset` | Cell 8 | Stage-1 text dataset from PDFs | No | Cell 9 |
| `domain_trainer` | Cell 9 | Stage-1 SFT trainer | Internal state during training | Cell 10 |
| `dataset` | Cell 11 | Stage-2 instruction dataset | No | Cell 12 |
| `trainer` | Cell 12 | Stage-2 SFT trainer | Internal state during training | Cell 13 |
| `domain_train_stats` | Cell 10 | Stage-1 train return stats | No | Printed only |
| `trainer_stats` | Cell 13 | Stage-2 train return stats | No | Printed only |
| `prompt` | Cell 14 | Inference prompt string | No | Tokenization in Cell 14 |
| `response` | Cell 14 | Decoded model output text | No | Printed in Cell 14 |

### Artifact path tracking
- Input PDFs: `cfg.pdf_dir` (default `/home/n0etem01/LLM/books`).
- Input JSON: `cfg.instruction_json_path` (default `/home/n0etem01/LLM/BigDataset.json`).
- Stage 1 outputs: `cfg.stage1_output_dir`.
- Stage 2 outputs: `cfg.stage2_output_dir`.
- GGUF directory: `cfg.gguf_output_dir`.
- GGUF zip: `cfg.gguf_zip_path`.

---

## 5. Methodology-to-code mapping

| Methodology component | Where in notebook |
|---|---|
| Environment setup | Cells 0, 1, 3 |
| Global configuration | Cell 2 |
| HF token/auth checks | Cells 3, 4 |
| Optional local model snapshot | Cell 5 |
| Llama model loading | Cell 6 |
| PEFT/LoRA setup | Cell 7 |
| Stage-1 dataset preparation (PDF) | Cell 8 |
| Stage-1 trainer setup | Cell 9 |
| Stage-1 fine-tuning | Cell 10 |
| Stage-2 dataset loading/cleaning | Cell 11 |
| Prompt template construction | Cell 12 (`formatting_prompts_func`) + Cell 14 (`prompt`) |
| Stage-2 trainer setup | Cell 12 |
| Stage-2 fine-tuning | Cell 13 |
| Inference/evaluation sample | Cell 14 |
| Model saving/export | Cell 15 |

### Tokenization mapping
- Implicit tokenization during training: `SFTTrainer` in Cells 9 and 12.
- Explicit tokenization for inference: `tokenizer(prompt, return_tensors="pt")` in Cell 14.

---

## 6. Important functions, classes, and parameters

### Functions/classes defined in notebook
1. `FineTuneCfg` (Cell 2)
   - Type: dataclass.
   - Input: default fields or overrides.
   - Output: config object with properties.
   - Why important: centralizes all behavior knobs.

2. `attach_lora_once(model, cfg)` (Cell 7)
   - Input: model + config.
   - Output: model with LoRA adapters.
   - Why important: applies PEFT once and prevents duplicate attachment.

3. `extract_text_from_pdf(pdf_path)` (Cell 8)
   - Input: PDF path.
   - Output: full text string from all pages.
   - Why important: raw data ingestion from documents.

4. `split_paragraphs(pages, chunk_size=50)` (Cell 8)
   - Input: text + chunk size.
   - Output: list of fixed-size text chunks.
   - Why important: converts long raw text into trainable segments.

5. `clean_text(text)` (Cell 11)
   - Input: string.
   - Output: stripped string.
   - Why important: light normalization for JSON fields.

6. `formatting_prompts_func(examples)` (Cell 12)
   - Input: batch dict with instruction/input/response.
   - Output: formatted text list.
   - Why important: defines training prompt contract for stage-2 SFT.

### External classes/tools used
- `FastLanguageModel` (Unsloth): load model, attach PEFT, inference mode, GGUF export.
- `SFTTrainer` (TRL): supervised fine-tuning on plain/structured text.
- `TrainingArguments` (Transformers): training hyperparameter container.
- `Dataset` (datasets): training data structure.
- `snapshot_download`, `whoami`, `hf_hub_download` (huggingface_hub): auth/download utilities.

### Important parameters
- `model_name='unsloth/llama-3-70b-bnb-4bit'`
- `load_in_4bit=True`
- `max_seq_length=2048`
- `lora_r=64`, `lora_alpha=128`, `target_modules=(...)`
- Stage1: epochs=1, LR=2e-4, batch=2, grad_acc=4
- Stage2: epochs=3, LR=2e-4, batch=2, grad_acc=4
- `gguf_quantization_method='q4_k_m'`

---

## 7. Hidden assumptions and implementation details

1. **Hard-coded filesystem root**
   - Default root `/home/n0etem01/LLM` assumes a specific machine layout.

2. **Input file expectations**
   - Stage-2 requires `BigDataset.json` (or overridden name) to exist.
   - Stage-1 expects PDFs in `books/` folder.

3. **JSON schema assumption**
   - Every JSON item must contain `instruction`, `input`, `response` keys and string-like values.

4. **GPU assumption**
   - Pipeline is designed around CUDA and mixed precision settings.
   - CPU fallback exists for inference tensor placement, but full training on CPU is impractical.

5. **Token/auth assumption**
   - Some operations assume valid HF token and gated model access.

6. **Version compatibility assumption**
   - Relies on interaction between torch/unsloth/trl/transformers/bitsandbytes versions.

7. **Prompt-format assumption**
   - Stage-2 and inference use matching template with `### Instruction/Input/Response`.
   - Deviating format at inference may degrade behavior.

8. **PDF cleanup behavior is custom**
   - whitespace restoration logic is heuristic and can alter text semantics.

9. **No explicit validation stage**
   - Quality judged by train logs + one manual generation sample.

---

## 8. Common failure points

1. **Cell 0 package install failures**
   - Cause: network issues, incompatible CUDA wheel, dependency mismatch.

2. **Cell 1 CUDA unavailable**
   - Cause: missing GPU driver/runtime, wrong environment.

3. **Cell 3 file checks**
   - `BigDataset.json` missing triggers hard `FileNotFoundError`.

4. **Cell 4 auth failures**
   - Invalid token or missing access to gated repo causes `whoami`/`hf_hub_download` errors.

5. **Cell 6 model load OOM or download issues**
   - 70B 4-bit still requires significant VRAM + RAM + disk.

6. **Cell 8 text preprocessing oddities**
   - Custom whitespace logic may produce malformed chunks.

7. **Cell 9/12 trainer init errors**
   - Can fail if dataset empty, tokenizer mismatch, or package API differences.

8. **Cell 10/13 training OOM**
   - Despite LoRA/4-bit, memory may still be insufficient.

9. **Cell 11 JSON schema errors**
   - Missing keys or non-string values break list comprehensions/`clean_text`.

10. **Cell 15 GGUF export issues**
   - Export may fail depending on backend/tooling compatibility and model access.

11. **Notebook state/order issues**
   - Running out of order can leave undefined variables (`cfg`, `hf_token`, `model`, etc.).

---

## 9. Plain-language interpretation
This notebook teaches a large Llama model in two passes.

- First, it reads your PDFs and trains the model on that text so it becomes familiar with your domain language.
- Then it trains on instruction-style examples so the model learns how to answer in your target prompt format.
- After training, it asks one test question and can save the model in GGUF format for easier deployment.

So practically, it is a full pipeline to build a custom assistant from your documents plus Q&A-style examples.

---

## 10. Final technical summary
`LLM_Fine_Tuning_Llama_v2.ipynb` defines a configurable two-stage SFT pipeline using Unsloth and TRL on a 4-bit Llama checkpoint with LoRA adapters. Stage 1 performs domain adaptation from PDF-derived text chunks, and Stage 2 performs instruction tuning using JSON records formatted as `### Instruction / ### Input / ### Response` with `<|endoftext|>` termination. Training uses mixed precision (BF16 when supported, else FP16), 8-bit AdamW, gradient accumulation, epoch-based checkpointing, and no validation loop. The notebook ends with sample inference and optional GGUF export plus ZIP packaging.
