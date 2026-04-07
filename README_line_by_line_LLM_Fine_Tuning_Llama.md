# README Line-by-Line for `LLM_Fine_Tuning_Llama.ipynb`

## 1. Purpose of this notebook
This notebook builds a **two-stage fine-tuning pipeline** around a Llama 3 model using Unsloth + TRL in Google Colab.

At a high level, it does two separate training phases on the same model object:

1. **Domain adaptation stage**: ingest PDF documents, convert them into text chunks, and train with supervised fine-tuning (SFT) so the model internalizes domain language/content.
2. **Instruction tuning stage**: ingest a structured instruction dataset (`BigDataset.json`) and train on prompt-formatted instruction/input/response examples.

After these two stages, the notebook runs a small inference test and exports the final model to GGUF format for downstream deployment.

Important context from the code itself:
- This notebook is optimized for **Colab-like paths** (`/content/...`, `google.colab.drive`).
- It assumes availability of a large quantized model (`unsloth/llama-3-70b-bnb-4bit`).
- It uses PEFT/LoRA adapters rather than full model finetuning.

## 2. Big-picture methodology map
The methodology, in execution order, is:

1. **Environment setup**
   - Install libraries: `PyMuPDF`, `unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes`.
2. **Hardware check**
   - Detect CUDA and print GPU name.
3. **Model bootstrapping**
   - Load Llama model + tokenizer in 4-bit mode via Unsloth.
4. **Stage 1 data loading (PDF)**
   - Read all PDFs from `/content/books`, extract text, split into chunks.
5. **Stage 1 dataset creation**
   - Create Hugging Face `Dataset` with one `text` field.
6. **Stage 1 LoRA setup + trainer config**
   - Attach LoRA adapters and configure `SFTTrainer`.
7. **Stage 1 training**
   - Train for 1 epoch on domain text chunks.
8. **Stage 2 data loading (instruction JSON)**
   - Load `BigDataset.json`, clean strings, build dataset columns.
9. **Stage 2 prompt formatting + trainer config**
   - Define instruction prompt template function; configure second `SFTTrainer`.
10. **Stage 2 training**
    - Train for 3 epochs.
11. **Inference smoke test**
    - Build prompt, tokenize, generate, decode response.
12. **Saving/export**
    - Mount Google Drive, save GGUF, zip model folder, copy archive to Drive.

## 3. Cell-by-cell and line-by-line explanation

> Notes on structure:
> - The notebook has **34 cells** (some markdown, some code, two empty code cells).
> - Below, code cells are explained with line-by-line or small grouped snippets.
> - “Cell N” refers to the notebook’s zero-based index.

---

### Cell 0: Notebook title (markdown)
**Purpose:** Label the notebook as a two-stage fine-tuning workflow.

**Inputs:** None.

**Outputs/state changes:** None (documentation only).

**Why it matters:** Sets expectation that there are two phases of training.

---

### Cell 1: Install dependencies
**Purpose:** Install runtime packages required by later cells.

**Inputs:** Internet/package index availability.

**Outputs/state changes:** Colab environment gains installed libraries.

**Why this cell matters:** All later imports and training utilities depend on these installs.

#### Code
```python
!pip install PyMuPDF
!pip install unsloth trl peft accelerate bitsandbytes
```

#### Explanation
- `!pip install PyMuPDF`
  - Literal action: Runs shell command to install PyMuPDF.
  - Purpose in notebook: Provides `fitz` module for PDF text extraction.
  - Methodology connection: Needed for Stage 1 corpus creation.
  - Notes/cautions: Runtime restart may be needed in some environments.

- `!pip install unsloth trl peft accelerate bitsandbytes`
  - Literal action: Installs LLM fine-tuning stack.
  - Purpose in notebook: Enables model loading, LoRA adapters, SFT training, and quantized optimization.
  - Methodology connection: Core infrastructure for both training stages.
  - Notes/cautions: Version compatibility can break if latest packages drift.

---

### Cell 2: GPU availability check
**Purpose:** Verify hardware acceleration.

**Inputs:** Local CUDA runtime state.

**Outputs/state changes:** Prints CUDA availability and device name.

**Why this cell matters:** 70B model finetuning in 4-bit strongly depends on GPU.

#### Code
```python
# For GPU check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Explanation
- `import torch`
  - Literal action: Imports PyTorch.
  - Purpose in notebook: Needed for device checks and later dtype logic.
  - Methodology connection: Hardware-dependent training settings rely on this.

- `torch.cuda.is_available()` print
  - Literal action: Reports whether CUDA backend is visible.
  - Purpose in notebook: Quick preflight check.
  - Methodology connection: Influences `fp16` vs `bf16` flags later.

- `torch.cuda.get_device_name(0) ...`
  - Literal action: Retrieves GPU model string when CUDA is available.
  - Purpose in notebook: Confirms actual accelerator assigned.
  - Notes/cautions: Index `0` assumes at least one CUDA device.

---

### Cell 3: Section header (markdown)
**Purpose:** Marks model selection phase.

**Inputs/outputs:** None.

**Why this cell matters:** Documents transition from setup to model loading.

---

### Cell 4: Load Llama model and tokenizer with Unsloth
**Purpose:** Initialize base model and tokenizer.

**Inputs:** Model repo name, sequence length, quantization mode.

**Outputs/state changes:** Creates `model` and `tokenizer` objects.

**Why this cell matters:** This is the foundation for both training stages.

#### Code
```python
from unsloth import FastLanguageModel
import torch

#model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_name = "unsloth/llama-3-70b-bnb-4bit"

max_seq_length = 2048  # Choose sequence length
dtype = None  # Auto detection

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)
```

#### Explanation
- `from unsloth import FastLanguageModel`
  - Literal action: Imports Unsloth model wrapper.
  - Purpose in notebook: Simplifies quantized loading + PEFT integration.
  - Methodology connection: Used in model load, adapter setup, inference mode.

- `import torch`
  - Literal action: Ensures torch is in scope for later cells.
  - Purpose: dtype and precision logic later depends on torch.

- `#model_name = "...8b..."` (commented)
  - Literal action: Inactive alternative model option.
  - Purpose: Shows optional smaller model.
  - Notes/cautions: Indicates experimentation path.

- `model_name = "unsloth/llama-3-70b-bnb-4bit"`
  - Literal action: Selects 70B 4-bit model repo.
  - Purpose: High-capacity base model.
  - Methodology connection: Large base + LoRA adaptation strategy.
  - Notes/cautions: Resource heavy.

- `max_seq_length = 2048`
  - Literal action: Sets context length used in trainers/tokenization.
  - Purpose: Controls sequence truncation and memory usage.

- `dtype = None`
  - Literal action: Leaves dtype selection to auto behavior.
  - Purpose: Let backend infer best precision.

- `FastLanguageModel.from_pretrained(...)`
  - Literal action: Downloads/loads quantized model and tokenizer.
  - Purpose: Instantiates trainable base.
  - Methodology connection: Bootstrap before stage 1/2 fine-tuning.
  - Notes/cautions: Requires network and compatible CUDA setup.

---

### Cell 5: Stage 1 header (markdown)
**Purpose:** Announces domain-specific fine-tuning stage.

---

### Cell 6: Subheader for PDF processing (markdown)
**Purpose:** Documents that next cells prepare PDF text.

---

### Cell 7: PDF text extraction function
**Purpose:** Define utility to read text from a PDF file.

**Inputs:** `pdf_path` string.

**Outputs/state changes:** Returns concatenated document text.

**Why this cell matters:** Stage 1 training data comes from this function.

#### Code
```python
import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

#### Explanation
- `import fitz`
  - Literal action: Imports PyMuPDF module.
  - Purpose: PDF parsing backend.

- `def extract_text_from_pdf(pdf_path):`
  - Literal action: Defines reusable function.
  - Purpose: Encapsulates extraction logic for many files.

- `doc = fitz.open(pdf_path)`
  - Literal action: Opens PDF object.
  - Purpose: Access pages sequentially.

- `text = ""`
  - Literal action: Initialize accumulator.
  - Purpose: Concatenate page text.

- `for page in doc:`
  - Literal action: Iterate each page object.
  - Purpose: read whole document.

- `text += page.get_text()`
  - Literal action: Append extracted text per page.
  - Purpose: Build full raw corpus string.
  - Notes/cautions: No page delimiter inserted.

- `return text`
  - Literal action: Return full extracted string.
  - Methodology connection: Feed chunking in next cells.

---

### Cell 8: Text cleaning/chunking function
**Purpose:** Turn raw text into fixed-size chunks.

**Inputs:** `pages` (passed as text string later), optional `chunk_size`.

**Outputs/state changes:** Returns list of chunk strings.

**Why this cell matters:** Creates train samples for stage 1 `Dataset`.

#### Code
```python
import re

def split_paragraphs(pages, chunk_size=50):
    # Combine all pages into one large text
    full_text = " ".join(pages)

    # 1. Preserve actual word boundaries:
    # Replace 2 or more spaces (or newlines) with a temporary placeholder
    full_text = re.sub(r'\s{2,}', '___BOUNDARY___', full_text)

    # 2. Squash the stray letters together by removing all single spaces
    full_text = full_text.replace(' ', '')

    # 3. Restore the actual spaces between words
    full_text = full_text.replace('___BOUNDARY___', ' ')

    # 4. Clean up any remaining messy whitespace
    clean_text = re.sub(r'\s+', ' ', full_text).strip()

    # Split into actual words
    words = clean_text.split()

    # Group words into chunks
    paragraphs = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        paragraphs.append(chunk)

    return paragraphs
```

#### Explanation
- `import re`
  - Literal action: imports regex module.
  - Purpose: whitespace transformations.

- `def split_paragraphs(pages, chunk_size=50):`
  - Literal action: defines chunking function.
  - Purpose: produce uniform train chunks.

- `full_text = " ".join(pages)`
  - Literal action: joins iterable `pages` by spaces.
  - Purpose: build one string for cleanup.
  - Notes/cautions: In actual notebook, caller passes a single string (`raw_text`), so this joins characters, not pages. This is likely unintended/fragile.

- `re.sub(r'\s{2,}', '___BOUNDARY___', full_text)`
  - Literal action: replaces runs of whitespace with marker.
  - Purpose: attempt to preserve “true boundaries”.

- `full_text.replace(' ', '')`
  - Literal action: removes all single spaces.
  - Purpose: tries to remove stray spacing noise.
  - Notes/cautions: This aggressively alters token boundaries.

- `full_text.replace('___BOUNDARY___', ' ')`
  - Literal action: restore placeholder to real spaces.
  - Purpose: recover boundary markers.

- `clean_text = re.sub(r'\s+', ' ', full_text).strip()`
  - Literal action: normalize whitespace.
  - Purpose: tidy output before splitting.

- `words = clean_text.split()`
  - Literal action: tokenize by whitespace.
  - Purpose: convert to list for fixed-size grouping.

- loop with `range(0, len(words), chunk_size)`
  - Literal action: iterate chunk starts.
  - Purpose: make chunks of 50 words by default.

- `chunk = " ".join(words[i : i + chunk_size])`
  - Literal action: reassemble words into chunk text.
  - Methodology connection: each chunk becomes one training sample.

- `paragraphs.append(chunk)` + `return paragraphs`
  - Literal action: collect and return all chunks.
  - Notes/cautions: Name “paragraphs” is conceptual; actual chunks are fixed word windows.

---

### Cell 9: Bulk PDF loading and chunk aggregation
**Purpose:** Process all PDFs in folder and aggregate chunks.

**Inputs:** Files under `/content/books/*.pdf`.

**Outputs/state changes:** Populates `all_pdf_chunks`.

**Why this cell matters:** Produces the full stage 1 corpus.

#### Code
```python
import os
import glob

# 1. Define the directory where your books are stored
pdf_directory = "/content/books"

# 2. Find all files ending in .pdf within that directory
pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

# This master list will hold the chunks from EVERY book
all_pdf_chunks = []

# 3. Loop through each PDF file found
for pdf_path in pdf_files:
    print(f"Processing: {os.path.basename(pdf_path)}...")

    # Extract text from the current PDF in the loop
    raw_text = extract_text_from_pdf(pdf_path)

    # Split the text into chunks
    chunks = split_paragraphs(raw_text)

    # Add this book's chunks to our master list
    all_pdf_chunks.extend(chunks)

# 4. Verify the final results across all books
print("-" * 30)
print(f"Successfully processed {len(pdf_files)} PDF(s).")
print(f"Total chunks combined: {len(all_pdf_chunks)}")

if all_pdf_chunks:
    print(f"First chunk overall: {all_pdf_chunks[0]}")
else:
    print("No chunks were generated. Please check if the folder contains readable PDFs.")
```

#### Explanation
- `import os`, `import glob`
  - Literal action: imports path and pattern utilities.
  - Purpose: file discovery.

- `pdf_directory = "/content/books"`
  - Literal action: sets input folder path.
  - Purpose: defines where source PDFs are expected.
  - Notes/cautions: hardcoded Colab path.

- `pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))`
  - Literal action: builds list of PDF paths.
  - Purpose: batch process all books.

- `all_pdf_chunks = []`
  - Literal action: initialize accumulator list.
  - Purpose: combine chunks from every file.

- loop over `pdf_files`
  - Literal action: sequential processing of each document.
  - Methodology connection: corpus assembly phase.

- `raw_text = extract_text_from_pdf(pdf_path)`
  - Literal action: get full text per PDF.
  - Purpose: prepare for chunking.

- `chunks = split_paragraphs(raw_text)`
  - Literal action: chunk text.
  - Purpose: convert document into sample units.

- `all_pdf_chunks.extend(chunks)`
  - Literal action: append all chunks to master list.
  - Output change: grows training text corpus.

- print verification block
  - Literal action: logs counts and first sample.
  - Purpose: quick sanity check before dataset creation.

---

### Cell 10: Stage 1 dataset section header (markdown)
**Purpose:** Announces creation of HF dataset from chunks.

---

### Cell 11: Build Hugging Face dataset from chunks
**Purpose:** Convert Python list to `datasets.Dataset`.

**Inputs:** `all_pdf_chunks` list.

**Outputs/state changes:** Creates `domain_dataset`.

**Why this cell matters:** `SFTTrainer` expects dataset-like input.

#### Code
```python
from datasets import Dataset

# Create the dataset from the text chunks
domain_dataset = Dataset.from_dict({"text": all_pdf_chunks})

# Verify the dataset
print(domain_dataset)
```

#### Explanation
- `from datasets import Dataset`
  - Literal action: imports dataset class.
  - Purpose: structured training data container.

- `Dataset.from_dict({"text": all_pdf_chunks})`
  - Literal action: constructs table with one column `text`.
  - Purpose: align with `dataset_text_field="text"` in trainer.
  - Methodology connection: stage 1 data packaging.

- `print(domain_dataset)`
  - Literal action: prints schema + row count.
  - Purpose: confirm non-empty dataset.

---

### Cell 12: Inspect dataset sample
**Purpose:** Quick manual inspection of a late sample.

**Inputs:** `domain_dataset`.

**Outputs/state changes:** prints one sample.

#### Code
```python
print(domain_dataset[-20])
```

#### Explanation
- `domain_dataset[-20]`
  - Literal action: index from the end.
  - Purpose: inspect content quality away from first row.
  - Notes/cautions: fails if dataset has <20 rows.

---

### Cell 13: Attach LoRA adapters for stage 1
**Purpose:** Convert base model into PEFT trainable model.

**Inputs:** `model` from Cell 4.

**Outputs/state changes:** Reassigns `model` with LoRA adapters.

**Why this cell matters:** Makes training memory-feasible.

#### Code
```python
# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank - higher = more capacity, more memory
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # LoftQ
)
```

#### Explanation
- `FastLanguageModel.get_peft_model(...)`
  - Literal action: wraps model with PEFT/LoRA layers.
  - Purpose: train small adapter params instead of full 70B weights.
  - Methodology connection: core fine-tuning strategy.

- `r=64`
  - Literal action: LoRA rank.
  - Purpose: adapter capacity.
  - Notes: higher rank usually more expressiveness + memory.

- `target_modules=[...]`
  - Literal action: modules where LoRA injected.
  - Purpose: adapt attention + MLP projection paths.

- `lora_alpha=128`
  - Literal action: scaling factor.
  - Purpose: controls update magnitude.

- `lora_dropout=0`, `bias="none"`
  - Literal action: no dropout, no bias tuning.
  - Purpose: optimized path in Unsloth comments.

- `use_gradient_checkpointing="unsloth"`
  - Literal action: enables memory-saving checkpointing strategy.
  - Purpose: fit long sequences/large model.

- `random_state=3407`
  - Purpose: reproducibility of adapter init aspects.

- `use_rslora=False`, `loftq_config=None`
  - Literal action: disables optional advanced variants.
  - Purpose: keep standard LoRA path.

---

### Cell 14: Configure stage 1 trainer
**Purpose:** Define stage 1 training args and instantiate trainer.

**Inputs:** `model`, `tokenizer`, `domain_dataset`, `max_seq_length`.

**Outputs/state changes:** Creates `domain_training_args`, `domain_trainer`.

#### Code
```python
#!pip install -q trl
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Configure training arguments for domain pre-training
domain_training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="domain_pretrain_outputs",
    save_strategy="epoch",
    report_to="none"
)

# Initialize the SFTTrainer
domain_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=domain_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=domain_training_args,
)
```

#### Explanation
- `#!pip install -q trl` (commented)
  - Literal action: inactive install command.
  - Purpose: fallback reminder if TRL missing.

- imports `SFTTrainer`, `TrainingArguments`, `torch`
  - Purpose: trainer and config objects.

- `TrainingArguments(...)` block
  - `per_device_train_batch_size=2`
    - Literal: micro-batch size per GPU.
    - Purpose: memory control.
  - `gradient_accumulation_steps=4`
    - Literal: accumulate 4 steps before optimizer step.
    - Purpose: effective batch ~8 samples/device.
  - `warmup_steps=10`
    - Purpose: stabilize early training.
  - `num_train_epochs=1`
    - Purpose: short domain adaptation pass.
  - `learning_rate=2e-4`
    - Purpose: adapter learning rate.
  - `fp16=not torch.cuda.is_bf16_supported()` / `bf16=torch.cuda.is_bf16_supported()`
    - Literal: choose bf16 if supported else fp16.
    - Purpose: precision optimization by hardware.
  - `logging_steps=10`
    - Purpose: print periodic logs.
  - `optim="adamw_8bit"`
    - Purpose: memory-efficient optimizer via bitsandbytes.
  - `weight_decay=0.01`, `lr_scheduler_type="linear"`, `seed=3407`
    - Purpose: regularization, schedule, reproducibility.
  - `output_dir="domain_pretrain_outputs"`
    - Purpose: checkpoint location.
  - `save_strategy="epoch"`
    - Purpose: save each epoch end.
  - `report_to="none"`
    - Purpose: disable tracker integrations.

- `domain_trainer = SFTTrainer(...)`
  - `train_dataset=domain_dataset`, `dataset_text_field="text"`
    - Literal: trainer reads raw text from `text` column.
    - Methodology: stage 1 plain-text SFT.
  - `max_seq_length=max_seq_length`
    - Purpose: align token truncation with model context.
  - `dataset_num_proc=2`
    - Purpose: parallel data preprocessing workers.

---

### Cell 15: Stage 1 run header (markdown)
**Purpose:** Announces execution of first fine-tuning stage.

---

### Cell 16: Train stage 1 model
**Purpose:** Run training and print stats.

**Inputs:** `domain_trainer`.

**Outputs/state changes:** Updates adapter weights; stores `domain_train_stats`.

#### Code
```python
from transformers.trainer_callback import ProgressCallback

# Remove the ProgressCallback to stop the notebook progress bars from flooding the output
domain_trainer.remove_callback(ProgressCallback)
domain_trainer.args.disable_tqdm = True

# Train the model on the domain dataset
domain_train_stats = domain_trainer.train()

# Print training statistics
print(domain_train_stats)
```

#### Explanation
- `from transformers.trainer_callback import ProgressCallback`
  - Purpose: identify callback class to remove.

- `domain_trainer.remove_callback(ProgressCallback)`
  - Literal: unregister progress bar callback.
  - Purpose: reduce noisy notebook output.

- `domain_trainer.args.disable_tqdm = True`
  - Literal: disable tqdm UI.
  - Purpose: quieter logs.

- `domain_trainer.train()`
  - Literal: launches training loop.
  - Purpose: perform stage 1 adaptation.
  - Methodology connection: first of two training phases.

- `print(domain_train_stats)`
  - Purpose: show training result summary object.

---

### Cell 17: Stage 2 header (markdown)
**Purpose:** Announces instruction fine-tuning phase.

---

### Cell 18: Load instruction JSON dataset
**Purpose:** Read labeled instruction data file.

**Inputs:** `BigDataset.json` in working directory.

**Outputs/state changes:** Creates `file` list/dict structure.

#### Code
```python
import json

file = json.load(open("BigDataset.json"))
print(f"Number of samples: {len(file)}")
print(file[1])
```

#### Explanation
- `import json`
  - Purpose: parse JSON dataset.

- `json.load(open("BigDataset.json"))`
  - Literal: open + deserialize JSON.
  - Purpose: bring instruction examples into memory.
  - Notes/cautions: file must exist locally; no context manager used.

- `len(file)` print
  - Purpose: quick cardinality check.

- `print(file[1])`
  - Purpose: inspect example schema/content.
  - Notes/cautions: fails if dataset has <2 elements.

---

### Cell 19: Clean and restructure instruction dataset
**Purpose:** Normalize strings and build HF dataset fields.

**Inputs:** `file` object from Cell 18.

**Outputs/state changes:** Creates `dataset` with instruction/input/response columns.

#### Code
```python
def clean_text(text):
    # This removes leading/trailing '#' and whitespace
    return text.strip('# ')

# Reorganize the list, clean the text, and map to a dictionary of lists
separated_data = {
    "instruction": [clean_text(item["instruction"]) for item in file],
    "input": [clean_text(item["input"]) for item in file],
    "response": [clean_text(item["response"]) for item in file]
}

# Create the final dataset
dataset = Dataset.from_dict(separated_data)

# Print the second item to verify the cleaned output
print(dataset[1])
```

#### Explanation
- `def clean_text(text): return text.strip('# ')`
  - Literal: strips leading/trailing `#` and spaces.
  - Purpose: remove formatting artifacts.

- `separated_data = {...}`
  - Literal: builds dict of column -> list values.
  - Purpose: columnar format for dataset construction.
  - Methodology: stage 2 structured supervised dataset.
  - Notes/cautions: assumes each record has keys `instruction`, `input`, `response` and values compatible with `strip`.

- `dataset = Dataset.from_dict(separated_data)`
  - Literal: creates HF dataset.
  - Purpose: feed to stage 2 trainer.

- `print(dataset[1])`
  - Purpose: sanity check cleaned row.

---

### Cell 20: Re-apply LoRA adapter call before stage 2
**Purpose:** Ensure model is in PEFT configuration before second stage.

**Inputs:** existing `model`.

**Outputs/state changes:** reassigns `model` via `get_peft_model`.

#### Code
```python
# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank - higher = more capacity, more memory
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None, # LoftQ
)
```

#### Explanation
- Entire block mirrors Cell 13.
  - Literal action: calls LoRA injection utility again.
  - Purpose in notebook: likely to ensure adapters active for stage 2.
  - Methodology connection: keeps PEFT pathway for second training stage.
  - Notes/cautions: Reapplying adapter creation on already-adapted model can be ambiguous; exact behavior depends on Unsloth internals and notebook state.

---

### Cell 21: Stage 2 formatting function + trainer config
**Purpose:** Create instruction prompt format and stage 2 trainer.

**Inputs:** `dataset`, `model`, `tokenizer`, `max_seq_length`.

**Outputs/state changes:** defines `formatting_prompts_func`; creates `trainer`.

#### Code
```python
from trl import SFTTrainer
from transformers import TrainingArguments

# Define formatting function
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Format the text using the instruction structure
        text = f"### Instruction: {instruction}\n### Input: {input}\n### Response: {output}<|endoftext|>"
        texts.append(text)
    return texts

# Training arguments optimized for Unsloth
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    formatting_func=formatting_prompts_func, # Pass the formatting function here
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
        report_to="none", # Disable Weights & Biases logging
    ),
)
```

#### Explanation
- imports `SFTTrainer`, `TrainingArguments`
  - Purpose: stage 2 trainer setup.

- `def formatting_prompts_func(examples):`
  - Literal: transforms batch dict into list of strings.
  - Purpose: convert structured columns to single causal-LM text samples.

- `instructions = examples["instruction"]` etc.
  - Literal: extract batched columns.
  - Purpose: align fields for zipped iteration.

- loop `for instruction, input, output in zip(...)`
  - Literal: iterate example-wise triples.
  - Purpose: build prompt+target sample text.

- formatted string with `### Instruction`, `### Input`, `### Response`, plus `<|endoftext|>`
  - Literal: template creation.
  - Purpose: teach consistent instruction format and explicit termination token.
  - Methodology connection: core instruction-tuning format.

- `trainer = SFTTrainer(...)`
  - `train_dataset=dataset`
    - Purpose: stage 2 supervision source.
  - `dataset_text_field="text"` with `formatting_func=...`
    - Literal: trainer is told text field name while formatting function supplies transformed texts.
    - Notes/cautions: dataset does not have a literal `text` field; behavior relies on formatting function handling.
  - `num_train_epochs=3`
    - Purpose: longer instruction alignment than stage 1.
  - `logging_steps=25`, `save_total_limit=2`, `output_dir="outputs"`
    - Purpose: manage logs/checkpoints.
  - other settings mirror stage 1 defaults.

---

### Cell 22: Train stage 2
**Purpose:** Run second fine-tuning pass.

**Inputs:** `trainer`.

**Outputs/state changes:** updates model/adapters; creates `trainer_stats`.

#### Code
```python
# Train the model
trainer_stats = trainer.train()
```

#### Explanation
- `trainer.train()`
  - Literal action: executes training loop.
  - Purpose: instruction tuning on formatted dataset.
  - Methodology connection: final alignment stage.

- assignment to `trainer_stats`
  - Purpose: keep summary object for inspection/debugging.

---

### Cell 23: Empty code cell
**Purpose:** None (no code).

**Why it matters:** No state change.

---

### Cell 24: Empty code cell
**Purpose:** None.

---

### Cell 25: Inference test after training
**Purpose:** Quick qualitative check of tuned model output.

**Inputs:** trained `model`, `tokenizer`; hardcoded instruction and input string.

**Outputs/state changes:** prints generated response text.

#### Code
```python
# Test the fine-tuned model
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# 1. Define your instruction and input
instruction = "What is my 3D printer doing? Be specific"
input_data = "[170.0, 60.5, 1.4, -1.4, 0.0, 1.1, 1.1, 0.5, 0.9, 0]"

# 2. Match your EXACT training format, leaving the space after "Output: " empty for the model
prompt = f"### Instruction: {instruction}\n### Input: {input_data}\n### Response: "

# 3. Tokenize the raw string
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 4. Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=False,  # Set to False to prevent KV cache shape mismatch after training
    pad_token_id=tokenizer.eos_token_id,
)

# 5. Slice the output to ignore the prompt tokens and ONLY keep the new generation
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]

# 6. Decode and print just the response section
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)
```

#### Explanation
- `FastLanguageModel.for_inference(model)`
  - Literal: switch model to inference-optimized mode.
  - Purpose: faster generation.

- `instruction` / `input_data`
  - Literal: define sample test query and telemetry-like input string.
  - Purpose: smoke test with expected use-case style.

- `prompt = f"### Instruction...### Response: "`
  - Literal: constructs prompt matching training template prefix.
  - Purpose: keep train/infer format aligned.

- `tokenizer(...).to("cuda")`
  - Literal: tokenize prompt and move tensors to GPU.
  - Notes/cautions: hardcoded CUDA requirement.

- `model.generate(...)`
  - `max_new_tokens=256`: cap response length.
  - `use_cache=False`: explicit workaround comment for cache mismatch issues.
  - `pad_token_id=tokenizer.eos_token_id`: avoid padding token issues.

- slicing with `input_length`
  - Literal: removes echoed prompt tokens from output sequence.
  - Purpose: isolate model continuation only.

- `tokenizer.decode(..., skip_special_tokens=True)` + print
  - Purpose: convert generated tokens to readable answer.

---

### Cell 26: Download/export section header (markdown)
**Purpose:** Announces model export steps.

---

### Cell 27: Acquire HF token (insecurely hardcoded)
**Purpose:** Prepare authentication token variable.

**Inputs:** none from runtime (token string is literal).

**Outputs/state changes:** sets `hf_token`.

#### Code
```python
from google.colab import userdata

# Get your Hugging Face token from Colab secrets.
# If you don't have one, please create it in your Hugging Face settings
# and then add it to Colab secrets under the name 'HF_TOKEN'.
hf_token = "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"
```

#### Explanation
- `from google.colab import userdata`
  - Literal action: imports Colab secrets API.
  - Purpose in notebook: intended secure token retrieval.

- comments mention secret retrieval pattern
  - Purpose: documentation for safer approach.

- `hf_token = "..."`
  - Literal action: hardcodes token string.
  - Notes/cautions: security risk; contradicts comment intent.
  - Methodology connection: required for some export/upload actions.

---

### Cell 28: Mount Google Drive
**Purpose:** Connect Colab runtime to Drive path.

#### Code
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Explanation
- import and mount call
  - Literal action: requests auth and mounts Drive filesystem.
  - Purpose: destination for saved GGUF and zip.
  - Notes/cautions: works only in Colab.

---

### Cell 29: Save model in GGUF format
**Purpose:** Export trained model for downstream inference ecosystems.

#### Code
```python
model.save_pretrained_gguf("/content/drive/MyDrive/LLM models/Llama 3/gguf_model", tokenizer, quantization_method="q4_k_m", token=hf_token)
```

#### Explanation
- `save_pretrained_gguf(path, tokenizer, quantization_method="q4_k_m", token=hf_token)`
  - Literal action: writes GGUF model artifacts to Drive path.
  - Purpose: produce portable quantized deployment format.
  - Methodology connection: final artifact generation.
  - Notes/cautions: relies on method availability in current Unsloth version and valid permissions.

---

### Cell 30: Zip exported folder and trigger download
**Purpose:** Package model folder as zip and download to local machine.

#### Code
```python
import os
from google.colab import files

# 1. Define the folder path and the name of the zip file you want to create
folder_path = "/content/gguf_model_gguf"
zip_name = "llama3_model_folder.zip"

# 2. Create the zip archive
# This command zips the folder at 'folder_path' into 'zip_name'
os.system(f"zip -r {zip_name} {folder_path}")

# 3. Download the zipped file
files.download(zip_name)
```

#### Explanation
- imports `os`, `files`
  - Purpose: shell zip + browser download.

- `folder_path = "/content/gguf_model_gguf"`
  - Literal action: sets source folder for zip.
  - Notes/cautions: may not match path used in Cell 29, so this can fail or zip wrong folder.

- `zip_name = "llama3_model_folder.zip"`
  - Purpose: archive filename.

- `os.system(f"zip -r ...")`
  - Literal: executes shell zip command recursively.
  - Purpose: package artifacts.

- `files.download(zip_name)`
  - Literal: triggers browser download in Colab.

---

### Cell 31: Mount Google Drive again
**Purpose:** Ensure Drive is mounted before copy.

#### Code
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Explanation
- Repeats mount step.
  - Purpose: redundancy if disconnected.
  - Notes/cautions: unnecessary if still mounted.

---

### Cell 32: Copy zip archive to Drive
**Purpose:** Persist zipped model into Drive folder.

#### Code
```python
!cp "/content/llama3_model_folder.zip" "/content/drive/MyDrive/LLM models/Llama 3"
```

#### Explanation
- shell `cp` command
  - Literal action: copies zip file to target directory.
  - Purpose: ensure file retained in cloud storage.
  - Notes/cautions: source file must exist from Cell 30.

---

### Cell 33: Flush and unmount drive
**Purpose:** Safely finish Drive operations.

#### Code
```python
drive.flush_and_unmount()
```

#### Explanation
- `drive.flush_and_unmount()`
  - Literal action: flushes pending writes and unmounts.
  - Purpose: reduce risk of incomplete save.

---

## 4. Variable and object tracking

| Variable / Object | Created in Cell | Contains | How it changes | Used later |
|---|---:|---|---|---|
| `model_name` | 4 | HF model repo string | Set once (8B alt is commented) | `from_pretrained` |
| `max_seq_length` | 4 | int (`2048`) | constant | Both trainers, token handling |
| `model` | 4 | Base quantized model | LoRA-wrapped (13), potentially re-wrapped (20), trained (16/22), inference mode (25) | all training/inference/export cells |
| `tokenizer` | 4 | Tokenizer for base model | mostly unchanged | trainers, inference, GGUF save |
| `extract_text_from_pdf` | 7 | function | unchanged | called in 9 |
| `split_paragraphs` | 8 | function | unchanged | called in 9 |
| `pdf_files` | 9 | list of PDF paths | generated once | loop in 9 |
| `all_pdf_chunks` | 9 | list of chunk strings | extended per PDF | dataset creation in 11 |
| `domain_dataset` | 11 | HF dataset (`text`) | read-only thereafter | stage 1 trainer |
| `domain_training_args` | 14 | TrainingArguments | constant | stage 1 trainer |
| `domain_trainer` | 14 | SFTTrainer | internal state updates during `train()` | stage 1 run in 16 |
| `domain_train_stats` | 16 | train output metrics/state | assigned once | printed in 16 |
| `file` | 18 | loaded JSON data list | read in 19 | stage 2 dataset prep |
| `clean_text` | 19 | helper function | unchanged | list comprehensions in 19 |
| `separated_data` | 19 | dict with instruction/input/response lists | built once | dataset creation |
| `dataset` | 19 | HF dataset with 3 fields | read-only | stage 2 trainer |
| `formatting_prompts_func` | 21 | formatting function | unchanged | passed to stage 2 trainer |
| `trainer` | 21 | stage 2 SFTTrainer | internal state changes on training | `train()` in 22 |
| `trainer_stats` | 22 | stage 2 train output | assigned once | (not further used) |
| `instruction`, `input_data`, `prompt` | 25 | inference prompt elements | local test variables | tokenization/generation in 25 |
| `inputs`, `outputs` | 25 | tokenized inputs and generated ids | computed once | response decoding |
| `hf_token` | 27 | HF token string | set once | GGUF save call in 29 |
| `folder_path`, `zip_name` | 30 | zip source/name strings | constant | zip + download |

## 5. Methodology-to-code mapping

| Methodology Concept | Cells | Key code signals |
|---|---|---|
| Environment setup | 1–2 | pip installs, CUDA check |
| Model loading | 4 | `FastLanguageModel.from_pretrained(...)` |
| Domain dataset prep | 7–12 | PDF extraction, chunking, `Dataset.from_dict({"text": ...})` |
| Stage 1 PEFT config | 13 | `get_peft_model(...)` |
| Stage 1 training config | 14 | `TrainingArguments(... output_dir="domain_pretrain_outputs")` |
| Stage 1 training run | 16 | `domain_trainer.train()` |
| Instruction dataset prep | 18–19 | `json.load`, `clean_text`, `Dataset.from_dict(separated_data)` |
| Prompt construction for stage 2 | 21 | `formatting_prompts_func(...)` template with `### Instruction/Input/Response` |
| Stage 2 training config | 21 | `TrainingArguments(... num_train_epochs=3, output_dir="outputs")` |
| Stage 2 training run | 22 | `trainer.train()` |
| Inference/evaluation | 25 | prompt creation, tokenizer, `model.generate`, decode |
| Model export | 27–33 | token, drive mount, `save_pretrained_gguf`, zip/copy/unmount |

## 6. Important functions, classes, and parameters

### Core functions/classes
- **`FastLanguageModel.from_pretrained` (Unsloth)**
  - What: Loads model/tokenizer with Unsloth optimizations.
  - Inputs here: `model_name`, `max_seq_length`, `dtype=None`, `load_in_4bit=True`.
  - Output: `(model, tokenizer)`.
  - Why it matters: initializes quantized base suitable for constrained fine-tuning.

- **`FastLanguageModel.get_peft_model` (Unsloth)**
  - What: Injects LoRA adapters into specified transformer modules.
  - Inputs: LoRA rank/alpha/dropout/bias, module list, checkpointing options.
  - Output: PEFT-wrapped model.
  - Why: enables parameter-efficient adaptation.

- **`SFTTrainer` (TRL)**
  - What: supervised fine-tuning trainer wrapper for causal LMs.
  - Inputs here: model/tokenizer/dataset plus training args and optionally `formatting_func`.
  - Output: trainer object with `.train()`.
  - Why: executes both stage 1 and stage 2 training loops.

- **`TrainingArguments` (Transformers)**
  - What: central training hyperparameter container.
  - Why: governs batch size, precision, optimizer, schedule, checkpointing.

- **`extract_text_from_pdf` (user-defined)**
  - What: gets concatenated text from each PDF.
  - Why: source data ingestion for domain stage.

- **`split_paragraphs` (user-defined)**
  - What: whitespace normalization + fixed-length chunking.
  - Why: transforms raw text into trainable samples.

- **`formatting_prompts_func` (user-defined)**
  - What: maps stage 2 structured fields to single prompt string.
  - Why: aligns supervised data with instruction-tuning format.

### Important parameter groups
- **Model params:** `model_name`, `max_seq_length`, `load_in_4bit`, `dtype`.
- **LoRA params:** `r=64`, `lora_alpha=128`, `lora_dropout=0`, `target_modules=[...]`, `use_gradient_checkpointing="unsloth"`.
- **Training params (both stages):** batch size 2, grad accumulation 4, lr `2e-4`, optimizer `adamw_8bit`, linear scheduler.
- **Stage-specific differences:** epochs (1 vs 3), output dirs (`domain_pretrain_outputs` vs `outputs`), logging frequency and checkpoint cap.

## 7. Hidden assumptions and implementation details
- Assumes notebook is run in **Google Colab** (use of `/content`, `google.colab.drive`, `files.download`).
- Assumes large GPU is available for `llama-3-70b-bnb-4bit` even with 4-bit + LoRA.
- Assumes data files exist:
  - PDFs in `/content/books`.
  - `BigDataset.json` in current working directory.
- Assumes `BigDataset.json` schema has keys exactly: `instruction`, `input`, `response`.
- Assumes CUDA is available during inference (`to("cuda")` hardcoded).
- Token handling assumes sequence length 2048; long formatted samples may be truncated.
- Export workflow assumes GGUF conversion support in current package versions.
- Security assumption is broken: token is hardcoded rather than securely fetched.
- `split_paragraphs` implementation assumes certain PDF spacing artifacts; may distort text.

## 8. Common failure points
1. **Dependency/version mismatch**
   - `unsloth`, `trl`, `transformers`, `bitsandbytes` compatibility can fail.
2. **Model loading failures**
   - Network/auth/rate limits when pulling model.
   - GPU OOM despite quantization.
3. **Data path issues**
   - `/content/books` absent or empty.
   - `BigDataset.json` missing or malformed.
4. **Dataset shape issues**
   - Stage 2 trainer uses `dataset_text_field="text"` though dataset has no literal `text` field; relies on formatting function behavior.
5. **Runtime state issues**
   - Re-running cells out of order can rewrap model unexpectedly or stale trainers.
6. **Inference device issues**
   - `to("cuda")` breaks on CPU-only runtimes.
7. **Export mismatches**
   - `folder_path` in zip cell may not match actual save output path from GGUF cell.
8. **Security/credential issues**
   - hardcoded token can be invalid, revoked, or leaked.

## 9. Plain-language interpretation
This notebook teaches a Llama model in two rounds.

- First, it reads your PDF documents and trains the model on chunks of that text so the model gets familiar with your domain language.
- Second, it trains the model on instruction examples so it learns to answer questions in a specific prompt format.

Then it tests one sample question and saves the trained model in a deployment-friendly format (GGUF) to Google Drive.

## 10. Final technical summary
`LLM_Fine_Tuning_Llama.ipynb` is a Colab-first, two-stage Unsloth + TRL SFT pipeline: it loads `unsloth/llama-3-70b-bnb-4bit` in 4-bit mode, applies LoRA adapters (rank 64 over attention/MLP projection modules), performs stage 1 domain adaptation on PDF-derived text chunks, then stage 2 instruction tuning on `BigDataset.json` using a custom formatting function that emits `### Instruction / ### Input / ### Response` training strings. The notebook concludes with an inference smoke test and GGUF export to Drive, but has fragilities around path assumptions, dataset formatting assumptions, repeated LoRA wrapping, hardcoded credentials, and minimal quantitative evaluation.
