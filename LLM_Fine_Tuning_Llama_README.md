# README for `LLM_Fine_Tuning_Llama.ipynb`

## 1. High-level overview
This notebook implements a **two-stage fine-tuning workflow** for a Llama 3 model using the Unsloth ecosystem:

1. **Stage 1 (domain adaptation):** It extracts text from PDF files, chunks that text, and runs supervised fine-tuning (SFT) to adapt the base model to domain language/style.
2. **Stage 2 (instruction tuning):** It loads a JSON instruction-response dataset (`BigDataset.json`), formats examples into an instruction prompt template, and runs a second SFT pass to align behavior with task instructions.

The notebook is designed for a Google Colab workflow, with package installation in early cells and model export to Google Drive at the end. It uses a 4-bit quantized Llama model (`unsloth/llama-3-70b-bnb-4bit`) and LoRA adapters to make fine-tuning feasible on limited GPU memory.

In short: this is an end-to-end Colab pipeline that starts from raw domain PDFs + instruction JSON data, then trains and exports a task-specific Llama model.

## 2. Main purpose of the notebook
The practical goal is to produce a Llama-based model that is:

- **familiar with domain-specific textual content** (via PDFs in `/content/books`), and
- **able to follow structured instructions and generate responses** (via `BigDataset.json` instruction tuning).

The engineering intent appears to be a blend of:

- lightweight domain pretraining-like adaptation using SFT over raw text chunks, and
- downstream instruction behavior shaping using a custom prompt format (`### Instruction`, `### Input`, `### Response`).

Given the final inference test prompt (“What is my 3D printer doing?” with numeric sensor-like input), the notebook appears targeted at manufacturing / machine-state interpretation use cases.

## 3. Methodology
The notebook follows this methodology:

### Step A — Environment + model bootstrap
- Install required packages (`PyMuPDF`, `unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes`).
- Check GPU availability via `torch.cuda`.
- Load a quantized Llama 3 model through `FastLanguageModel.from_pretrained(...)` with:
  - `model_name = "unsloth/llama-3-70b-bnb-4bit"`
  - `max_seq_length = 2048`
  - `load_in_4bit=True`

### Step B — Stage 1: Domain-specific fine-tuning
- Extract text from PDFs with `fitz` (`extract_text_from_pdf`).
- Preprocess/chunk text with `split_paragraphs(...)` and a `chunk_size` of 50 words.
- Build Hugging Face `Dataset` with one column: `text`.
- Add LoRA adapters (`FastLanguageModel.get_peft_model(...)`) with rank 64 and target projection modules.
- Train with `trl.SFTTrainer` using `dataset_text_field="text"` and `TrainingArguments` (`num_train_epochs=1`, `learning_rate=2e-4`, batch and accumulation settings, linear scheduler, 8-bit AdamW optimizer).

### Step C — Stage 2: Instruction fine-tuning
- Load `BigDataset.json` and normalize text with `clean_text(...)` (`strip('# ')`).
- Build dataset columns: `instruction`, `input`, `response`.
- Re-apply LoRA setup (same adapter config function call is used again).
- Define `formatting_prompts_func(...)` to convert each sample into:
  - `### Instruction: ...`
  - `### Input: ...`
  - `### Response: ...<|endoftext|>`
- Train another `SFTTrainer` pass (3 epochs) with `formatting_func=formatting_prompts_func`.

### Step D — Inference and export
- Switch model to inference mode (`FastLanguageModel.for_inference(model)`).
- Run a sample generation with the same instruction template and print decoded output.
- Export model as GGUF (`save_pretrained_gguf(..., quantization_method="q4_k_m")`) to Google Drive.
- Zip and copy artifacts to Drive.

## 4. Notebook workflow, section by section

### 4.1 Setup and dependency installation
**Purpose:** Prepare Colab environment.

**Important code:**
- `!pip install PyMuPDF`
- `!pip install unsloth trl peft accelerate bitsandbytes`
- CUDA/GPU print checks.

**Inputs/outputs:**
- Input: Colab runtime.
- Output: installed Python packages, confirmation of GPU visibility.

**Why it matters:**
- The rest of the notebook depends on Unsloth + TRL + quantization tooling.

---

### 4.2 Base model selection and loading
**Purpose:** Load quantized Llama backbone and tokenizer.

**Important code:**
- `model_name = "unsloth/llama-3-70b-bnb-4bit"` (with 8B alternative commented out)
- `FastLanguageModel.from_pretrained(... load_in_4bit=True, max_seq_length=2048)`

**Inputs/outputs:**
- Input: remote model repo + runtime GPU.
- Output: `model`, `tokenizer` objects.

**Why it matters:**
- Determines model capacity, memory profile, and sequence length limits.

---

### 4.3 PDF ingestion and chunking (Stage 1 data prep)
**Purpose:** Convert raw PDF documents into trainable text segments.

**Important code:**
- `extract_text_from_pdf(pdf_path)` using `fitz.open` and `page.get_text()`.
- `split_paragraphs(...)` applies whitespace transforms, then chunks into groups of words (`chunk_size=50`).
- Loads `*.pdf` from `/content/books` and aggregates all chunks into `all_pdf_chunks`.

**Inputs/outputs:**
- Input: PDF files in `/content/books`.
- Output: `all_pdf_chunks` list.

**Why it matters:**
- Produces the domain corpus used for first-stage adaptation.

**Implementation caution:**
- `split_paragraphs` is named as paragraph splitting but actually performs heavy whitespace manipulation and fixed-word chunking. It may merge tokens unexpectedly.

---

### 4.4 Domain dataset creation and first LoRA configuration
**Purpose:** Build Hugging Face dataset and attach LoRA adapters.

**Important code:**
- `Dataset.from_dict({"text": all_pdf_chunks})`
- `FastLanguageModel.get_peft_model(...)` with:
  - `r=64`
  - `lora_alpha=128`
  - `lora_dropout=0`
  - `bias="none"`
  - `use_gradient_checkpointing="unsloth"`
  - target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**Inputs/outputs:**
- Input: raw text chunk list.
- Output: `domain_dataset`, LoRA-augmented `model`.

**Why it matters:**
- Enables parameter-efficient training instead of full-weight updates.

---

### 4.5 Stage 1 training execution
**Purpose:** Run first-stage domain fine-tuning.

**Important code:**
- `domain_training_args = TrainingArguments(...)`
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=4`
  - `num_train_epochs=1`
  - `learning_rate=2e-4`
  - `optim="adamw_8bit"`
  - `output_dir="domain_pretrain_outputs"`
  - `save_strategy="epoch"`
- `domain_trainer = SFTTrainer(...)` with `dataset_text_field="text"`.
- Remove progress callback and disable tqdm.
- `domain_trainer.train()`.

**Inputs/outputs:**
- Input: `domain_dataset` + configured model.
- Output: domain-adapted LoRA parameters and training stats (`domain_train_stats`).

**Why it matters:**
- Establishes domain knowledge before instruction alignment.

---

### 4.6 Stage 2 instruction dataset preparation
**Purpose:** Load supervised instruction data and standardize text fields.

**Important code:**
- `file = json.load(open("BigDataset.json"))`
- `clean_text(text): return text.strip('# ')`
- Build `separated_data` with keys:
  - `instruction`
  - `input`
  - `response`
- `dataset = Dataset.from_dict(separated_data)`

**Inputs/outputs:**
- Input: local JSON file `BigDataset.json`.
- Output: cleaned structured `dataset`.

**Why it matters:**
- This data drives instruction-following behavior.

**Implementation caution:**
- The code assumes every item in `file` has all three keys with string-like values.

---

### 4.7 Stage 2 prompt formatting and trainer setup
**Purpose:** Transform structured fields into a single training text and train.

**Important code:**
- `formatting_prompts_func(examples)` builds formatted strings ending with `<|endoftext|>`.
- `trainer = SFTTrainer(... formatting_func=formatting_prompts_func, dataset_text_field="text", ...)`
- Stage 2 `TrainingArguments`:
  - `num_train_epochs=3`
  - `logging_steps=25`
  - `save_total_limit=2`
  - `output_dir="outputs"`

**Inputs/outputs:**
- Input: instruction dataset.
- Output: instruction-tuned model state and `trainer_stats`.

**Why it matters:**
- Converts general domain adaptation into task-following behavior with a stable prompt schema.

**Implementation caution:**
- `dataset_text_field="text"` is set even though dataset has `instruction/input/response` columns; this likely works only because `formatting_func` is supplied.

---

### 4.8 Inference test cell
**Purpose:** Quick validation with a realistic instruction/input example.

**Important code:**
- `FastLanguageModel.for_inference(model)`
- Prompt format mirrors training template.
- `model.generate(... max_new_tokens=256, use_cache=False, pad_token_id=tokenizer.eos_token_id)`
- Strips prompt tokens and decodes only generated continuation.

**Inputs/outputs:**
- Input: hardcoded instruction and numeric input list string.
- Output: printed model response.

**Why it matters:**
- Confirms end-to-end usability and template consistency.

---

### 4.9 Model export and Drive operations
**Purpose:** Persist trained model in portable format.

**Important code:**
- Retrieves a Hugging Face token (comment says from `userdata`, but token is hardcoded in code).
- Mounts Google Drive.
- `model.save_pretrained_gguf(..., quantization_method="q4_k_m", token=hf_token)`.
- Zips `/content/gguf_model_gguf` to `llama3_model_folder.zip`.
- Copies zip to `"/content/drive/MyDrive/LLM models/Llama 3"`.

**Inputs/outputs:**
- Input: trained model in memory, Drive mounted.
- Output: GGUF export directory + zip archive in Drive.

**Why it matters:**
- Produces deployment-friendly artifact.

**Implementation caution:**
- Hardcoded token and hardcoded Colab paths reduce portability and security.

## 5. Libraries, tools, and dependencies
Major dependencies and their role:

- **PyMuPDF (`fitz`)**
  - PDF text extraction.
- **unsloth**
  - Efficient LLM loading (`FastLanguageModel`) and PEFT integration.
- **trl (SFTTrainer)**
  - Supervised fine-tuning over text/instruction datasets.
- **transformers**
  - `TrainingArguments`, callback controls, generation.
- **peft**
  - LoRA mechanism (used indirectly via Unsloth wrapper).
- **bitsandbytes**
  - 4-bit model loading and 8-bit optimizer support.
- **accelerate**
  - Runtime acceleration/backend support for distributed/optimized training stacks.
- **datasets**
  - Dataset container for train inputs.
- **torch**
  - Core deep learning runtime + GPU checks.
- **google.colab (`drive`, `files`, `userdata`)**
  - Mount/export/download workflow.

## 6. Data and input format

### 6.1 Domain PDFs (Stage 1)
- Expected location: `/content/books`.
- File matching: `glob("*.pdf")`.
- Processing:
  - Extract text page-wise.
  - Whitespace normalization + aggressive space handling.
  - Chunk into 50-word segments.
- Result format:
  - `Dataset` with a single `text` column.

### 6.2 Instruction JSON (Stage 2)
- Expected file: `BigDataset.json` in working directory.
- Expected item schema:
  - `{"instruction": ..., "input": ..., "response": ...}`
- Cleaning step:
  - `clean_text` strips leading/trailing `#` and spaces.
- Prompt conversion:
  - `### Instruction: {instruction}\n### Input: {input}\n### Response: {response}<|endoftext|>`

### 6.3 Tokenization / dataset transformation
- Tokenization is handled inside `SFTTrainer`/tokenizer pipeline.
- For inference, explicit tokenizer call:
  - `tokenizer(prompt, return_tensors="pt").to("cuda")`.

## 7. Model and fine-tuning setup

### 7.1 Model choice
- Active model: `unsloth/llama-3-70b-bnb-4bit`.
- Commented alternative: `unsloth/llama-3-8b-Instruct-bnb-4bit`.

### 7.2 Fine-tuning strategy
- **PEFT LoRA** adapters over transformer projection modules.
- Quantized backbone (`load_in_4bit=True`) + 8-bit optimizer (`adamw_8bit`) = memory-efficient training.
- Two-stage SFT:
  - Stage 1 on raw domain chunks.
  - Stage 2 on formatted instruction data.

### 7.3 Key hyperparameters
Shared or repeated values:
- `r=64`, `lora_alpha=128`, `lora_dropout=0`, `bias="none"`
- `max_seq_length=2048`
- `learning_rate=2e-4`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4` (effective batch size ~8 per device)
- `warmup_steps=10`
- scheduler: `linear`
- seed: `3407`

Differences by stage:
- Stage 1 epochs: `1`
- Stage 2 epochs: `3`
- Stage 2 `save_total_limit=2`

## 8. Training logic

### 8.1 Batching and optimization
- Micro-batch size = 2 samples/device.
- Gradient accumulation = 4 steps.
- Effective update batch approximates 8 samples/device (ignoring sequence length variance).
- Optimizer: 8-bit AdamW.

### 8.2 Objective
- Uses `SFTTrainer` standard causal LM next-token prediction objective over prepared text sequences.
- No custom loss function is defined in the notebook.

### 8.3 Checkpointing and saving
- Both stages use `save_strategy="epoch"`.
- Stage 1 outputs to `domain_pretrain_outputs`.
- Stage 2 outputs to `outputs`, with total checkpoint cap = 2.

### 8.4 Logging and progress
- Stage 1 explicitly disables tqdm/progress callback.
- `logging_steps` is set (10 for stage 1, 25 for stage 2).
- `report_to="none"` disables WandB/other tracker backends.

### 8.5 Evaluation flow
- There is **no formal validation split or metric-based evaluation** in the notebook.
- Evaluation is a single ad-hoc generation test prompt after training.

## 9. Outputs and generated artifacts
Potential artifacts produced:

- Stage 1 checkpoints under `domain_pretrain_outputs/`.
- Stage 2 checkpoints under `outputs/`.
- In-memory trained model with LoRA adaptation.
- GGUF export at `/content/drive/MyDrive/LLM models/Llama 3/gguf_model`.
- Zip file: `llama3_model_folder.zip`.
- Copied archive in Google Drive target folder.

Additionally, notebook prints:
- dataset sizes/samples,
- training stats objects,
- test generation output text.

## 10. Practical interpretation
If this notebook runs successfully, a user gets:

- a domain-and-instruction-tuned Llama model that can answer in the custom instruction format,
- a deployment-friendly GGUF model suitable for local inference stacks that support GGUF,
- a reproducible (though Colab-centric) fine-tuning recipe that can be adapted to other manufacturing datasets.

## 11. Limitations and things to watch out for

### 11.1 Data/format fragility
- `split_paragraphs` manipulates spaces in a non-standard way; may distort extracted text.
- No guardrails for malformed `BigDataset.json` entries.
- No train/validation split.

### 11.2 Training/evaluation limitations
- No quantitative metrics (accuracy, perplexity, BLEU, task metrics, etc.).
- Single qualitative test prompt is insufficient to establish model quality.
- Re-applying `get_peft_model(...)` in stage 2 may be redundant or risky depending on model state.

### 11.3 Security and reproducibility issues
- Hugging Face token is hardcoded directly in notebook code (security risk).
- Colab-specific absolute paths (`/content/...`) reduce portability.
- Export cell zips `/content/gguf_model_gguf` while save path uses Drive directory; potential path mismatch.

### 11.4 Resource assumptions
- 70B model, even in 4-bit, may require substantial GPU memory/runtime constraints.
- Notebook assumes CUDA availability for inference (`.to("cuda")`).

## 12. Beginner-friendly summary
This notebook teaches a Llama model in two rounds:

1. First, it reads your PDF documents so the model learns your domain language.
2. Then, it trains on instruction-response examples so it learns how to answer your tasks.

After training, it tests one prompt and exports the model to Google Drive in GGUF format for easier use elsewhere.

## 13. Technical summary
A Colab-native two-stage Unsloth/TRL SFT pipeline loads `llama-3-70b-bnb-4bit`, applies LoRA (r=64 over attention/MLP projections), performs domain adaptation on PDF-derived plain text chunks, then instruction tuning on `BigDataset.json` using a custom instruction template via `formatting_func`, followed by ad-hoc inference and GGUF export (`q4_k_m`).

## 14. Suggested improvements

### Reproducibility
- Replace hardcoded secrets with secure `userdata.get(...)` usage only.
- Add deterministic data splits and persist config as a single structured block.
- Parameterize all paths (`/content/books`, output dirs, Drive paths).

### Data quality
- Replace current chunker with sentence/paragraph-aware splitter.
- Add dataset schema validation and null/empty sample filtering.
- Log dataset statistics (length distribution, truncation rate).

### Training quality
- Add validation dataset and periodic evaluation metrics.
- Save and compare stage checkpoints systematically.
- Add early stopping and learning-rate schedule diagnostics.

### Code/readability
- Refactor notebook into reusable functions or a training script.
- Consolidate duplicated LoRA setup code.
- Document intended prompt template and expected input schema in markdown cells.

### Deployment
- Verify export and zip paths are consistent.
- Add post-export load test for GGUF artifact.

## Observed vs Inferred

### Observed directly in notebook code
- Two SFT stages are implemented (domain text then instruction data).
- Model is loaded via `FastLanguageModel.from_pretrained` with 4-bit quantization.
- LoRA config and major training hyperparameters are explicitly defined.
- Inference test uses the same instruction/input/response template.
- GGUF export and Drive copy steps are included.

### Reasonable inferences (not explicitly proven in notebook)
- The end use case likely involves manufacturing state interpretation (inference prompt suggests 3D printer telemetry use).
- Stage 1 is intended as domain adaptation before instruction alignment.
- The whitespace-heavy PDF preprocessing may have been designed to repair poor PDF extraction artifacts, but may also introduce distortions.
- The notebook appears exploratory/prototype-oriented rather than production-ready due to minimal evaluation and hardcoded operational details.
