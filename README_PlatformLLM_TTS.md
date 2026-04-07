# README for `PlatformLLM TTS.py`

## 1. High-level overview
`PlatformLLM TTS.py` is a **real-time 3D-printer monitoring desktop application** that combines four main capabilities in one interface:

- serial sensor ingestion from an Arduino,
- live telemetry plotting (acceleration, motor currents, and temperatures),
- an LLM chat assistant grounded on current printer values,
- text-to-speech playback of AI responses.

The script is written as a single Tkinter app with many global variables and nested functions inside `main()`. It continuously reads comma-separated sensor values from a serial port, updates plot buffers, writes rows to CSV, and lets the user ask questions in a chat panel. For chat answers, it uses an Ollama-served Llama model (`llama-3-8b`) via LangChain, then converts the response to speech with gTTS and plays it using pygame.

This is **runtime inference software**, not a model training pipeline. There is no fine-tuning loop in this file.

## 2. Main purpose of the script
The script appears designed to provide an operator-facing “smart dashboard” for 3D printing experiments or monitoring sessions.

Practical goals visible from code:
- monitor machine behavior in real time,
- log machine data to CSV for later analysis,
- enable natural-language interpretation of current telemetry using an LLM,
- provide spoken output so users can hear status/guidance while observing the printer.

So, the likely engineering purpose is to reduce the gap between raw telemetry and actionable interpretation during operation.

## 3. Methodology
The script’s methodology is event-driven and thread-assisted:

1. **Initialize dependencies and global runtime state**
   - imports UI/plotting/serial/LLM/TTS modules,
   - initializes pygame mixer,
   - sets global arrays/flags and plotting limits.

2. **Set up LLM chain**
   - creates `OllamaLLM(model="llama-3-8b")`,
   - defines a prompt template with `{instruction}` and `{input}`,
   - composes `chain = prompt | model`.

3. **Build GUI and controls**
   - creates frames for header, plots, control buttons, and chat,
   - configures callbacks for start, save, zoom/pause, and chat submit.

4. **Acquire live data**
   - on start, opens serial port at 9600 baud,
   - runs a background thread (`pointsA`) that parses incoming CSV lines,
   - appends values to telemetry arrays and CSV file.

5. **Visualize telemetry**
   - uses Matplotlib `FuncAnimation` callbacks (`aPlotData`, `ampPlotData`, `tPlotData`) to refresh graphs.

6. **Run LLM + TTS on user prompt**
   - user message enters chat,
   - background thread calls `ask_llm()` with latest `llmdata`,
   - model response is inserted into chat and synthesized to audio via `gTTS`,
   - audio is played by `pygame.mixer.music`.

7. **Save and stop**
   - optional CSV export from collected buffers,
   - clean shutdown closes serial and GUI.

## 4. Script workflow, section by section

### 4.1 Imports and runtime initialization
**Purpose:** load all dependencies and initialize base runtime.

**Important code:**
- device print with `torch.cuda.is_available()`,
- `pygame.mixer.init()`,
- many global flags and data buffers.

**Inputs/outputs:**
- Input: Python environment with all packages installed.
- Output: ready process state and initialized audio mixer.

**Why it matters:** all later features rely on these imports and global state.

---

### 4.2 Global constants, buffers, and paths
**Purpose:** define telemetry storage, plot limits, run flags, and file path.

**Important code:**
- arrays: `accXdata`, `ampM1data`, `temdata`, etc.
- plotting bounds: `accymin/accymax`, `ampymin/ampymax`, etc.
- path: `drive_path = r"C:\Users\...\arduino_data.csv"`.

**Inputs/outputs:**
- Input: none.
- Output: mutable shared state used by functions/threads.

**Why it matters:** this script is stateful; these variables are central to control flow.

---

### 4.3 LLM prompt chain setup
**Purpose:** prepare language model inference pipeline.

**Important code:**
- `model = OllamaLLM(model="llama-3-8b")`
- prompt template:
  - instruction question,
  - real-time printer values (`{input}`).
- `chain = prompt | model`.

**Inputs/outputs:**
- Input: local Ollama runtime + model availability.
- Output: callable chain for question answering.

**Why it matters:** enables telemetry-aware chat responses.

---

### 4.4 `main()` and serial port scanning (`puertos_seriales`)
**Purpose:** define app lifecycle and discover COM ports.

**Important code:**
- scans `COM1..COM256`, tries open/close, keeps available list,
- sets first found port in `port1`.

**Inputs/outputs:**
- Input: host serial devices.
- Output: list for combobox selection.

**Why it matters:** data capture depends on opening correct serial device.

**Notable issue:** file-header creation code appears after `return` in this function and is unreachable.

---

### 4.5 Start routine (`Iniciar`)
**Purpose:** reset buffers, connect serial, start acquisition thread, initialize plot canvases and animations.

**Important code:**
- reinitializes all telemetry arrays,
- opens serial: `serial.Serial(portUsed.get(), 9600, timeout=1)`,
- starts global thread `Thread(target=pointsA)`,
- creates/embeds acceleration/current/temperature figures and animations.

**Inputs/outputs:**
- Input: chosen COM port.
- Output: running acquisition + live plots.

**Why it matters:** this transitions app from idle to active monitoring.

**Notable issue:** large duplicated `try/except` plotting block increases maintenance risk.

---

### 4.6 Acquisition loop (`pointsA`)
**Purpose:** continuously read serial packets, parse them, update buffers and CSV.

**Important code:**
- blocks until serial data available (`arduino.inWaiting()`),
- reads line and splits by comma,
- expects at least 11 values and maps into telemetry variables,
- updates `llmdata` (latest feature vector for LLM context),
- appends timestamp + raw values to `drive_path` CSV.

**Inputs/outputs:**
- Input: comma-separated serial line from firmware.
- Output:
  - updated in-memory arrays for plots,
  - updated `llmdata`,
  - appended CSV rows.

**Why it matters:** this is the data backbone for both plotting and LLM context.

---

### 4.7 Plot update callbacks (`aPlotData`, `ampPlotData`, `tPlotData`)
**Purpose:** render live data and maintain scrolling time window.

**Important code:**
- each function sets line data from global arrays,
- conditionally shifts x-limits when `current_time > view_time`,
- enables navigation controls.

**Inputs/outputs:**
- Input: global telemetry arrays.
- Output: refreshed Matplotlib lines in Tk canvas.

**Why it matters:** provides immediate visual diagnosis.

---

### 4.8 Control functions (stop/save/zoom/pause/navigation)
**Purpose:** user interaction and lifecycle controls.

**Important code:**
- `exit()`: stop thread, close serial, destroy root.
- `stop()`: stop animations and reset some state.
- `startSampling()/saveSampling()`: record/export selected data using pandas.
- `ZoomIN/ZoomOUT/pausePlot/playPlot/leftFunc/rightFunc`: visualization navigation.

**Inputs/outputs:**
- Input: button/menu actions.
- Output: UI state changes, export files.

**Why it matters:** makes dashboard usable in longer sessions.

**Notable issues:** `ambanim` is stopped but not clearly initialized in active path; `localtime` variable name conflicts with imported function.

---

### 4.9 Chat and TTS pipeline
**Purpose:** convert user question + telemetry into text and audio answer.

**Important code:**
- `insert_formatted_text`: regex-based markdown-ish formatting in chat output.
- `ask_llm(question)`:
  - reads `llmdata`,
  - invokes chain with `{"instruction": question, "input": str(printer_data)}`,
  - inserts result to chat,
  - calls `speak_text(result)`.
- `speak_text(text)`:
  - strips some markdown symbols (`*`, `` ` ``, `$`),
  - generates speech with `gTTS`,
  - writes to in-memory `BytesIO`,
  - plays via `pygame.mixer.music` and waits until playback ends.
- `send_message`: posts user text, adds “Thinking...”, runs `ask_llm` in daemon thread.

**Inputs/outputs:**
- Input: user message + latest sensor vector.
- Output: chat text + spoken response.

**Why it matters:** this is the core “LLM-assisted + TTS” value proposition.

---

### 4.10 GUI assembly and event loop
**Purpose:** create all frames/widgets, bind callbacks, launch app.

**Important code:**
- builds three-column layout (chat, plots, controls),
- sets combobox values from serial ports,
- binds Start/Record/Save/Exit and chat send actions,
- runs `root.mainloop()`.

**Inputs/outputs:**
- Input: none.
- Output: running interactive desktop application.

**Why it matters:** ties all modules into one executable interface.

## 5. Libraries, tools, and dependencies
Major dependencies and role in this script:

- **`serial` (pyserial):** read telemetry stream from microcontroller.
- **`tkinter`, `ttk`:** GUI layout/widgets/events.
- **`matplotlib` + `animation` + `FigureCanvasTkAgg`:** live plots embedded in Tkinter.
- **`pandas`/`csv`:** CSV logging/export.
- **`langchain_ollama.OllamaLLM`:** call local Ollama-served LLM.
- **`langchain_core.prompts.ChatPromptTemplate`:** build prompt with placeholders.
- **`gtts.gTTS`:** text-to-speech synthesis.
- **`pygame`:** in-app MP3 playback.
- **`torch`:** only used to detect CUDA and print device.
- **Also imported but limited/unclear usage:** `numpy`, `scipy.fftpack`, `seaborn`, `cProfile`, `PIL.ImageTk/Image`.

## 6. Data and input format

### 6.1 Expected runtime inputs
- **Serial packets** from Arduino over selected COM port.
- **User chat text** from Tk entry field.
- **Ollama model availability** (`llama-3-8b`).

### 6.2 Serial data shape (as parsed)
Code expects at least 11 comma-separated values mapped to:
1. nozzle temp,
2. acceleration X,
3. acceleration Y,
4. acceleration Z,
5. ambient temp,
6. ambient humidity,
7. motor current Z,
8. motor current Y,
9. motor current X,
10. filament current,
11. bed temp.

### 6.3 Preprocessing and prompt formatting
- Telemetry is minimally preprocessed: `split(',')` then `float(...)` conversions.
- LLM prompt template uses:
  - `instruction` = user question,
  - `input` = `str(llmdata)`.
- Chat output formatting parses markdown-like patterns for bold/italic/code/math display.

### 6.4 TTS text preparation
Before synthesis, `speak_text` removes a subset of markdown symbols (`*`, `` ` ``, `$`) using regex to avoid awkward spoken output.

## 7. Model or API setup

### 7.1 Model/service used
- `OllamaLLM(model="llama-3-8b")` from LangChain’s Ollama integration.
- This implies a local Ollama server and pulled model are required.

### 7.2 Configuration and parameters
- LLM model name is hardcoded: `llama-3-8b`.
- Prompt template defines assistant persona (“expert in 3D printing”).
- Serial link parameters are hardcoded to `9600` baud.
- CSV output path is hardcoded to a specific Windows OneDrive path.

### 7.3 Authentication/platform requirements
- No explicit API key in code for Ollama (local runtime expected).
- gTTS typically requires internet access to Google TTS backend.
- GUI requires a desktop-capable Python environment.

## 8. Processing logic

### 8.1 Text handling and prompt construction
- User enters message in `user_entry`.
- `send_message` validates non-empty text.
- `ask_llm` constructs LLM call payload with:
  - `instruction`: user text,
  - `input`: latest telemetry list (`llmdata`).

### 8.2 Inference/generation
- `chain.invoke(...)` executes prompt + model inference through LangChain.
- Result is inserted into chat via `add_message` / `insert_formatted_text`.

### 8.3 Speech synthesis
- `speak_text` sanitizes markdown symbols.
- `gTTS(text=clean_text, lang='en', slow=False)` generates speech.
- Audio written to `io.BytesIO`, loaded into `pygame.mixer.music`, played to completion.

### 8.4 Saving outputs/logging
- Continuous logging: each serial packet appended to CSV at `drive_path`.
- Optional save report: `saveSampling()` writes dataframe-based CSV with semicolon separator.
- Status/logging mainly via prints and chat text updates.

## 9. Outputs and generated artifacts
Outputs produced by this script:

- **Live GUI state:** telemetry plots + chat history.
- **Real-time CSV log:** append-only rows at hardcoded `drive_path`.
- **Optional report CSV:** generated by `saveSampling()`.
- **Audio output:** spoken response played in memory (not persisted as file).

No model checkpoints or fine-tuned weights are generated.

## 10. Practical interpretation
If run successfully, this script gives a user:
- a real-time monitoring dashboard,
- quick natural-language interpretation of current machine state,
- spoken AI responses while monitoring,
- basic data logging for post-run analysis.

In practical terms, it combines telemetry, visualization, and AI assistance in one operator interface.

## 11. Limitations and things to watch out for

### 11.1 Hard-coded environment assumptions
- Windows-specific serial naming (`COM...`).
- Hardcoded Windows file path for CSV output.
- Heavy dependence on GUI-capable environment.

### 11.2 Reliability and code quality risks
- Extensive global mutable state across threads.
- Broad `except:` blocks obscure root causes.
- Duplicate logic blocks in `Iniciar`.
- Typo in protocol string: `WM_DELATE_WINDOW` (likely intended `WM_DELETE_WINDOW`).

### 11.3 Data integrity / runtime issues
- Serial parsing assumes fixed packet schema without robust validation.
- `saveSampling` references arrays that may not be populated consistently in current flow.
- Potential thread lifecycle issues if trying to restart completed threads.

### 11.4 LLM/TTS constraints
- Ollama model must exist locally and be runnable.
- gTTS usually needs internet.
- TTS playback runs in worker thread but still blocks that thread until completion.

## 12. Beginner-friendly summary
This program is a 3D-printer dashboard with an AI voice assistant.

It reads live sensor data from an Arduino, shows plots, lets you ask questions, and speaks the AI answer out loud. It also saves sensor values to CSV. It does not train an AI model; it only uses an existing Llama model through Ollama.

## 13. Technical summary
`PlatformLLM TTS.py` is a monolithic Tkinter application that integrates serial telemetry ingestion, Matplotlib animation, CSV logging, LangChain-Ollama inference (`llama-3-8b`), and gTTS+pygame audio playback. The inference payload is grounded by latest telemetry (`llmdata`) through a prompt template. The design is functional but tightly coupled, state-heavy, and sensitive to environment assumptions.

## 14. Suggested improvements

### Architecture
- Split into modules (`serial_io.py`, `ui.py`, `llm.py`, `tts.py`, `storage.py`).
- Replace globals with state objects/dataclasses.

### Reliability
- Replace bare `except:` with specific exceptions.
- Remove duplicated try/except setup logic.
- Use thread-safe queues between acquisition and UI/LLM.

### Portability
- Move config (path, baud, model name) to external config file.
- Add cross-platform serial detection (`/dev/tty*`, etc.).

### Data quality
- Validate packet length/types before writing CSV.
- Add schema version and explicit headers for all output CSVs.

### UX
- Replace permanent “Thinking...” message with transient status indicator.
- Add connection status, packet rate, and error banner in UI.

### TTS/LLM robustness
- Add timeout/retry around Ollama invocation.
- Add optional offline/local TTS engine fallback.
- Add explicit prompt guardrails for uncertainty and safety.

## Observed vs Inferred

### Observed (directly in code)
- Uses `OllamaLLM(model="llama-3-8b")` with a prompt template containing `{instruction}` and `{input}`.
- Uses `gTTS` + `pygame` for speech playback from in-memory MP3 bytes.
- Reads serial values, parses at least 11 fields, updates telemetry arrays, logs CSV.
- Builds a Tkinter GUI with chat panel, plots, and controls.
- No training/fine-tuning code exists in this file.

### Inferred (reasonable interpretation)
- Intended for real-time operator support in a 3D-printing lab/workshop context.
- Some imports and variables suggest iterative evolution and partially unfinished features.
- Goal likely includes beginner guidance, not only raw data display, due to prompt persona and TTS integration.
