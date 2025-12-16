# 🤖 Project Context & Architecture (AGENTS.md)

**READ THIS BEFORE EDITING CODE.**

This document outlines the unique "Satellite" architecture of `Kandinsky-Studio`. As an AI agent working in this repository, you must understand that **you only see half the system.**

## ⚠️ CRITICAL: Environment Isolation & Testing
**You are currently working in the `Kandinsky-Studio` repository.**

1.  **Missing Dependencies:** The heavy AI engine (folder `kandinsky-5`) is **NOT** present in this workspace.
2.  **Import Errors are Normal:** If you try to analyze `web_ui.py`, you will see `ModuleNotFoundError` for `kandinsky`. **This is expected behavior.** Do not try to "fix" this by installing packages; the package exists only on the user's local machine, not in this repo.
3.  **No Runtime Testing:** You **cannot** run `web_ui.py` or `run.bat` in this environment. It will crash immediately.
    * *Action:* Do not attempt to execute generation tests.
    * *Action:* Assume that `get_T2V_pipeline` and `pipe()` work as described. Mock them if you are writing unit tests.

## 1. Architecture: The Twin-Folder System
This project operates as a UI Overlay attached to a separate Core Engine.

* **Host (Current Repo):** Contains `web_ui.py` (logic), `run.bat` (launcher), and UI assets.
* **Target (External Folder):** The user's local `kandinsky-5` folder, which contains the Torch environment and 10GB+ model weights.

**Operational Flow:**
1.  User runs `run.bat`.
2.  The script **copies** `web_ui.py` from here -> into the external `../kandinsky-5` folder.
3.  It activates the external virtual environment.
4.  It runs the script *inside* that external context.

*Constraint:* Any changes you make to imports in `web_ui.py` must be valid for the **Target** folder structure, not the current one.

## 2. The Inference Pipeline (`web_ui.py`)

### Model Loading (`lifespan`)
* **Global Persistence:** The model (`pipe`) is loaded into a global variable on startup.
* **Optimization Flags:**
    * `offload=True`: Aggressively moves layers to CPU (Critical for 16GB VRAM).
    * `quantized_qwen=True`: Uses NF4 quantization for the text encoder.
    * `attention_engine="auto"`: Uses PyTorch SDPA (Standard Nightly build). Do not change to "sage" unless explicitly requested, as it breaks on standard installs.

### Generation Logic (`/generate`)
* **Concurrency:** A `threading.Lock()` (`gpu_lock`) wraps the generation call. The model is not thread-safe.
* **Resolution:** Hardcoded to `512x512` to prevent OOM errors on RTX 5080/4080 class cards.

## 3. Future Roadmap / Agent Tasks
If you are asked to improve this repo, focus on these areas that do not require runtime testing:

1.  **Frontend Polish:** You can safely edit the `HTML_TEMPLATE` string in `web_ui.py` to improve the CSS/JS (e.g., adding progress bars, better error handling in JS).
2.  **API Structure:** You can modify the FastAPI routes and Pydantic models.
3.  **Launcher Logic:** You can improve `run.bat` error checking (Windows Batch scripting).