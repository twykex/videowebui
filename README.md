Here are your new documentation files.

1. README.md (The Public Guide)
This is updated to reflect the simplified "One-Click" setup using run.bat and the auto attention engine (so users don't get stuck installing SageAttention).

Markdown

# 🍏 Kandinsky 5.0 Studio

**A professional, Apple-inspired Web Interface for the [Kandinsky 5.0](https://github.com/kandinskylab/kandinsky-5) video generation model.**

Designed for high-end NVIDIA GPUs (RTX 3090, 4090, 5080/5090). It features a "Twin-Folder" architecture to keep your interface clean while leveraging the massive power of the Kandinsky engine.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-red) ![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

## ✨ Features
* **One-Click Launch:** Smart `run.bat` script handles environment switching and dependency checks.
* **Instant Inference:** Keeps the model loaded in RAM; no loading bars between generations.
* **Safety First:** optimized resolution settings (`512x512`) to prevent VRAM crashes on 16GB cards.
* **RTX 50-Series Ready:** Built for Blackwell architecture compatibility.

---

## 📂 Project Structure
This system relies on two repositories sitting side-by-side:

```text
Projects/
├── Kandinsky-Studio/      <-- [THIS REPO] The UI, Launcher, and Docs.
└── kandinsky-5/           <-- [ENGINE] The official model code & huge weights.
🚀 Installation
1. Clone Repositories
Create a folder for your project and clone both this UI and the engine:

PowerShell

git clone [https://github.com/YOUR_USERNAME/Kandinsky-Studio.git](https://github.com/YOUR_USERNAME/Kandinsky-Studio.git)
git clone [https://github.com/kandinskylab/kandinsky-5.git](https://github.com/kandinskylab/kandinsky-5.git)
2. Setup the Engine
Open a terminal in the kandinsky-5 folder:

PowerShell

cd kandinsky-5
python -m venv venv
.\venv\Scripts\activate

# A. Install PyTorch Nightly (CRITICAL for RTX 50-series / CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)

# B. Install Dependencies
pip install -r requirements.txt
pip install "numpy<2.3.0"  # Fix compatibility
3. Download Model Weights
Still in the engine folder, download the fast "Distilled" model:

PowerShell

python download_models.py --models kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s
⚡ How to Run
You never need to touch the terminal again.

Open the Kandinsky-Studio folder.

Double-click run.bat.

Wait for the server to start.

Open http://localhost:8000 in your browser.

🛠️ Troubleshooting
"FileNotFoundError: configs/..."

Your run.bat isn't switching folders correctly. Ensure you are using the latest version which uses cd /d.

"User provided device_type of 'cuda', but CUDA is not available"

You installed the wrong PyTorch. You must force re-install the Nightly build: pip install --pre torch ... --force-reinstall --index-url ...nightly/cu128

"Sage engine selected, but can't be imported"

Edit web_ui.py and ensure attention_engine="auto". This uses the built-in PyTorch acceleration which is stable and fast.