import os
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Lock

# --- KANDINSKY IMPORTS ---
# This assumes the script is running inside the 'kandinsky-5' folder
from kandinsky import get_T2V_pipeline

# --- CONFIGURATION ---
CONFIG_PATH = "configs/k5_lite_t2v_5s_distil_sd.yaml"
OUTPUT_DIR = "outputs"
DEVICE_MAP = {"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}

# --- CRITICAL FIX: Create folder BEFORE app starts ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
pipe = None
gpu_lock = Lock()


# --- MODEL LOADING ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print("\n🍏 INITIALIZING KANDINSKY STUDIO...")
    print("✨ Loading Distilled Model + SageAttention + Quantization...")

    try:
        # Initialize the pipeline with your optimized settings
        pipe = get_T2V_pipeline(
            device_map=DEVICE_MAP,
            conf_path=CONFIG_PATH,
            offload=True,  # Saves VRAM
            quantized_qwen=True,  # Fits text encoder in 16GB
            attention_engine="auto",  # Maximum Speed
        )
        print("✅ Model Loaded & Ready! Open http://localhost:8000 in your browser.\n")
    except Exception as e:
        print(f"\n❌ FATAL ERROR LOADING MODEL: {e}")
        print("Tip: If this is a CUDA error, re-install PyTorch Nightly.\n")
        raise e

    yield
    print("Shutting down...")


# --- APP SETUP ---
app = FastAPI(lifespan=lifespan)
# Mount the output folder so the browser can see the generated videos
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


class PromptRequest(BaseModel):
    prompt: str


# --- HTML FRONTEND (Apple Design) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kandinsky Studio</title>
    <style>
        :root {
            --bg-color: #F5F5F7;
            --card-bg: #FFFFFF;
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --accent-blue: #0071E3;
            --accent-hover: #0077ED;
            --shadow: 0 10px 30px rgba(0,0,0,0.08);
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            width: 100%;
            max-width: 680px;
            padding: 20px;
        }

        .card {
            background: var(--card-bg);
            border-radius: 24px;
            padding: 40px;
            box-shadow: var(--shadow);
            text-align: center;
            transition: transform 0.3s ease;
        }

        h1 {
            font-weight: 600;
            font-size: 28px;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        p.subtitle {
            color: var(--text-secondary);
            font-size: 15px;
            margin-bottom: 30px;
        }

        .input-group {
            margin-bottom: 25px;
            text-align: left;
        }

        textarea {
            width: 100%;
            padding: 16px;
            font-size: 17px;
            border: 1px solid #D2D2D7;
            border-radius: 12px;
            outline: none;
            resize: none;
            font-family: inherit;
            box-sizing: border-box;
            background: #FAFAFA;
            transition: all 0.2s;
            height: 100px;
        }

        textarea:focus {
            border-color: var(--accent-blue);
            background: #FFF;
            box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.15);
        }

        button {
            background-color: var(--accent-blue);
            color: white;
            border: none;
            padding: 14px 28px;
            font-size: 17px;
            font-weight: 500;
            border-radius: 980px; /* Pill shape */
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
        }

        button:hover {
            background-color: var(--accent-hover);
            transform: scale(1.01);
        }

        button:active {
            transform: scale(0.98);
        }

        button:disabled {
            background-color: #E5E5E5;
            color: #A1A1A6;
            cursor: not-allowed;
            transform: none;
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Video Result */
        #result-area {
            margin-top: 30px;
            display: none;
            animation: fadeIn 0.5s ease;
        }

        video {
            width: 100%;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            background: #000;
        }

        .status {
            margin-top: 15px;
            font-size: 14px;
            color: var(--text-secondary);
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

<div class="container">
    <div class="card">
        <h1>Kandinsky Studio</h1>
        <p class="subtitle">Powered by RTX 5080 & SageAttention</p>

        <div class="input-group">
            <textarea id="promptInput" placeholder="Describe your video imagination..."></textarea>
        </div>

        <button id="generateBtn" onclick="generateVideo()">
            <span id="btnText">Generate Video</span>
            <div class="spinner" id="btnSpinner"></div>
        </button>

        <div id="result-area">
            <video id="videoPlayer" controls loop autoplay muted playsinline></video>
            <div class="status" id="statusText">Generation complete</div>
        </div>
    </div>
</div>

<script>
    async function generateVideo() {
        const prompt = document.getElementById('promptInput').value;
        if (!prompt) return;

        // UI Updates
        const btn = document.getElementById('generateBtn');
        const btnText = document.getElementById('btnText');
        const spinner = document.getElementById('btnSpinner');
        const resultArea = document.getElementById('result-area');
        const statusText = document.getElementById('statusText');

        btn.disabled = true;
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        resultArea.style.display = 'none';

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });

            if (!response.ok) throw new Error("Generation failed");

            const data = await response.json();

            // Show Result
            const videoPlayer = document.getElementById('videoPlayer');
            // Add a timestamp to prevent browser caching the old video
            videoPlayer.src = data.url + "?t=" + new Date().getTime(); 
            resultArea.style.display = 'block';
            statusText.innerText = `Saved to: ${data.path}`;

        } catch (error) {
            alert("Error generating video: " + error.message);
        } finally {
            // Reset UI
            btn.disabled = false;
            btnText.style.display = 'block';
            spinner.style.display = 'none';
        }
    }
</script>

</body>
</html>
"""


# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.post("/generate")
async def generate(request: PromptRequest):
    if not gpu_lock.acquire(blocking=False):
        return {"error": "GPU is busy"}

    try:
        filename = f"vid_{int(time.time())}.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)

        # --- GENERATION LOGIC ---
        # Safe defaults for 16GB VRAM
        pipe(
            text=request.prompt,
            time_length=5,  # 5 Seconds
            width=512,  # Safe Resolution
            height=512,  # Safe Resolution
            save_path=save_path
        )

        return {"url": f"/outputs/{filename}", "path": save_path}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    finally:
        gpu_lock.release()


if __name__ == "__main__":
    # Runs on port 8000. Access via http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)