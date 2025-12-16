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
            --bg-gradient: radial-gradient(circle at 0% 0%, #e0f7fa 0%, #ffffff 50%, #f3e5f5 100%);
            --glass-bg: rgba(255, 255, 255, 0.65);
            --glass-border: rgba(255, 255, 255, 0.5);
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --accent-blue: #0071E3;
            --accent-glow: rgba(0, 113, 227, 0.3);
            --danger: #FF3B30;
            --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            --shadow-lg: 0 20px 40px -5px rgba(0, 0, 0, 0.1);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-gradient: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #000000 100%);
                --glass-bg: rgba(28, 28, 30, 0.65);
                --glass-border: rgba(255, 255, 255, 0.1);
                --text-primary: #F5F5F7;
                --text-secondary: #86868B;
            }
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
            background: var(--bg-gradient);
            background-size: 200% 200%;
            animation: gradientBG 15s ease infinite;
            color: var(--text-primary);
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Main Layout */
        main {
            width: 100%;
            max-width: 800px;
            padding: 40px 20px;
            box-sizing: border-box;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 40px;
        }

        /* Glass Card */
        .glass-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(25px) saturate(180%);
            -webkit-backdrop-filter: blur(25px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: var(--shadow-lg);
            transition: transform 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 20px;
        }

        h1 {
            font-weight: 700;
            font-size: 42px;
            margin: 0;
            letter-spacing: -1px;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge {
            display: inline-block;
            margin-top: 12px;
            padding: 6px 12px;
            background: rgba(0, 113, 227, 0.1);
            color: var(--accent-blue);
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* Input Area */
        .input-wrapper {
            position: relative;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            padding: 20px;
            font-size: 18px;
            line-height: 1.5;
            background: rgba(255, 255, 255, 0.5);
            border: none;
            border-radius: 18px;
            color: var(--text-primary);
            resize: none;
            font-family: inherit;
            box-sizing: border-box;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }

        @media (prefers-color-scheme: dark) {
            textarea { background: rgba(0, 0, 0, 0.2); }
        }

        textarea:focus {
            outline: none;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 0 0 4px var(--accent-glow);
        }

        /* Primary Button */
        .primary-btn {
            background: var(--accent-blue);
            color: white;
            border: none;
            padding: 18px 36px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 99px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .primary-btn:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 20px rgba(0, 113, 227, 0.3);
        }

        .primary-btn:active {
            transform: scale(0.98);
        }

        .primary-btn:disabled {
            background: var(--text-secondary);
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Gallery */
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 24px;
            margin-top: 40px;
            opacity: 0;
            animation: fadeIn 1s ease forwards;
        }

        .video-card {
            background: var(--glass-bg);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--glass-border);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
        }

        .video-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
        }

        video {
            width: 100%;
            display: block;
            background: #000;
        }

        .card-footer {
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .timestamp {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .download-btn {
            color: var(--accent-blue);
            text-decoration: none;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        /* Toast Notification */
        #toast-container {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .toast {
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            font-size: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
            animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeIn { to { opacity: 1; } }

        /* Spinner */
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

    </style>
</head>
<body>

    <main>
        <div class="glass-panel">
            <header>
                <h1>Kandinsky Studio</h1>
                <div class="badge">AI Motion Engine</div>
            </header>

            <div class="input-wrapper">
                <textarea id="promptInput" placeholder="Imagine a scene... e.g., 'A cyberpunk city in rain, neon lights reflecting on wet pavement'"></textarea>
            </div>

            <button id="generateBtn" class="primary-btn" onclick="generateVideo()">
                <span id="btnText">Generate Cinematic</span>
                <div class="spinner" id="btnSpinner" style="display: none;"></div>
            </button>
        </div>

        <div id="gallery" class="gallery">
            <!-- New videos appear here -->
        </div>
    </main>

    <div id="toast-container"></div>

    <script>
        // Icons
        const ICON_DOWNLOAD = `<svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg>`;

        async function generateVideo() {
            const promptInput = document.getElementById('promptInput');
            const prompt = promptInput.value.trim();

            if (!prompt) {
                showToast("Please enter a prompt first.", "error");
                promptInput.focus();
                return;
            }

            // UI State: Loading
            setLoading(true);

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });

                if (!response.ok) throw new Error("Server error");

                const data = await response.json();

                if (data.error) throw new Error(data.error);

                // Success
                addVideoToGallery(data.url, data.path);
                showToast("Video generated successfully!");
                promptInput.value = ""; // Clear input on success

            } catch (error) {
                console.error(error);
                showToast("Generation failed: " + error.message, "error");
            } finally {
                setLoading(false);
            }
        }

        function setLoading(isLoading) {
            const btn = document.getElementById('generateBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('btnSpinner');

            btn.disabled = isLoading;
            if (isLoading) {
                btnText.style.display = 'none';
                spinner.style.display = 'block';
                btn.style.opacity = '0.8';
            } else {
                btnText.style.display = 'inline';
                spinner.style.display = 'none';
                btn.style.opacity = '1';
            }
        }

        function addVideoToGallery(url, path) {
            const gallery = document.getElementById('gallery');
            const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            const card = document.createElement('div');
            card.className = 'video-card';

            // Prevent cache with timestamp query
            const videoUrl = `${url}?t=${Date.now()}`;

            card.innerHTML = `
                <video controls autoplay loop muted playsinline>
                    <source src="${videoUrl}" type="video/mp4">
                    Your browser does not support video.
                </video>
                <div class="card-footer">
                    <span class="timestamp">${timestamp}</span>
                    <a href="${videoUrl}" download="${path.split('/').pop()}" class="download-btn">
                        ${ICON_DOWNLOAD} Download
                    </a>
                </div>
            `;

            // Add to top
            gallery.prepend(card);
        }

        function showToast(message, type = "success") {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast';

            // Icon based on type
            const icon = type === 'error' ? '⚠️' : '✨';

            toast.innerHTML = `<span>${icon}</span> ${message}`;

            if (type === 'error') {
                toast.style.background = 'rgba(255, 59, 48, 0.9)';
            }

            container.appendChild(toast);

            // Remove after 3s
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateY(20px)';
                setTimeout(() => toast.remove(), 300);
            }, 3500);
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