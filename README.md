# 🍏 Kandinsky 5.0 Studio (Web UI)

A standalone, Apple-designed Web Interface for the [Kandinsky 5.0](https://github.com/kandinskylab/kandinsky-5) video generation model.

**Note:** This repository contains the frontend logic (`web_ui.py`). It is designed to run *inside* the official Kandinsky 5.0 environment.

## 🚀 Installation

1. **Clone the Official Repository:**
   First, set up the base model environment:
   ```bash
   git clone [https://github.com/kandinskylab/kandinsky-5.git](https://github.com/kandinskylab/kandinsky-5.git)
   cd kandinsky-5
   pip install -r requirements.txt
Install Web UI Dependencies:

Bash

pip install fastapi uvicorn
Add the Studio UI: Download web_ui.py from this repository and place it in the root of the kandinsky-5 folder.

⚡ How to Run
Run the script from inside the folder:

Bash

python web_ui.py
Open http://localhost:8000 to start generating videos instantly.


### **Step 3: Push to GitHub**
Now your folder contains exactly what you wanted: clean, standalone code without the heavy model files.

Run these commands to finish the upload:

```powershell
# 1. Initialize Git (if you haven't already)
git init

# 2. Add your 3 clean files
git add .

# 3. Commit
git commit -m "Initial release of Kandinsky Studio Web UI"

# 4. Connect to your NEW repo (Create 'Kandinsky-Studio' on GitHub first!)
# Replace YOUR_USERNAME with your actual GitHub username
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Kandinsky-Studio.git

# 5. Push
git push -u origin main
Result:

GitHub: Contains only web_ui.py, README.md, and requirements.txt.

Usage: People will download your script and drop it into their folder to get the beautiful interface