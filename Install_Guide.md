# 🚀 Installation & Setup Guide

Complete setup instructions for the AI-Powered Aerospace Design Assistant.

---

## 📋 Prerequisites

Before starting, ensure you have:
- **Python 3.10 or higher** ([Download](https://www.python.org/downloads/))
- **pip** (included with Python)
- **Git** ([Download](https://git-scm.com/downloads))
- **Anthropic API key** ([Get one here](https://console.anthropic.com/))

---

## 🐧 Linux Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/capstone-496.git
cd capstone-496
```

### Step 2: Create Virtual Environment
```bash
# Using venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# OR using conda
conda create -n aerospace python=3.10
conda activate aerospace
```

### Step 3: Upgrade pip and Install Build Tools
```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Project as Package
```bash
# Install in editable mode (important for imports to work)
pip install -e .
```

This installs your project so Python can properly find the `tools`, `graph`, and `nodes` modules.

### Step 5: Configure Environment Variables
```bash
# Create .env file
nano .env
```

Add your API key:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Save and exit (Ctrl+X, Y, Enter in nano).

### Step 6: Verify Installation
```bash
# Test imports
python -c "from tools import size_drone; from graph.workflow import build_design_graph; print('✅ Installation successful!')"

# Test the main script
python main.py --help
```

### Step 7: Run a Test Design
```bash
python main.py --design "drone with 2kg payload and 30 minutes flight time"
```

Expected output: Complete drone design with specifications.

### Step 8: (Optional) Start LangGraph Studio
```bash
# Install CLI if not already installed
pip install langgraph-cli

# Start studio
langgraph dev
```

Access at: `http://localhost:8123`

---

## 🍎 macOS Setup

### Step 1: Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python 3.10+
```bash
# Check if you have Python 3.10+
python3 --version

# If not, install it
brew install python@3.10
```

### Step 3: Clone the Repository
```bash
git clone https://github.com/yourusername/capstone-496.git
cd capstone-496
```

### Step 4: Create Virtual Environment
```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate

# Verify activation (you should see (.venv) in prompt)
which python
```

### Step 5: Upgrade pip and Install Build Tools
```bash
pip install --upgrade pip setuptools wheel
```

### Step 6: Install Project as Package
```bash
# Install in editable mode
pip install -e .
```

**⚠️ Apple Silicon (M1/M2/M3) Users:**

If you encounter issues with `numpy` or `scipy`:
```bash
brew install openblas
export OPENBLAS="$(brew --prefix openblas)"
pip install -e .
```

### Step 7: Configure Environment Variables
```bash
# Create .env file
touch .env
nano .env
# OR use your favorite editor
open -e .env
```

Add your API key:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 8: Verify Installation
```bash
# Test imports
python -c "from tools import size_drone; from graph.workflow import build_design_graph; print('✅ Installation successful!')"

# Test the CLI
python main.py --help
```

### Step 9: Run a Test Design
```bash
python main.py --design "fixed wing drone with 4 hours flight time and 4kg payload"
```

### Step 10: (Optional) Start LangGraph Studio
```bash
langgraph dev
```

Access at: `http://localhost:8123`

---

## 🪟 Windows Setup

### Step 1: Install Python
1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. **⚠️ IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Clone the Repository
```cmd
git clone https://github.com/yourusername/capstone-496.git
cd capstone-496
```

### Step 3: Create Virtual Environment

**Using Command Prompt:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Using PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get a PowerShell execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Upgrade pip and Install Build Tools
```cmd
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install Project as Package
```cmd
pip install -e .
```

**Common Windows Issues:**

**Issue 1**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**: Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Download the installer
- Select "Desktop development with C++"
- Install

**Issue 2**: ChromaDB installation fails

**Solution**:
```cmd
pip install chromadb --no-cache-dir
```

**Issue 3**: Long path issues

**Solution**: Enable long paths in Windows:
```cmd
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### Step 6: Configure Environment Variables

**Using Notepad:**
```cmd
notepad .env
```

**Using PowerShell:**
```powershell
New-Item .env -ItemType File -Force
notepad .env
```

Add your API key:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Save and close.

### Step 7: Verify Installation
```cmd
python -c "from tools import size_drone; from graph.workflow import build_design_graph; print('Installation successful!')"

python main.py --help
```

### Step 8: Run a Test Design
```cmd
python main.py --design "drone with 2kg payload and 30 minutes flight time"
```

### Step 9: (Optional) Start LangGraph Studio
```cmd
pip install langgraph-cli
langgraph dev
```

If `langgraph` command not found:
```cmd
python -m langgraph.cli dev
```

Access at: `http://localhost:8123`

---

## 🔑 Getting Your Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **Create Key**
5. Name it (e.g., "Aerospace Design Assistant")
6. Copy the key (starts with `sk-ant-`)
7. Paste into `.env` file

**⚠️ Security Notes:**
- Never commit `.env` to git (already in `.gitignore`)
- Never share your API key publicly
- Monitor your API usage to avoid unexpected costs
- Free tier: $5 credit, then pay-as-you-go

---

## 📊 Using LangGraph Studio

LangGraph Studio provides visual debugging and execution monitoring.

### Installation
```bash
pip install langgraph-cli
```

### Starting Studio
```bash
cd capstone-496
langgraph dev
```

### Accessing Dashboard
Open browser to: `http://localhost:8123`

### Features
- **Visual Graph**: See all nodes and connections
- **Step Execution**: Run node-by-node
- **State Inspector**: View state at each step
- **LLM Tracing**: See all API calls and responses
- **Debug Mode**: Set breakpoints and inspect

### Test Input Example
```json
{
  "raw_input": "design a fixed wing drone with 4 hours flight time and 4kg payload"
}
```

### Troubleshooting Studio

**Issue**: "Cannot import 'tools'"

**Solution**: Make sure you ran `pip install -e .` first!

**Issue**: Graph doesn't load

**Solution**: 
```bash
# Verify graph builds
python -c "from graph.workflow import build_design_graph; print(build_design_graph())"
```

**Issue**: Port already in use

**Solution**:
```bash
langgraph dev --port 8124
```

---

## 🧪 Testing Your Installation

### Quick Tests

**Test 1: Drone Design**
```bash
python main.py --design "drone with 2kg payload and 30 min flight time"
```

Expected: Drone specifications with motor, battery, flight time.

**Test 2: Fixed-Wing UAV**
```bash
python main.py --design "fixed wing drone with 4 hours flight time and 4kg payload"
```

Expected: Wing design, weight breakdown, range calculations.

**Test 3: Small Rocket**
```bash
python main.py --design "small rocket to reach 1km altitude with 1kg payload"
```

Expected: Staging, propellant mass, delta-v calculations.

**Test 4: Satellite**
```bash
python main.py --design "satellite for 400km orbit with 20kg payload"
```

Expected: Orbital parameters, power system, design life.

### Verification Checklist

All tests should show:
- LLM parameter completion with reasoning
- RAG document search results
- Data validation (if issues found)
- Calculation results
- Final design specifications
- No Python errors

---

## 🔧 Common Issues & Solutions

### Import Errors

**Problem**: `ImportError: cannot import name 'ALL_TOOLS'`

**Solution**: You didn't install the project as a package!
```bash
pip install -e .
```

**Problem**: `ModuleNotFoundError: No module named 'anthropic'`

**Solution**: Dependencies not installed:
```bash
pip install -r requirements.txt
```

### API Errors

**Problem**: `AuthenticationError: Invalid API key`

**Solution**: 
1. Check `.env` file exists
2. Verify API key format: `sk-ant-...`
3. No quotes around the key in `.env`

**Problem**: `RateLimitError: Too many requests`

**Solution**: 
- Wait a few seconds between requests
- Check your API usage limits
- Consider upgrading your plan

### ChromaDB Issues

**Problem**: `ChromaDB initialization failed`

**Solution**: Re-ingest documents:
```bash
rm -rf chroma_db/
python ingest_data.py
```

**Problem**: `No documents found in vector store`

**Solution**: Make sure research papers are in `data/papers/` directory.

### LangGraph Studio Issues

**Problem**: Graph won't load in studio

**Solution**: Check `langgraph.json` is in project root:
```json
{
  "dependencies": ["."],
  "graphs": {
    "aerospace_design": "./graph/workflow.py:build_design_graph"
  },
  "env": ".env"
}
```

**Problem**: "Backend unavailable" error

**Solution**: Upgrade setuptools:
```bash
pip install --upgrade setuptools wheel
pip install -e .
```

### Platform-Specific Issues

**macOS**: If `python` doesn't work, use `python3`
```bash
alias python=python3  # Add to ~/.zshrc or ~/.bashrc
```

**Windows**: If scripts don't run, activate virtual environment:
```cmd
.venv\Scripts\activate
```

**Linux**: Permission denied on scripts:
```bash
chmod +x main.py
```

---

## 🔄 Updating the Project

```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies (if requirements.txt changed)
pip install -e . --upgrade

# Re-ingest documents (if data structure changed)
python ingest_data.py
```

---

## 📂 Project Structure

```
capstone-496/
├── .env                    # API keys (create this)
├── langgraph.json         # LangGraph Studio config
├── requirements.txt       # Dependencies
├── setup.py              # Package configuration
├── main.py               # Main CLI entry point
├── graph/
│   ├── workflow.py       # LangGraph workflow
│   ├── nodes.py          # Core nodes
│   └── state.py          # State definitions
├── nodes/
│   ├── llm_supervisor.py        # Vehicle classification
│   ├── llm_parameter_completer.py # Smart param completion
│   ├── llm_data_validator.py   # Scale validation
│   └── ...
├── tools/
│   ├── drone_tools.py
│   ├── fixed_wing_tools.py
│   ├── rocket_tools.py
│   └── ...
├── data/
│   └── papers/          # Research papers for RAG
└── chroma_db/           # Vector database
```

---

## 💡 Usage Examples

### Basic Usage
```bash
# Simple design request
python main.py --design "your design requirements here"

# Interactive mode (planned)
python main.py --interactive

# Batch processing (planned)
python main.py --batch designs.txt
```

### Example Requests

**Surveillance Drone:**
```bash
python main.py --design "surveillance drone with 60 minute flight time, 2kg camera payload, 5km range"
```

**Light Sport Aircraft:**
```bash
python main.py --design "light sport aircraft for 2 passengers, 500km range, 150km/h cruise"
```

**CubeSat:**
```bash
python main.py --design "1U CubeSat for LEO at 450km altitude, 5 year mission"
```

**Model Rocket:**
```bash
python main.py --design "model rocket to reach 500m altitude, recoverable with parachute"
```

---

## 🎯 Next Steps

After successful installation:

1.  **Test all vehicle types** to familiarize yourself with outputs
2.  **Explore LangGraph Studio** to understand the workflow
3.  **Review the code** in `graph/nodes.py` to see node implementations
4.  **Read README.md** for technical details
5.  **Watch the demo video** (link in README)

---


**Ready to design! 🚀 Happy Building!**

