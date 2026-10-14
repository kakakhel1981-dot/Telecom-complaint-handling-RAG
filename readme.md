# 📡 Enterprise Telecom Customer Complaint Handling Assistant

An intelligent Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **FAISS**, **Sentence Transformers (`all-MiniLM-L6-v2`)**, **PyMuPDF**, and **Groq (`openai/gpt-oss-20b` / Llama 3)**.

## ✨ Key Features
- **PDF Document Ingestion**: Upload internal telecom standard operating procedures (SOPs), service level agreements (SLAs), or complaint handling policy manuals.
- **Semantic Search (FAISS + Sentence Transformers)**: Instantly retrieves exact policy sections relevant to customer disputes.
- **Groq LLM Integration**: Generates precise, professional agent responses and compliance guidance.
- **Professional Dark/Navy Theme**: Designed specifically for telecom operations centers with clean interactive buttons.
- **Hidden API Configuration**: API credentials and model settings are securely tucked away in the collapsible sidebar.

---

## 🚀 Deployment Guide (GitHub → Streamlit Cloud)
*No VS Code, Terminal, or Google Colab required!*

### Step 1: Prepare Your GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click the **`+`** icon in the top right corner and select **New repository**.
3. Name your repository (e.g., `telecom-complaint-rag`).
4. Set it to **Public** (required for free Streamlit Community Cloud deployment).
5. Check **Add a README file** and click **Create repository**.

### Step 2: Upload Project Files
1. Inside your new repository, click **Add file** → **Upload files**.
2. Drag and drop the following four files into the browser:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Click **Commit changes**.

### Step 3: Deploy on Streamlit Cloud
1. Go to [Streamlit Cloud](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Fill in your deployment settings:
   - **Repository:** Select your `telecom-complaint-rag` repository.
   - **Branch:** `main` (or `master`).
   - **Main file path:** `app.py`.
4. *(Optional)* Expand **Advanced settings** to add your Groq API key securely as a secret:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"