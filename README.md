# Telecom Customer Complaint Handling Assistant (RAG)

A Streamlit app that lets a support agent upload a searchable PDF complaint-handling document and ask questions in natural language. It uses:

- **PyMuPDF** to extract PDF text
- **Sentence Transformers** (`all-MiniLM-L6-v2`) to create embeddings
- **FAISS** to retrieve relevant document chunks
- **Groq** with `openai/gpt-oss-20b` to answer from retrieved excerpts
- **Streamlit** for the web interface

The assistant is designed to answer only from the uploaded document. It should say when the document does not specify an answer and cite source page numbers where possible.

## Files

- `app.py` — application
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes local secrets and generated files

## Important notes

- Upload a **searchable PDF**. Scanned PDFs may need OCR before upload.
- The first load can take a few minutes because the embedding model must download.
- The app processes the PDF in the Streamlit app session and builds the FAISS index in memory. The index is not saved as a persistent database by this code.
- When you ask a question, the app sends the question and retrieved PDF excerpts to Groq for answer generation. Follow your organization's information-security rules and do not upload customer personal data or confidential material unless approved.
- Do not place API keys in `app.py`, commit them to GitHub, or share them publicly.
- RAG reduces unsupported answers but does not guarantee correctness. Verify operational steps against approved procedures.
- `openai/gpt-oss-20b` is the configured Groq model ID. If Groq reports that the model is unavailable to your account, check the current model IDs in Groq Console and update `GROQ_MODEL` in `app.py`.

## Deploy with GitHub and Streamlit Community Cloud (no terminal, VS Code, or Colab)

### 1. Get a Groq API key

1. Open [Groq Console](https://console.groq.com/).
2. Sign in and create an API key.
3. Copy it somewhere private temporarily. Do not put it in a GitHub file.

### 2. Create a GitHub repository in the browser

1. Open [GitHub](https://github.com/) and sign in.
2. Click **+** → **New repository**.
3. Name it `telecom-complaint-rag-assistant`.
4. Choose **Private** if the code should not be public.
5. Click **Create repository**.

### 3. Upload the project files using GitHub's website

1. Open the new repository.
2. Select **Add file** → **Upload files**.
3. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore` from this project.
4. Click **Commit changes**.

You do not need to upload your complaint-handling PDF to the repository. Upload it through the running app when you use the assistant.

### 4. Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
2. Connect/authorize your GitHub account if prompted.
3. Click **Create app** (or **New app**).
4. Select the repository `telecom-complaint-rag-assistant`, branch `main`, and main file path `app.py`.
5. Open **Advanced settings** if shown. Choose a supported Python version if requested; Python 3.11 is a sensible default for dependency compatibility.
6. Click **Deploy** and wait for the app to build. The first build may take several minutes.

### 5. Add the Groq key securely

1. Open your app's management page on Streamlit Community Cloud.
2. Open **Settings** → **Secrets**.
3. Add this line, replacing the placeholder with your real key:

   ```toml
   GROQ_API_KEY = "paste_your_real_key_here"
   ```

4. Save the secrets and reboot/redeploy the app if prompted.

Never add your real key to `README.md`, `app.py`, screenshots, or GitHub commits.

### 6. Test the app

1. Open the deployed app URL.
2. Upload a searchable PDF that contains your approved complaint-handling procedures.
3. Ask a narrow question such as “Which applications are involved in SIM modification?”
4. Expand **Retrieved document excerpts** to inspect the pages/chunks used for the answer.
5. Confirm that the answer matches the source document.

## Troubleshooting

- **Build fails:** Open the app's **Manage app** / logs view in Streamlit Cloud and read the first dependency error. Confirm the repository contains `app.py` and `requirements.txt` at its root.
- **Groq API key missing:** Check **Settings → Secrets** and the exact key name `GROQ_API_KEY`.
- **Model not found / unavailable:** Verify `openai/gpt-oss-20b` is currently available to your Groq account and update `GROQ_MODEL` in `app.py` if needed.
- **PDF has no text:** It may be scanned. Run OCR using an organization-approved tool, save a searchable PDF, and upload it again.
- **App is slow on first use:** Model downloads and embedding-index creation take time, especially on a free hosting instance.
- **Memory or dependency issue:** Try a smaller PDF first. Very large PDFs may exceed free instance memory or take longer to index.

## Privacy and security

This starter app does not write the uploaded PDF or FAISS index to a persistent database. However, the hosting provider processes the app session, and retrieved excerpts plus the question are sent to Groq. Use only documents you are authorized to process. A private GitHub repository protects source-code visibility but does not make the deployed app itself private; configure app access and sharing appropriately in Streamlit Cloud.
