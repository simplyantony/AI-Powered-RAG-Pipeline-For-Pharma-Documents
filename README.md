# AI-Powered RAG Pipeline for Pharmaceutical Documents

A notebook-based retrieval-augmented generation (RAG) pipeline for asking questions about pharmaceutical PDF documents. The project combines PDF text extraction, OCR for scanned documents, semantic embeddings, FAISS similarity search, and the Phi-2 language model behind a Gradio chat interface.

> **Important:** This project is intended for research, experimentation, and document-navigation support. It is not a medical device, regulatory system, clinical decision-support tool, or substitute for review by a qualified professional.

## Features

- **PDF ingestion:** Upload one or more PDF documents through the Gradio interface.
- **Digital and scanned PDF support:** Detects scanned PDFs and uses Tesseract OCR with Poppler when text extraction is insufficient.
- **Metadata-aware chunking:** Splits extracted text into overlapping chunks while preserving source filename and page number.
- **Semantic retrieval:** Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2` and searches them with a normalized FAISS inner-product index.
- **Grounded generation:** Uses `microsoft/phi-2` to answer questions using only the retrieved document context.
- **Source citations:** Answers include source and page references, along with retrieval confidence information.
- **Interactive UI:** Provides a Gradio upload, processing, chat, and clear-history workflow.
- **Evaluation support:** The notebook includes an evaluation section for retrieval and end-to-end metrics such as Recall@K, Precision@K, MRR, answer matching, citation checks, and latency measurements.

## Repository Contents

| File | Description |
| --- | --- |
| [`FinalRAGPipeline (1).ipynb`](./FinalRAGPipeline%20(1).ipynb) | Main Google Colab/Jupyter Notebook containing the complete RAG pipeline and Gradio application. |
| [`RAG Pipeline Evaluation.pdf`](./RAG%20Pipeline%20Evaluation.pdf) | Evaluation document associated with the pipeline. |
| [`LICENSE`](./LICENSE) | MIT License for this repository. |

## How the Pipeline Works

1. **Upload documents** through the Gradio UI. The application accepts multiple `.pdf` files.
2. **Classify each PDF** as digital or scanned by checking whether text can be extracted from the first few pages.
3. **Extract page text** with `pypdf` for digital PDFs or render pages and run Tesseract OCR for scanned PDFs.
4. **Normalize and chunk text** into 300-character chunks with a 50-character overlap. Chunks shorter than 40 characters are discarded.
5. **Create embeddings** with `all-MiniLM-L6-v2` and add normalized vectors to a FAISS `IndexFlatIP` index.
6. **Retrieve the top five chunks** for each user question using cosine-similarity-equivalent inner-product search.
7. **Build a constrained prompt** containing the retrieved chunks, source names, and page numbers.
8. **Generate an answer** with Phi-2, configured to answer only from the supplied context and to report when the answer is not present.
9. **Display citations and confidence** in the chat response so users can inspect the supporting document locations.

## Usage

### Option 1: Run in Google Colab

The notebook is configured for Google Colab and requests a T4 GPU. Colab is the recommended environment because the notebook installs system packages for OCR and is configured to load Phi-2 with GPU acceleration.

1. Open [`FinalRAGPipeline (1).ipynb`](./FinalRAGPipeline%20(1).ipynb) in Google Colab.
2. Select a runtime with GPU acceleration, preferably a T4 or better:
   - **Runtime → Change runtime type → T4 GPU**
3. Run the notebook cells from top to bottom.
4. Allow the setup cell to install the Python and system dependencies:

   ```bash
   pip install -q gradio
   pip install -q transformers torch accelerate einops bitsandbytes
   pip install -q sentence-transformers faiss-cpu
   pip install -q pypdf pillow pytesseract pdf2image
   apt-get install -q tesseract-ocr poppler-utils
   ```

5. Wait for the embedding model and language model to load. The notebook loads them lazily, so the first processing or question may take longer.
6. When the Gradio interface appears, upload one or more pharmaceutical PDF documents.
7. Click **Process Documents** and wait for the status box to confirm that the files have been indexed.
8. Enter a question about the uploaded documents, or press **Send**. You can also submit a question by pressing Enter.
9. Review the generated answer, source/page citations, confidence values, and retrieval timing.
10. Uploading a new set of files resets the in-memory document state and builds a new index.

The notebook launches Gradio with `share=True`, which creates a temporary public URL while the notebook runtime is active. Treat that URL and any uploaded documents as sensitive. Do not upload confidential or regulated documents to a public or unapproved environment.

### Option 2: Run locally with Jupyter

A local Jupyter environment can be used if the required Python packages and system dependencies are installed. GPU acceleration is strongly recommended for Phi-2 inference.

1. Create and activate a Python environment.
2. Install the Python packages listed in the notebook setup cell:

   ```bash
   pip install gradio transformers torch accelerate einops bitsandbytes \
     sentence-transformers faiss-cpu pypdf pillow pytesseract pdf2image
   ```

3. Install the OCR and PDF-rendering system tools:

   - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr poppler-utils`
   - **macOS:** install Tesseract and Poppler with Homebrew.
   - **Windows:** install Tesseract and Poppler separately and ensure their executables are available on `PATH`.

4. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook
   ```

5. Run the cells in order, select PDF files in the Gradio interface, process them, and ask questions.

The notebook uses Hugging Face model identifiers and may download model files on first use. Network access is therefore required for the initial model download unless the models have already been cached locally.

## Configuration

The main configuration values are defined near the beginning of the notebook:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model used for document and query vectors. |
| `LLM_MODEL_NAME` | `microsoft/phi-2` | Causal language model used to generate answers. |
| `CHUNK_SIZE` | `300` characters | Maximum size of each text chunk. |
| `CHUNK_OVERLAP` | `50` characters | Overlap between adjacent chunks. |
| `TOP_K` | `5` | Number of chunks retrieved for each question. |
| `MAX_NEW_TOKENS` | `400` | Maximum number of generated tokens. |
| `PHI2_REP_PENALTY` | `1.2` | Repetition penalty used during generation. |
| `MIN_CHUNK_LEN` | `40` characters | Minimum chunk length retained for indexing. |

These settings can be adjusted for different document lengths, retrieval behavior, available memory, and response latency. Changes should be evaluated against a representative test set before being used for important workflows.

## Evaluation

The notebook includes an evaluation section that should be run after the main pipeline cells have executed and after at least one PDF has been processed. It evaluates retrieval, answer quality, citation behavior, and performance.

The evaluation workflow is:

1. Run the setup, document-processing, retrieval, generation, and UI cells.
2. Upload and process at least one PDF.
3. Run the evaluation cell.
4. Review the printed report.
5. Inspect the generated `evaluation_report.json` file.

The included test queries use expected terms and answer terms. For meaningful results, update the test set with questions and expected evidence that reflect the documents you intend to use.

## Limitations and Responsible Use

- **No guarantee of correctness:** Retrieval-augmented generation can produce incomplete, misleading, or incorrect answers.
- **OCR quality varies:** Scanned documents, tables, formulas, multi-column layouts, and low-quality images may be extracted incorrectly.
- **Citation limitations:** A cited page indicates retrieved context, not proof that the generated statement is correct.
- **In-memory state:** The notebook keeps the current chunks and FAISS index in process memory. Restarting the runtime clears the state.
- **Model limitations:** Phi-2 and the embedding model may not be optimized for every pharmaceutical terminology, document type, or language.
- **No access controls:** The notebook does not implement authentication, authorization, audit logging, encryption, retention policies, or enterprise document governance.
- **Sensitive information:** Do not upload protected health information, confidential research, proprietary manufacturing information, personal data, or regulated records unless the execution environment and handling process have been reviewed and approved for that data.
- **Regulatory decisions:** Do not use outputs as the sole basis for medical, safety, quality, manufacturing, compliance, or regulatory decisions.

## Licensing

This repository is released under the **MIT License**, with copyright attributed to **Anthony Bryant, 2026**. The license permits you to use, copy, modify, merge, publish, distribute, sublicense, and sell the repository software, subject to the following main conditions:

- Preserve the copyright notice and MIT permission notice in copies or substantial portions of the software.
- The software is provided **“as is,” without warranty**. The author is not liable for claims, damages, or other liability arising from its use.

The MIT License applies to the original material in this repository. It does **not automatically grant rights to third-party components, model weights, datasets, uploaded documents, or generated content**. Before redistributing or deploying the complete pipeline, review the applicable terms for:

- The `microsoft/phi-2` model and its model-specific license or usage restrictions.
- The `sentence-transformers/all-MiniLM-L6-v2` model and its associated license.
- Python packages and system tools such as PyTorch, Transformers, FAISS, Gradio, Tesseract, Poppler, and their dependencies.
- Any pharmaceutical, proprietary, copyrighted, or otherwise restricted documents processed by the pipeline.

For the authoritative repository license text, see [`LICENSE`](./LICENSE).

## Acknowledgments

This project was developed as a Pfizer externship project and uses open-source libraries from the Python, Hugging Face, FAISS, Gradio, Tesseract, and Poppler ecosystems.
