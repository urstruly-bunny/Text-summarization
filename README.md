# 📝 Text Summarization Pipeline

A modular text summarization project using Hugging Face's Transformer models. It processes dialogue-style text data, generates concise summaries, and evaluates them using ROUGE scores.

---

## 📦 Features

- 🗣️ Summarizes **dialogue-style conversations** into short, coherent summaries  
- 🤗 Uses **Hugging Face Transformers** (`BartForConditionalGeneration`)  
- 🧩 Modular codebase with clearly separated components:
  - Data loading & preprocessing
  - Tokenization
  - Model inference
  - Evaluation (ROUGE scores)
- 🔁 Easy to extend with other models and datasets

---

## ⚙️ Setup Instructions

### 1. 🚀 Clone the Repository

```bash
git clone https://github.com/your-username/text-summarization-project.git
cd text-summarization-project
```

### 2. 🧪 Set Up Virtual Environment

**For Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. ▶️ Run the Pipeline

```bash
python main.py
```

This will:

- Load the **SAMSum** dataset  
- Tokenize and summarize the dialogues  
- Compute **ROUGE** scores  
- Print a few sample summaries along with evaluation metrics

---

## 🧱 Design & Architecture

- **🔧 Modularity**: Pipeline is split into separate components for preprocessing, summarization, and evaluation — making debugging and experimentation easier.  
- **🧠 Model Choice**: Utilizes the pre-trained `facebook/bart-large-cnn` model for summarization, known for strong performance on summarization tasks.  
- **📏 Evaluation**: Uses ROUGE metrics (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`) to assess the quality of the generated summaries.  
- **📦 Reproducibility**: The `main.py` script wraps everything in an end-to-end flow — making it easy to run, test, or deploy the full pipeline.

---

## 💡 Future Improvements

- Add support for more datasets (e.g., CNN/DailyMail, XSum)
- Plug in different transformer models for comparison
- Build a simple web UI to interact with the summarizer

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo, create a new branch, and submit a PR.

---

## 📄 License

This project is licensed under the MIT License.
