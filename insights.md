# 📊 Insights Document

A summary of key findings, challenges, and areas for improvement in our text summarization pipeline.

---

## 🔍 Summary of Findings

### ❓ What Kind of Text is Hard to Summarize?

- 🧩 **Incoherent or fragmented text**: Lacks structure or flow, making it difficult for the model to generate coherent summaries.
- 🪶 **Overly short texts**: Minimal context leads to vague or uninformative summaries.
- 🧪 **Highly domain-specific language**: Technical terms are often misinterpreted or omitted if the model wasn't trained on similar data.

---

### 🔁 Recurring Errors Observed

- 🔂 **Redundancy**: Some summaries repeat phrases unnecessarily.
- ⚠️ **Loss of specificity**: Models tend to generalize, often dropping proper nouns, figures, or key details.
- 🧠 **Hallucinations**: Occasionally, summaries include fabricated information not present in the input text.

---

### 💬 Handling Dialogue-style Data

- 🧵 **Challenges**: Out-of-the-box models (like BART trained on news data) often:

  - Confuse speaker turns
  - Produce linear monologues
  - Miss conversational nuance

- 🔧 **Improvement with fine-tuning**: Training on conversational datasets (e.g., SAMSum) improves coherence and speaker handling significantly.

---

## 🧹 Preprocessing & Post-processing

### 🔄 Preprocessing Steps

- Removed newline characters, special symbols, and normalized whitespace.
- Truncated input to the model’s maximum token limit (e.g., 1024 tokens for BART).

### 🎯 Post-processing Steps

- Cleaned decoded summaries (e.g., removed trailing spaces, fixed token artifacts).
- Normalized ROUGE scores for consistent comparison across varying input lengths.

---

## 🚧 Limitations

- No fine-tuning done — uses a pre-trained model.
- 🧠 **Model generalization**: Without task-specific fine-tuning, pretrained models struggle with conversational or noisy data.
- ⚖️ **Dataset bias**: Some datasets bias summaries toward journalistic or formal tones.
- Evaluation is purely ROUGE-based — no human evaluation.



---

## 💡 Future Exploration Ideas

- 🛠️ **Fine-tuning**: Train on domain-specific datasets (e.g., technical docs, dialogues) to improve output quality.
- 📚 **Multi-document summarization**: Extend the pipeline to summarize information from multiple sources at once.
- 🔍 **RAG-style approaches**: Combine retrieval with generation to improve factual consistency and reduce hallucinations.
- 🖼️ **Interactive UI**: Build a real-time summarization demo using Gradio or Streamlit.

---

✨ *These insights aim to guide future improvements in summarization workflows and model selection.*
