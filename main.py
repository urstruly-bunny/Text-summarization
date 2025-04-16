from transformers import pipeline

def main():
    # Load the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Input text
    text = """
    Rita – Hey Tina? Is it you?

Tina – Oh Rita! How are you? It’s been a long time.

Rita – I am fine, what about you? Yes, we last met during the board exams.

Tina – I’m good too.

Rita – What are you doing now?

Tina – Well, I have started my undergraduate studies in English Honours at St. Xaviers College in Mumbai.

Rita – Wow! You finally got to study the subject you loved the most in school.

Tina – True. What about you Rita? Wasn’t History your favourite subject?

Rita – You guessed it right. I took up History Honours in Lady Shri Ram College for Women in Delhi.

Tina – That’s nice. I am so happy for you.

Rita – I am happy for you too. Let’s meet up again soon.

Tina – Yes, sure! We have a lot to catch up on.

Rita – Bye for now. I have to pick up my sister from tuition. Take care.

Tina – Bye, will see you soon.
    """

    # Generate summary
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    print("Original Text:\n", text)
    print("\nSummary:\n", summary[0]['summary_text'])

if __name__ == "__main__":
    main()
