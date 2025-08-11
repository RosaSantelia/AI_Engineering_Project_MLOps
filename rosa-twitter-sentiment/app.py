import gradio as gr
from transformers import pipeline

# Carica il modello dal Model Hub
# Sostituisci con il tuo repo modello su HF, es. "RosaSantelia/sentiment-analysis-roberta"
model_name = "RosaSantelia/sentiment-analysis-roberta"
classifier = pipeline("sentiment-analysis", model=model_name)

def predict_sentiment(text):
    if not text.strip():
        return "⚠️ Inserisci del testo per analizzarlo."
    result = classifier(text)[0]
    label = result["label"]
    score = round(result["score"], 4)
    return f"Sentiment: **{label}** (confidenza: {score})"

# Interfaccia Gradio
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Inserisci un testo o un tweet"),
    outputs=gr.Markdown(label="Risultato"),
    title="Twitter Sentiment Analysis",
    description="Analizza il sentiment di un testo usando un modello Hugging Face"
)

if __name__ == "__main__":
    demo.launch()
