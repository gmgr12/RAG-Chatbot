from transformers import AutoModel, AutoTokenizer

# Specify model names
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def download_model(model_name):
    print(f"Downloading {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model {model_name} downloaded successfully.")

# Download both models
download_model(EMBEDDING_MODEL_NAME)
download_model(READER_MODEL_NAME)
