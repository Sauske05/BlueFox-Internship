from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load tokenizer and model
# Key learning: AutoTokenizer and AutoModelForCausalLM allow dynamic loading of pre-trained models.
model_name = "distilgpt2"  # Using a lightweight model for faster experimentation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize input text
# Note: Tokenizers convert text to token IDs that the model understands.
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")  # Return PyTorch tensors

# Generate text using the model directly
# Learning: Model.generate() handles decoding strategies like greedy or sampling.
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text (Direct):", generated_text)

# Use a pipeline for simplified inference
# Key learning: Pipelines abstract away tokenization and generation, making tasks easier.
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipeline_output = text_generator(input_text, max_length=50, do_sample=True, top_k=50)
print("Generated Text (Pipeline):", pipeline_output[0]["generated_text"])