from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt")
    # Generate response
    outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response