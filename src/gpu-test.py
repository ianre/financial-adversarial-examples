import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_gpu_and_generate(prompt):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU is available and will be used.")
    else:
        print("No GPU detected, running on CPU.")
    
    # Load a lightweight model from Hugging Face for fast checking
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Encode the prompt and move it to the device (GPU or CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response
    output = model.generate(**inputs, max_length=50)
    
    # Decode and print the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Response: {response}")

# Example usage
print("torch.__version__", torch.__version__)  # PyTorch version
print("torch.version.cuda", torch.version.cuda)  # CUDA version compatible with this PyTorch
print("torch.cuda.is_available()", torch.cuda.is_available())  # Should return True if GPU is accessible
prompt = "What is the capital of France?"
responses = check_gpu_and_generate(prompt)
print(responses)

