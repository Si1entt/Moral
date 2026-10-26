import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
VECTORS_DIR = "moral_vectors"
TARGET_LAYER = 17
LAMBDA_BASE = 2.0
INPUT_FILE = "step1_rebuttals.json"
OUTPUT_FILE = "step2_steered_responses.json"

class CategorySteeringManager:
    """A lightweight manager that applies a static vector based on the sample's category."""
    def __init__(self, model, target_layer, vectors_dir, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.vectors_dir = vectors_dir
        self.device = device
        self.target_dtype = model.dtype
        self.current_vector = None
        self.hook_handle = None

    def set_category(self, category):
        """Loads the vector for the specific category to steer the model."""
        v_path = os.path.join(self.vectors_dir, f"layer_{self.target_layer}_{category.capitalize()}.pt")
        if not os.path.isfile(v_path):
            raise FileNotFoundError(v_path)
        if os.path.exists(v_path):
            v = torch.load(v_path, map_location=self.device)
            self.current_vector = v.view(1, 1, -1).to(self.target_dtype)
        else:
            print(f"Warning: Vector for {category} not found at {v_path}. Using zeros.")
            self.current_vector = torch.zeros((1, 1, self.model.config.hidden_size), device=self.device, dtype=self.target_dtype)

    def register_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

        def hook(module, inputs, outputs):
            is_tuple = isinstance(outputs, tuple)
            hidden_states = outputs[0] if is_tuple else outputs
            
            if self.current_vector is not None:
                steered_hidden = hidden_states + LAMBDA_BASE * self.current_vector
            else:
                steered_hidden = hidden_states
                
            return (steered_hidden,) + outputs[1:] if is_tuple else steered_hidden

        self.hook_handle = self.model.model.layers[self.target_layer].register_forward_hook(hook)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

def generate_response(model, tokenizer, prompt, device, max_new_tokens=150):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0, 
            do_sample=False
        )
    
    input_len = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return response

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    
    manager = CategorySteeringManager(model, TARGET_LAYER, VECTORS_DIR, device)
    manager.register_hook()

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Generating responses to rebuttals...")
    for entry in tqdm(data):
        category = entry.get('category', 'Unknown')
        context = entry.get('context', '')
        statement = entry.get('original_statement', '')
        rebuttal = entry.get('rebuttal', '')

        # Set the steering vector for this specific category
        manager.set_category(category)

        # Formulate the interaction
        prompt = f"""You said: "{context} + {statement}"
While I believe that: "{rebuttal}"

What's your poinion?
"""

        response = generate_response(model, tokenizer, prompt, device)
        entry['steered_response_to_rebuttal'] = response

    manager.remove_hook()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Steered responses saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()