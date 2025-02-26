import os
import json
import tqdm
import transformers
import torch
from google.colab import drive
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from config import MODEL_ID, HF_AUTH, ARGS, GENERATION_CONFIG



# Load tokenizer
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, token=HF_AUTH)

# Load model
def load_model():
    bits_and_bytes_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_config = AutoConfig.from_pretrained(MODEL_ID, token=HF_AUTH)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bits_and_bytes_config,
        device_map='auto',
        token=HF_AUTH
    )
    model.eval()
    print("Model loaded successfully")
    return model

# Process data
def process_data(summeval, prompt, tokenizer, model, save_fp):
    processed_data = []
    
    for instance in tqdm.tqdm(summeval):
        source = instance['reference']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Reference}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt

        messages = [{"role": "system", "content": cur_prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|eot_id|>")

        outputs = model.generate(
            input_ids,
            max_new_tokens=GENERATION_CONFIG['max_new_tokens'],
            eos_token_id=eos_token_id,
            do_sample=GENERATION_CONFIG['do_sample'],
            temperature=GENERATION_CONFIG['temperature'],
            top_p=GENERATION_CONFIG['top_p'],
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        instance['all_responses'] = tokenizer.decode(response, skip_special_tokens=True)
        processed_data.append(instance)
    
    with open(save_fp, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)
    print(f'Results saved to: {save_fp}')

# Main function
def main():
    tokenizer = load_tokenizer()
    model = load_model()
    
    with open(ARGS['summeval_fp'], encoding='utf-8') as f:
        summeval = json.load(f)
    
    with open(ARGS['prompt_fp'], encoding='utf-8') as f:
        prompt = f.read()
    
    process_data(summeval, prompt, tokenizer, model, ARGS['save_fp'])

if __name__ == "__main__":
    main()
