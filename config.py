import os

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_AUTH = os.environ.get('HF_AUTH')

# File Paths
ARGS = {
    'prompt_fp': '/content/drive/MyDrive/rel_detailed_refrence.txt',
    'save_fp': '/content/drive/MyDrive/llm_rel_detailed_refrence_Llama3.json',
    'summeval_fp': '/content/drive/MyDrive/summeval.json',
}

# Generation Parameters
GENERATION_CONFIG = {
    'max_new_tokens': 128,  # Reduce tokens if you face OOM issues
    'do_sample': True,
    'temperature': 0.1,
    'top_p': 0.9,
}