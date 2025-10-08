from unsloth import FastLanguageModel
import torch
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel  # FastVisionModel for LLMs
from unsloth.chat_templates import get_chat_template
import torch


max_seq_length = 8192  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/Phi-4",  # Phi-4 2x faster!
    "unsloth/Phi-4-unsloth-bnb-4bit",  # Phi-4 Unsloth Dynamic 4-bit Quant
    "unsloth/gemma-2-9b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # Qwen 2.5 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]  # More models at https://docs.unsloth.ai/get-started/all-our-models

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
)

dataset = load_dataset('json', data_files='ranked_generations.json')
dataset2 = load_dataset("mlabonne/orca-agentinstruct-1M-v1-cleaned", split='train[:250]')

print(len(dataset['train']))
print(len(dataset2))

text_data = {'text': []}

for conversation in dataset['train']:
    turn = ''
    for key, example in conversation.items():
        if example:
            system_text = example['instruction']
            input_text = example['input']
            output_text = example['output']
            text_format = f"<|im_start|>system<|im_sep|>{system_text}<|im_end|><|im_start|>user<|im_sep|>{input_text}<|im_end|><|im_start|>assistant<|im_sep|>{output_text}<|im_end|>"
            turn += text_format
    text_data['text'].append(turn)

for example in dataset2:
    system_text = 'You are a language model trained to help user. Now your role as an assistant is to try to solve the following question.'
    turn = ''
    for message in example['messages']:
        if message['role'] == 'user':
            input_text = message['content']
            text_format = f"<|im_start|>system<|im_sep|>{system_text}<|im_end|><|im_start|>user<|im_sep|>{input_text}<|im_end|>"
        elif message['role'] == 'assistant':
            output_text = message['content']
            text_format = f"<|im_start|>assistant<|im_sep|>{output_text}<|im_end|>"
        turn += text_format
    text_data['text'].append(turn)

train_dataset = Dataset.from_dict(text_data)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=90,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_steps=5,
        save_total_limit=10,
        report_to="none",
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("Phi-4-Ads-8192-New")
tokenizer.save_pretrained("Phi-4-Ads-8192-New")
