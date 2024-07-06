#!pip install torch datasets trl accelerate peft bitsandbytes transformers ipywidgets 

import pandas as pd
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("arbml/CIDAR", split='train')

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame({
    'instruction': dataset['instruction'],
    'output': dataset['output']
})

# Displaying the first few rows of the DataFrame
df.head()


# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

df.head()

"""Split into train and test sets."""

# Split the data into train and test sets, with 90% in the train set
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save the dataframes to .jsonl files
train_df.to_json('train.jsonl', orient='records', lines=True)
test_df.to_json('test.jsonl', orient='records', lines=True)

"""# Install & import necessary libraries"""

import os
import torch
from datasets import load_dataset
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel
from trl import SFTConfig , SFTTrainer

#HAS_BFLOAT16 = torch.cuda.is_bf16_supported() false
HAS_BFLOAT16 = False

model_name = '/home/noob/AceGPT-7B-FP16/'
dataset_name = 'train.jsonl'
new_model = "AceGPT-7B-FP16-custom"

"""# Define Training Params"""

lora_r = 16
lora_alpha = 16
lora_dropout = 0.1
#use_4bit = False
#bnb_4bit_compute_dtype = "bfloat16"
#bnb_4bit_quant_type = "nf4"
#use_nested_quant = False
output_dir = r"Fine-tuned-shit"
num_train_epochs = 3
fp16 = True
#bf16 = True
# per_device_train_batch_size = 2
# per_device_eval_batch_size = 2
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
#device_map = {"": 0}

"""#Load Dataset




"""

# Load datasets
train_dataset = load_dataset('json', data_files='train.jsonl', split="train")
valid_dataset = load_dataset('json', data_files='test.jsonl', split="train")

"""#Preprocess Dataset

"""

# Preprocess datasets
train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'### Instruction:\n' + instruction + '\n\n### Response:\n' + output + '</s>' for instruction, output in zip(examples['instruction'], examples['output'])]}, batched=True)
valid_dataset_mapped = valid_dataset.map(lambda examples: {'text': [f'### Instruction:\n' + instruction + '\n\n### Response:\n' + output + '</s>' for instruction, output in zip(examples['instruction'], examples['output'])]}, batched=True)

"""#Train"""

#compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=use_4bit,
#    bnb_4bit_quant_type=bnb_4bit_quant_type,
#    bnb_4bit_compute_dtype=compute_dtype,
#    bnb_4bit_use_double_quant=use_nested_quant,
#)

#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_8bit=True,
    #quantization_config=bnb_config,
    #device_map=device_map
)

model.config.use_cache = False
model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

per_device_train_batch_size  = 20 
per_device_eval_batch_size  =  20
# Set training parameters
training_arguments = TrainingArguments(
    auto_find_batch_size=True,
    #per_device_train_batch_size=20,
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    #bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    
    #report_to="all",
    #use_cpu=True,

    #evaluation_strategy="steps",
    #eval_steps=5  # Evaluate every 20 steps
)
training_arguments = SFTConfig(training_arguments)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)




trainer.train()

trainer.model.save_pretrained(new_model)

"""# Test Trained model"""

#logging.set_verbosity(logging.CRITICAL)
#prompt = f"### Instruction:\nwho is the prophet Mohammed PBUH\n\n### Response:\n" # replace the command here with something relevant to your task
#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
#result = pipe(prompt)
print(result[0]['generated_text'])

"""#Run Inference"""

#from transformers import pipeline

#prompt = f'### Instruction:\nأكتب قصيدة فكاهية عن شخص خسر أمواله في الاسهم.\n\n### Response:\n' # replace the command here with something relevant to your task
#num_new_tokens = 700  # change to the number of new tokens you want to generate

# Count the number of tokens in the prompt
#num_prompt_tokens = len(tokenizer(prompt)['input_ids'])

# Calculate the maximum length for the generation
#max_length = num_prompt_tokens + num_new_tokens

#gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length,repetetion_panelty=1.2)
#result = gen(prompt)
print(result[0]['generated_text'].replace(prompt, ''))

"""#Merge the model and store"""

#device_map = {0}
#model_name = 'FreedomIntelligence/AceGPT-7B'
#model_path = r"E:\agft"  # change to your preferred path

# Reload model in FP16 and merge it with LoRA weights
#base_model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    low_cpu_mem_usage=True,
#    return_dict=True,
#    torch_dtype=torch.float16,
#    device_map=device_map,
#)

#model = PeftModel.from_pretrained(base_model, new_model)
#model = model.merge_and_unload()

# Reload tokenizer to save it
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"

# Save the merged model
#model.save_pretrained(model_path)
#tokenizer.save_pretrained(model_path)
