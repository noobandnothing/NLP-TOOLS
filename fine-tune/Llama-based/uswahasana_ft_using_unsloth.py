# -*- coding: utf-8 -*-
"""UswaHasana_FT (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_hwUq6u8voHmmbFSH1EER4HEvbls3Fqw
"""

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
max_seq_length = 4096
dtype = None
load_in_4bit = True

fourbit_models = [
    "unsloth/llama-3-8b-bnb-4bit",
] #

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

alpaca_prompt = """فيما يلي تعليمات تصف مهمة. اكتب استجابة تكمل الطلب بشكل مناسب.

### التعليمات:
{}

### الاستجابة:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs       = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import Dataset
import pandas as pd
df = pd.read_csv('dataset.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 36,
        num_train_epochs=1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("Uswa_model")
tokenizer.save_pretrained("Uswa_model")
if False: model.push_to_hub("noobandnothing/Uswa_model", token = "...")
if False: tokenizer.push_to_hub("noobandnothing/Uswa_model", token = "...")

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to q4_k_m GGUF
# Save to q8_0 GGUF

if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q8_0")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

cd model

rm -rf unsloth.F16.gguf

mv /content/model/unsloth.Q4_K_M.gguf /content/model/uswa.Q4_K_M.gguf

from google.colab import drive
drive.mount('/content/drive')

cp /content/model/uswa.Q4_K_M.gguf /content/drive/MyDrive/model/

drive.flush_and_unmount()