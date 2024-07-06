#!git clone https://github.com/ggerganov/llama.cpp
#!cd llama.cpp && LLAMA_CUBLAS=1 make && pip install -r requirements/requirements-convert-hf-to-gguf.txt
#from huggingface_hub import snapshot_download
#model_name = "sambanovasystems/SambaLingo-Arabic-Chat-70B"
#methods = ['q4_k_m','q8_0']
methods = ['q8_0']
base_model = "/home/noob/AceGPT-7B/mod/blobs"
quantized_path = "./quantized_model/"
#snapshot_download(repo_id=model_name, local_dir=base_model , local_dir_use_symlinks=False)
original_model = quantized_path+'/FP16.gguf'

!mkdir ./quantized_model/

!python llama.cpp/convert-hf-to-gguf.py /home/noob/AceGPT-7B/mod/blobs --outtype f16 --outfile ./quantized_model/FP16.gguf

import os

for m in methods:
    qtype = f"{quantized_path}/{m.upper()}.gguf"
    os.system("./llama.cpp/llama-quantize "+quantized_path+"/FP16.gguf "+qtype+" "+m)

#! ./llama.cpp/main -m ./quantized_model/Q4_K_M.gguf -n 90 --repeat_penalty 1.0 --color -i -r "User:" -f llama.cpp/prompts/chat-with-bob.txt
