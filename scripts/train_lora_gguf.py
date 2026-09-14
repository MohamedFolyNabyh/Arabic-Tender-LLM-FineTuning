# 1. تثبيت المكتبات السريعة
!pip install --force-reinstall --no-deps --no-cache-dir "torchvision==0.27.1"
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "triton>=3.0.0" bitsandbytes datasets trl peft

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 2. تحميل النموذج بضغطة 4-bit على Colab GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct", # أو Qwen2.5-7B-Instruct
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. إعداد LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 4. تحميل البيانات والتدريب
dataset = load_dataset("json", data_files="/content/train_qa_dataset (3).jsonl", split="train")

def format_prompts(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False
        )
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Changed from "messages" to "text"
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # اضبط عدد الخطوات حسب حجم بياناتك
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "outputs",
    ),
)
trainer.train()

# 5. تصدير النموذج المدمج كملف GGUF جاهز مباشرة!
model.save_pretrained_gguf("qwen_tender_model", tokenizer, quantization_method = "q4_k_m")