import os
import shutil
import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def run_training_pipeline(
    data_path: str = "data/processed/train_qa_dataset.jsonl",
    hf_repo_id: str = "your-username/qwen2.5-7b-saudi-tender-gguf",  # استبدل باسم حسابك واسم المستودع
    hf_token: str = None,  # يمكنك تمرير التوكن هنا أو ضبطه في متغيرات البيئة HF_TOKEN
    max_steps: int = 60
):
    # 1. إعداد وتسجيل الدخول إلى Hugging Face
    token = hf_token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("⚠️ لم يتم العثور على HF_TOKEN، تأكد من تسجيل الدخول أو إدخال التوكن لضمان نجاح الرفع.")

    # 2. إنشاء المجلدات اللازمة
    os.makedirs("outputs", exist_ok=True)

    # 3. تحميل النموذج بضغط 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # 4. إعداد LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    # 5. تحميل البيانات بتنسيق Chat Template
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_prompts(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    # 6. تهيئة المدرب
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="outputs",
        ),
    )
    
    # 7. بدء التدريب
    trainer.train()

    # 8. تنظيف الذاكرة المؤقتة ومجلد outputs لتوفير المساحة
    del trainer
    torch.cuda.empty_cache()
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")

    # 9. التكميم والرفع المباشر إلى Hugging Face Hub
    print(f"🚀 جاري تحويل النموذج ورفعه مباشرة إلى: {hf_repo_id}")
    model.push_to_hub_gguf(
        file_name=hf_repo_id,
        tokenizer=tokenizer,
        quantization_method="q4_k_m",
        token=token
    )
    print("✅ تم رفع النموذج بنجاح إلى Hugging Face!")


if __name__ == "__main__":
    # يمكن إعداد متغير البيئة HF_TOKEN في الترمينال أو تمريره مباشرة هنا
    run_training_pipeline(
        hf_repo_id="MohamedFoly/qwen2.5-saudi-tender-gguf",  # اسم المستودع المستهدف
        max_steps=60
    )
