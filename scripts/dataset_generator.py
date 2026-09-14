import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from json_repair import repair_json

from vllm import LLM, SamplingParams
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ==========================================
# 1. Schema للتحقق من المخرجات
# ==========================================
# class QAPair(BaseModel):
#     user_question: str = Field(description="سؤال واقعي ومباشر بناءً على النص")
#     assistant_answer: str = Field(description="إجابة دقيقة وشاملة مستخرجة من النص فقط")


# ==========================================
# 2. Main Dataset Engineering Pipeline (vLLM Enabled)
# ==========================================
class DatasetPipeline:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 50,
    ):
        print(f"[1/3] تحميل الـ Tokenizer: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"[2/3] تهيئة نموذج vLLM للسرعة القصوى ({model_id})...")
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,  # تقليص الاستهلاك لتجنب الـ OOM
            max_model_len=4096,          # تقييد الحد الأقصى للسياق لحماية VRAM كارت T4
            trust_remote_code=True,
            enforce_eager=True
        )

        print("[3/3] إعداد الـ Token-Based Splitter ومحرك Docling...")
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            separators=["\n\n", "\n", "؟", "!", ".", " ", ""]
        )

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True

            self.converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except Exception as e:
            self.converter = DocumentConverter()

        self.system_prompt = (
            "أنت مهندس بيانات متخصص في إعداد مجموعات البيانات لتدريب النماذج اللغوية.\n"
            "مهمتك: تحليل النص المرفق واستخراج أسئلة وإجابات متنوعة كأنها حوار طبيعي بين مستخدم ومساعد ذكي.\n"
            "القواعد:\n"
            "1. يجب أن تكون الإجابات معتمدة تماماً على النص المتاح فقط دون اختراع معلومات.\n"
            "2. احتفظ بتفاصيل الجداول والقوائم المذكورة في النص واشرحها بوضوح في الإجابة.\n"
            "3. اكتب بلغة عربية سليمة وواضحة."
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        pdf_path_obj = Path(pdf_path)
        print(f" 🔄 جاري تحويل {pdf_path_obj.name} باستخدام Docling...")

        try:
            result = self.converter.convert(str(pdf_path_obj))
            markdown_text = result.document.export_to_markdown()

            if markdown_text.strip():
                print(f"  ✅ تم استخراج النص بنجاح ({len(markdown_text)} حرف)")
                return markdown_text
        except Exception as e:
            print(f" ⚠️ فشل استخراج النص بـ Docling لـ {pdf_path_obj.name}: {e}")

        return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[\r\t\f\v]', ' ', text)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ ]+', ' ', text)
        return text.strip()

    def process_files_to_chunks(self, input_path_str: str) -> List[str]:
        all_chunks = []
        input_path = Path(input_path_str)

        files_to_process = [input_path] if input_path.is_file() else [f for f in input_path.rglob('*') if f.is_file()]

        print(f"\n مسح وقراءة المسار: {input_path_str}")
        for file_path in files_to_process:
            raw_text = ""
            if file_path.suffix.lower() == '.pdf':
                raw_text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.md', '.json']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
            else:
                continue

            cleaned = self.clean_text(raw_text)
            if not cleaned:
                print(f" ⚠️ لم يتم استخراج أي نص من {file_path.name}")
                continue

            chunks = self.text_splitter.split_text(cleaned)
            all_chunks.extend(chunks)
            print(f"  - تم معالجة {file_path.name}: تم إنتاج {len(chunks)} مقطع (Chunk).")

        print(f"\n إجمالي المقاطع (Chunks) الناتجة: {len(all_chunks)}")
        return all_chunks

    def run_pipeline(self, input_path: str, output_jsonl: str, pairs_per_chunk: int = 2):
        chunks = self.process_files_to_chunks(input_path)

        if not chunks:
            print("\n ⚠️ لم يتم العثور على مقاطع نصية معالجة.")
            return

        print(f"\n بدء إعداد الـ Prompts...")
        prompts = []
        for chunk in chunks:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"قم باستخراج {pairs_per_chunk} أزواج أسئلة وإجابات متكاملة بناءً على النص المرفق فقط:\n\n---\n{chunk}\n---\n\nأخرج الناتج بصيغة JSON فقط مطابقة للهيكل التالي:\n{{\"qa_pairs\": [{{\"user_question\": \"...\", \"assistant_answer\": \"...\"}}]}}"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9
        )

        print(f"\n بدء عملية توليد الأسئلة والإجابات على دفعات (Batched Generation) عبر vLLM...")
        batch_size = 50
        outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="vLLM Batched Generation"):
            batch_prompts = prompts[i:i + batch_size]
            batch_outputs = self.llm.generate(batch_prompts, sampling_params)
            outputs.extend(batch_outputs)

        final_dataset = []
        for output in outputs:
            raw_response = output.outputs[0].text
            try:
                parsed_data = repair_json(raw_response, return_objects=True)

                if isinstance(parsed_data, dict):
                    for item in parsed_data.get("qa_pairs", []):
                        u_q = item.get("user_question", "")
                        a_a = item.get("assistant_answer", "")

                        if u_q and a_a:
                            entry = {
                                "messages": [
                                    {"role": "system", "content": "أنت مساعد ذكي ومتخصص، تجيب بدقة وبشكل واضح بناءً على المعرفة المتاحة."},
                                    {"role": "user", "content": u_q},
                                    {"role": "assistant", "content": a_a}
                                ]
                            }
                            final_dataset.append(entry)
            except Exception as e:
                continue

        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for record in final_dataset:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"\n اكتملت العملية بنجاح!")
        print(f" عدد عينات التدريب النهائية: {len(final_dataset)}")
        print(f" تم حفظ الملف في: {output_jsonl}")


# ==========================================
# 3. التشغيل الرئيسي
# ==========================================
if __name__ == "__main__":
    pipeline_obj = DatasetPipeline(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        chunk_size_tokens=500,
        chunk_overlap_tokens=50
    )

    pipeline_obj.run_pipeline(
        input_path="./my_documents",
        output_jsonl="train_qa_dataset.jsonl",
        pairs_per_chunk=3
    )