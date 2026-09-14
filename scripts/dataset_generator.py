import os
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from tqdm import tqdm

# ==========================================
# 1. Pydantic Schemas للتحقق والـ Guided Generation
# ==========================================
class QAPair(BaseModel):
    user_question: str = Field(description="سؤال واقعي ومباشر بناءً على النص")
    assistant_answer: str = Field(description="إجابة دقيقة وشاملة مستخرجة من النص فقط")

class QADatasetSchema(BaseModel):
    qa_pairs: List[QAPair]


# ==========================================
# 2. Main Dataset Engineering Pipeline
# ==========================================
class SyntheticDatasetPipeline:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        gpu_utilization: float = 0.85,
        max_model_len: int = 4096,
    ):
        print(f"🚀 [1/2] تهيئة محرك vLLM ({model_id})...")
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        print("📄 [2/2] تهيئة محرك Docling المعزز بـ Layout-Aware Chunker...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer)

        self.system_prompt = (
            "أنت مهندس بيانات متخصص في إعداد مجموعات البيانات لتدريب النماذج اللغوية.\n"
            "مهمتك: تحليل النص المرفق واستخراج أسئلة وإجابات متنوعة كأنها حوار طبيعي بين مستخدم ومساعد ذكي.\n"
            "القواعد:\n"
            "1. يجب أن تكون الإجابات معتمدة تماماً على النص المتاح فقط دون اختراع معلومات.\n"
            "2. احتفظ بتفاصيل الجداول والقوائم المذكورة في النص واشرحها بوضوح في الإجابة.\n"
            "3. اكتب بلغة عربية سليمة وواضحة."
        )

    def extract_chunks(self, input_path: Path) -> List[str]:
        chunks = []
        files = [input_path] if input_path.is_file() else list(input_path.rglob("*"))
        
        for file_path in files:
            if file_path.suffix.lower() == ".pdf":
                print(f"🔄 معالجة المستند: {file_path.name}")
                try:
                    conv_res = self.converter.convert(str(file_path))
                    doc_chunks = list(self.chunker.chunk(conv_res.document))
                    for chunk in doc_chunks:
                        chunk_text = self.chunker.serialize(chunk)
                        if chunk_text.strip():
                            chunks.append(chunk_text)
                except Exception as e:
                    print(f"⚠️ خطأ في معالجة الملف {file_path.name}: {e}")
            elif file_path.suffix.lower() in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    if text:
                        chunks.append(text)

        print(f"✅ إجمالي المقاطع المستخرجة المحافظة على الهيكل: {len(chunks)}")
        return chunks

    def run(self, input_dir: str, output_file: str, pairs_per_chunk: int = 3):
        input_path = Path(input_dir)
        chunks = self.extract_chunks(input_path)

        if not chunks:
            print("⚠️ لم يتم العثور على أي نصوص معالجة!")
            return

        # إعداد الـ Prompts
        prompts = []
        for chunk in chunks:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"قم باستخراج {pairs_per_chunk} أزواج أسئلة وإجابات متكاملة بناءً على النص المرفق فقط:\n\n"
                        f"---\n{chunk}\n---"
                    ),
                },
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # إعداد البارامترات وضمان الـ Structured Output عبارة عن JSON Schema مطابقة لـ QADatasetSchema
        guided_params = GuidedDecodingParams(json=QADatasetSchema.model_json_schema())
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=1500,
            guided_decoding=guided_params,
        )

        print(f"⚡ بدء التوليد السريع (Continuous Batching) لـ {len(prompts)} مقطع...")
        outputs = self.llm.generate(prompts, sampling_params)

        final_dataset = []
        for output in outputs:
            raw_json = output.outputs[0].text
            try:
                data = json.loads(raw_json)
                for qa in data.get("qa_pairs", []):
                    if qa.get("user_question") and qa.get("assistant_answer"):
                        final_dataset.append({
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "أنت مساعد ذكي ومتخصص، تجيب بدقة وبشكل واضح بناءً على المعرفة المتاحة.",
                                },
                                {"role": "user", "content": qa["user_question"]},
                                {"role": "assistant", "content": qa["assistant_answer"]},
                            ]
                        })
            except json.JSONDecodeError:
                continue

        # حفظ البيانات بصيغة JSONL
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in final_dataset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"🎉 تم الإنشاء بنجاح! عدد العينات: {len(final_dataset)}")
        print(f"📁 المسار: {out_path.resolve()}")


# ==========================================
# 3. التشغيل المباشر بدون Parser
# ==========================================
if __name__ == "__main__":
    # اضبط المتغيرات هنا مباشرة
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    INPUT_DIR = "data/raw_documents"
    OUTPUT_FILE = "data/processed/train_qa_dataset.jsonl"
    PAIRS_PER_CHUNK = 3

    pipeline = SyntheticDatasetPipeline(model_id=MODEL_ID)
    pipeline.run(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        pairs_per_chunk=PAIRS_PER_CHUNK,
    )
