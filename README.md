# 🇸🇦 Arabic Tender LLM Fine-Tuning

## 🧠 نموذج ذكاء اصطناعي متخصص في كراسات الشروط والمناقصات باللغة العربية

مشروع متكامل لبناء نموذج لغة متخصص في **تحليل وفهم كراسات الشروط والمناقصات العربية** باستخدام تقنيات **Synthetic Data Generation** و**QLoRA Fine-Tuning**.

يعتمد المشروع على نموذج **Qwen2.5-7B-Instruct** كنموذج أساسي، ويتم تدريبه على بيانات اصطناعية يتم توليدها من مستندات المناقصات الفعلية باستخدام نموذج **Qwen2.5-3B-Instruct**، مع الحفاظ على بنية المستندات والجداول والقوائم باستخدام **Docling HybridChunker**.

بعد انتهاء التدريب، يتم تحويل النموذج إلى صيغة **GGUF Q4_K_M** وتشغيله محلياً باستخدام **llama.cpp** من خلال واجهة تفاعلية مبنية باستخدام **Streamlit**.

---

# 🎯 فكرة المشروع

الهدف من المشروع هو بناء مساعد ذكي قادر على فهم الأسئلة المتعلقة بكراسات الشروط والمناقصات باللغة العربية، والإجابة عنها بطريقة دقيقة ومباشرة.

بدلاً من الاعتماد فقط على نموذج عام، يتم تخصيص النموذج لمجال المناقصات من خلال:

```text
مستندات المناقصات
       ↓
استخراج النص والجداول
       ↓
تقسيم المستند إلى Chunks
       ↓
توليد أسئلة وإجابات اصطناعية
       ↓
Structured JSON Generation
       ↓
بناء Dataset
       ↓
QLoRA Fine-Tuning
       ↓
Qwen2.5-7B متخصص
       ↓
GGUF Q4_K_M
       ↓
llama.cpp
       ↓
Streamlit Chat
```

---

# ✨ المميزات الرئيسية

## 🇸🇦 دعم اللغة العربية

تم تصميم المشروع للتعامل مع المحتوى العربي، وخاصة المحتوى الرسمي والقانوني الخاص بالمناقصات وكراسات الشروط.

يدعم النموذج أسئلة مثل:

* ما هي مدة تنفيذ المشروع؟
* ما قيمة الضمان الابتدائي؟
* ما شروط التقديم؟
* ما المستندات المطلوبة؟
* ما غرامة التأخير؟
* ما شروط الدفع؟
* ما شروط فسخ العقد؟
* ما متطلبات التأهيل؟
* ما المهل الزمنية المحددة؟
* ما التزامات المقاول؟

---

## 📄 معالجة ملفات المناقصات

يستخدم المشروع **Docling** لمعالجة ملفات PDF مع مراعاة بنية المستند.

ويتم استخدام:

* PDF Parsing
* Layout-Aware Processing
* Table Structure Extraction
* Hybrid Chunking

وذلك بهدف الحفاظ قدر الإمكان على المعلومات الموجودة داخل:

* الفقرات
* الجداول
* القوائم
* العناوين
* المحتوى المنظم

---

# 🏗️ Architecture

المشروع يتكون من عدة مراحل رئيسية.

## المرحلة الأولى — Synthetic Dataset Generation

يتم استخدام:

**Qwen2.5-3B-Instruct**

لتوليد أسئلة وإجابات اصطناعية من نصوص المناقصات.

```text
Tender PDF
     │
     ▼
   Docling
     │
     ▼
HybridChunker
     │
     ▼
Document Chunks
     │
     ▼
Qwen2.5-3B-Instruct
     │
     ▼
Synthetic Q&A
     │
     ▼
JSON Schema Validation
     │
     ▼
JSONL Dataset
```

---

# 🧩 Dataset Engineering

يتم تحويل كل مستند إلى مجموعة من المقاطع النصية.

يستخدم المشروع:

```python
HybridChunker
```

من مكتبة Docling.

ويتم تمرير كل Chunk إلى نموذج التوليد مع Prompt متخصص لإنشاء عدة أزواج من الأسئلة والإجابات.

مثال:

```text
النص:

تبلغ مدة تنفيذ المشروع 180 يوماً من تاريخ استلام الموقع.

السؤال:

ما مدة تنفيذ المشروع؟

الإجابة:

مدة تنفيذ المشروع هي 180 يوماً من تاريخ استلام الموقع.
```

---

# 🔒 Structured Output

لضمان أن النموذج ينتج بيانات منظمة، يستخدم المشروع:

```python
Pydantic
```

مع:

```python
GuidedDecodingParams
```

من vLLM.

يتم تعريف Schema بالشكل التالي:

```text
QADatasetSchema
       │
       └── qa_pairs
              │
              ├── user_question
              └── assistant_answer
```

وهذا يقلل من احتمالية الحصول على مخرجات غير منظمة أو غير صالحة كبيانات تدريب.

---

# 📊 Dataset Format

يتم حفظ البيانات النهائية بصيغة:

```text
JSONL
```

مثال:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "أنت مساعد ذكي ومتخصص، تجيب بدقة وبشكل واضح بناءً على المعرفة المتاحة."
    },
    {
      "role": "user",
      "content": "ما هي مدة تنفيذ المشروع؟"
    },
    {
      "role": "assistant",
      "content": "مدة تنفيذ المشروع هي 180 يوماً من تاريخ استلام الموقع."
    }
  ]
}
```

---

# 🚀 Fine-Tuning

بعد إنشاء Dataset، يتم استخدام:

* Qwen2.5-7B-Instruct
* Unsloth
* QLoRA
* TRL
* Hugging Face Datasets

لتخصيص النموذج.

---

# ⚡ QLoRA

يتم تحميل النموذج باستخدام 4-bit Quantization:

```python
load_in_4bit=True
```

ثم يتم إضافة LoRA Adapters.

الإعدادات المستخدمة:

```text
LoRA Rank (r): 16
LoRA Alpha: 16
LoRA Dropout: 0
Bias: none
```

والطبقات المستهدفة:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

---

# 🧠 Model

النموذج الأساسي المستخدم في Fine-Tuning هو:

```text
Qwen/Qwen2.5-7B-Instruct
```

بينما يتم استخدام:

```text
Qwen/Qwen2.5-3B-Instruct
```

في مرحلة توليد البيانات الاصطناعية.

---

# 🏋️ Training Configuration

الإعدادات الحالية للتدريب:

```text
Batch Size: 2
Gradient Accumulation Steps: 4
Effective Batch Size: 8
Learning Rate: 2e-4
Warmup Steps: 5
Max Steps: 60
Sequence Length: 2048
```

ويتم اختيار:

```text
FP16
```

أو:

```text
BF16
```

حسب دعم كارت الشاشة.

---

# 📦 Model Quantization

بعد انتهاء Fine-Tuning، يتم تحويل النموذج إلى:

```text
GGUF
```

باستخدام:

```text
Q4_K_M
```

والهدف من ذلك هو تقليل حجم النموذج وتسهيل تشغيله محلياً.

---

# 💻 Local Inference

يتم تشغيل النموذج النهائي باستخدام:

```text
llama-cpp-python
```

بدلاً من الحاجة إلى تشغيل vLLM في بيئة الإنتاج الخاصة بالواجهة.

يتم تحميل النموذج من:

```text
Hugging Face Hub
```

أو من المسار المحلي:

```text
qwen_tender_model_gguf/
```

---

# 🔎 RAG Integration

يحتوي التطبيق النهائي أيضاً على محرك RAG:

```python
SimpleRAGEngine
```

والذي يسمح للمستخدم برفع كراسة الشروط ثم استرجاع الأجزاء الأكثر ارتباطاً بالسؤال.

التدفق النهائي:

```text
User
 │
 ▼
Upload PDF / TXT
 │
 ▼
SimpleRAGEngine
 │
 ▼
Document Processing
 │
 ▼
Relevant Context
 │
 ▼
Qwen2.5-7B Fine-Tuned
 │
 ▼
Arabic Answer
```

وبذلك يجمع المشروع بين:

**Fine-Tuning + RAG**

بدلاً من الاعتماد على إحدى التقنيتين فقط.

---

# 💬 Streamlit Application

تم بناء واجهة المستخدم باستخدام:

```text
Streamlit
```

وتدعم:

* واجهة Chat
* اللغة العربية
* RTL
* رفع PDF
* رفع TXT
* Streaming Responses
* Chat History
* RAG Retrieval
* Local LLM Inference

---

# 🎨 Arabic RTL Interface

تم تخصيص واجهة Streamlit لدعم اللغة العربية من خلال CSS.

ويتم استخدام خط:

```text
Cairo
```

مع:

```css
direction: rtl;
text-align: right;
```

مما يجعل تجربة استخدام التطبيق مناسبة للمستخدم العربي.

---

# 📁 Project Structure

البنية المقترحة للمشروع:

```text
Arabic-Tender-LLM-FineTuning/
│
├── data/
│   ├── raw_documents/
│   │   └── *.pdf
│   │
│   └── processed/
│       └── train_qa_dataset.jsonl
│
├── outputs/
│
├── qwen_tender_model_gguf/
│   └── qwen2.5-7b-instruct.Q4_K_M.gguf
│
├── app.py
│
├── dataset_pipeline.py
│
├── train.py
│
├── rag_engine.py
│
├── requirements.txt
│
└── README.md
```

> قد تختلف أسماء الملفات الفعلية حسب تنظيم المشروع المحلي.

---

# 🛠️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/MohamedFolyNabyh/Arabic-Tender-LLM-FineTuning.git
cd Arabic-Tender-LLM-FineTuning
```

---

## 2. إنشاء البيئة

باستخدام Conda:

```bash
conda create -n tender_llm python=3.10 -y
conda activate tender_llm
```

أو باستخدام venv:

```bash
python -m venv venv
```

Linux / macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

المشروع يعتمد بشكل أساسي على:

```text
torch
transformers
datasets
trl
unsloth
vllm
docling
pydantic
tqdm
streamlit
llama-cpp-python
huggingface_hub
```

---

# 🔐 Hugging Face Token

إذا كنت تريد رفع النموذج إلى Hugging Face، قم بتعيين:

Linux / macOS:

```bash
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

Windows:

```powershell
$env:HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

أو قم بتسجيل الدخول:

```bash
huggingface-cli login
```

> لا تقم أبداً بوضع Hugging Face Token داخل GitHub أو داخل ملفات المشروع.

---

# 📄 Preparing Documents

ضع ملفات المناقصات داخل:

```text
data/raw_documents/
```

مثال:

```text
data/raw_documents/
├── tender_01.pdf
├── tender_02.pdf
├── tender_03.pdf
└── requirements.pdf
```

يدعم Pipeline ملفات:

```text
PDF
TXT
MD
```

---

# 🤖 Generate Synthetic Dataset

بعد وضع المستندات، قم بتشغيل Pipeline توليد البيانات.

الإعداد الافتراضي:

```text
Model:
Qwen/Qwen2.5-3B-Instruct

Input:
data/raw_documents

Output:
data/processed/train_qa_dataset.jsonl

Pairs per Chunk:
3
```

مثال:

```bash
python dataset_pipeline.py
```

سيقوم النظام بـ:

1. قراءة ملفات المناقصات.
2. تحليل ملفات PDF باستخدام Docling.
3. استخراج النصوص والجداول.
4. تقسيم المستندات باستخدام HybridChunker.
5. توليد أسئلة وإجابات.
6. التحقق من Structured Output.
7. حفظ Dataset بصيغة JSONL.

---

# 🏋️ Start Fine-Tuning

بعد إنشاء:

```text
data/processed/train_qa_dataset.jsonl
```

قم بتشغيل التدريب:

```bash
python train.py
```

الإعداد الافتراضي:

```text
Base Model:
Qwen/Qwen2.5-7B-Instruct

Quantization:
4-bit

Method:
QLoRA

Max Steps:
60

Sequence Length:
2048
```

---

# 📤 Upload Model to Hugging Face

بعد انتهاء التدريب يتم تحويل النموذج إلى:

```text
GGUF Q4_K_M
```

ثم رفعه إلى Hugging Face Hub.

يمكن تحديد المستودع من خلال:

```python
hf_repo_id="MohamedFoly/qwen2.5-saudi-tender-gguf"
```

---

# 🧠 Running the Application

بعد توفر نموذج GGUF، يمكن تشغيل التطبيق:

```bash
streamlit run app.py
```

ثم فتح:

```text
http://localhost:8501
```

---

# 🔄 Runtime Flow

عند تشغيل التطبيق:

```text
Streamlit
   │
   ▼
Load GGUF Model
   │
   ▼
llama.cpp
   │
   ▼
Upload Tender
   │
   ▼
RAG Engine
   │
   ▼
Retrieve Relevant Context
   │
   ▼
Fine-Tuned Qwen
   │
   ▼
Streaming Arabic Response
```

---

# 📝 Example

بعد رفع كراسة الشروط، يمكن للمستخدم السؤال:

```text
ما هي غرامة التأخير في تنفيذ المشروع؟
```

يقوم النظام أولاً باسترجاع الأجزاء المرتبطة بالسؤال من المستند.

ثم يتم إرسال السياق إلى النموذج:

```text
System:
أنت مستشار قانوني متخصص...

Context:
[النصوص المسترجعة من كراسة الشروط]

User:
ما هي غرامة التأخير؟
```

ثم يقوم النموذج بتوليد الإجابة.

---

# ⚙️ Inference Parameters

يتم تشغيل النموذج باستخدام إعدادات مناسبة للإجابات الدقيقة:

```text
Context Length: 4096
Max Tokens: 512
Temperature: 0.1
Stop Token: <|im_end|>
```

ويتم استخدام:

```python
stream=True
```

لعرض الإجابة بشكل تدريجي داخل Streamlit.

---

# 🧪 Why Fine-Tuning + RAG?

يعتمد المشروع على تقنيتين مختلفتين لكل منهما وظيفة محددة.

## Fine-Tuning

يهدف إلى تعليم النموذج:

* أسلوب الإجابة.
* المصطلحات المتخصصة.
* طبيعة أسئلة المناقصات.
* التعامل مع اللغة الرسمية.
* نمط الحوار المطلوب.

## RAG

يهدف إلى توفير:

* المعلومات الموجودة في المستند الحالي.
* سياق كراسة الشروط.
* القدرة على التعامل مع مستندات جديدة.
* تقليل الاعتماد على المعرفة المحفوظة داخل النموذج.

لذلك:

```text
Fine-Tuning
=
تعليم النموذج كيف يجيب

RAG
=
تزويد النموذج بالمعلومات التي يحتاجها للإجابة
```

---

# ⚠️ Important Disclaimer

هذا المشروع **نموذج بحثي وتقني** لمعالجة وتحليل نصوص المناقصات وكراسات الشروط.

الإجابات التي ينتجها النموذج لا ينبغي اعتبارها بديلاً عن:

* الاستشارة القانونية المتخصصة.
* مراجعة المستندات الأصلية.
* الأنظمة واللوائح الرسمية.
* الجهات الحكومية المختصة.

يجب دائماً الرجوع إلى المصدر الرسمي قبل اتخاذ أي قرار قانوني أو تجاري.

---

# 🚀 Future Improvements

يمكن تطوير المشروع مستقبلاً من خلال:

* تحسين جودة Synthetic Dataset.
* زيادة حجم Dataset.
* استخدام بيانات حقيقية ومراجعة بشرية.
* إضافة Human Evaluation.
* تقييم النموذج باستخدام Benchmarks متخصصة.
* تحسين RAG Retrieval.
* إضافة Reranker.
* دعم OCR للمستندات الممسوحة.
* استخراج الجداول بشكل أكثر دقة.
* إظهار رقم الصفحة لكل إجابة.
* إضافة Citations للمصادر.
* دعم DOCX وXLSX.
* إضافة Multi-Document RAG.
* مقارنة عدة كراسات شروط.
* استخراج المتطلبات تلقائياً.
* إنشاء Tender Compliance Checklist.
* استخراج المخاطر القانونية والتجارية.
* دعم GPU Acceleration عبر CUDA.
* تحسين سرعة Streaming.
* إضافة Authentication للمستخدمين.

---

# 📈 Project Highlights

| Component             | Technology           |
| --------------------- | -------------------- |
| Base LLM              | Qwen2.5-7B-Instruct  |
| Synthetic Data Model  | Qwen2.5-3B-Instruct  |
| Fine-Tuning           | QLoRA                |
| Fine-Tuning Framework | Unsloth              |
| Training              | TRL / SFTTrainer     |
| PDF Processing        | Docling              |
| Chunking              | HybridChunker        |
| Structured Generation | vLLM Guided Decoding |
| Validation            | Pydantic             |
| Quantization          | GGUF Q4_K_M          |
| Local Inference       | llama.cpp            |
| RAG                   | SimpleRAGEngine      |
| UI                    | Streamlit            |
| Model Hosting         | Hugging Face Hub     |

---

# 👨‍💻 Author

Developed by:

**Mohamed Foly Nabyh Mohamed**

GitHub:

```text
MohamedFolyNabyh
```

---

# 🔗 Project

GitHub Repository:

https://github.com/MohamedFolyNabyh/Arabic-Tender-LLM-FineTuning

---

# ⭐ Support

إذا وجدت المشروع مفيداً، يمكنك دعم المشروع من خلال:

* ⭐ إعطاء Star للمستودع.
* 🐛 الإبلاغ عن الأخطاء.
* 💡 اقتراح تحسينات.
* 🔧 المساهمة في التطوير.
* 📚 مشاركة المشروع مع المهتمين بالـ Arabic LLMs وRAG وFine-Tuning.

---

# 🇸🇦 Arabic Tender AI

**Fine-Tuned Arabic LLM for Tender & Procurement Documents**

```text
Synthetic Data
      +
QLoRA Fine-Tuning
      +
RAG
      +
GGUF
      +
llama.cpp
      +
Streamlit
      =
Arabic Tender AI Assistant
```
