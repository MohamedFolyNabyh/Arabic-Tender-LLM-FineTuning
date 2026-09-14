import os
import tempfile
import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from rag_engine import SimpleRAGEngine

# --- 1. ضبط إعدادات الصفحة ---
st.set_page_config(
    page_title="مساعد المناقصات الذكي مع RAG",
    page_icon="📄",
    layout="wide"
)

# --- 2. تحسين مظهر الواجهة لدعم العربية (RTL) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stChatMessage { direction: rtl; text-align: right; }
    .stMarkdown p { text-align: right; }
    </style>
""", unsafe_allow_html=True)

# --- 3. تهيئة محرك RAG داخل Session State وتحميل LLM ---
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = SimpleRAGEngine()

rag = st.session_state.rag_engine

@st.cache_resource
def load_llm():
    model_repo = "foly884/qwen-saudi-tender-gguf"
    model_file = "qwen2.5-7b-instruct.Q4_K_M.gguf"
    local_path = os.path.join("qwen_tender_model_gguf", model_file)
    
    if os.path.exists(local_path):
        model_path = local_path
    elif os.path.exists(model_file):
        model_path = model_file
    else:
        hf_token = os.getenv("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            token=hf_token
        )
        
    return Llama(
        model_path=model_path,
        n_ctx=4096,       # إبقاء الـ Context 4096 لاستيعاب نصوص الـ RAG والمحادثات
        n_threads=6,      # زِد الأنوية بناءً على قدرة المعالج
        n_batch=512,      # تسريع معالجة الـ Prompt
        verbose=False
    )

llm = load_llm()

# --- 4. الشريط الجانبي ورفع الملفات ---
with st.sidebar:
    st.header("📂 رفع كراسة الشروط")
    uploaded_file = st.file_uploader("اختر ملف المناقصة (PDF أو TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("جاري معالجة الملف وتحليله..."):
                # إعادة إنشاء Collection جديدة لكل ملف جديد لمنع تداخل البيانات
                if hasattr(rag, 'reset_collection'):
                    rag.reset_collection()
                
                ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                num_chunks = rag.add_document(tmp_path)
                os.remove(tmp_path)
                
                st.session_state.processed_file = uploaded_file.name
                st.success(f"تم تحليل الملف بنجاح! ({num_chunks} أجزاء)")

# --- 5. الواجهة الرئيسية واستقبال الأسئلة ---
st.title("📄 مساعد كراسات الشروط والمناقصات الذكي")
st.caption("اسأل عن أي بند أو شروط داخل الكراسة المرفوعة")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحباً بك! قم برفع ملف كراسة الشروط من القائمة الجانبية وابدأ بطرح استفساراتك."}
    ]

# عرض سجل المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال السؤال ومعالجته
if prompt := st.chat_input("اطرح سؤالك حول الملف المرفوع..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والتوليد..."):
            context = rag.retrieve_context(prompt)
            
            # 1. بناء النظام (System Prompt) مع السياق المسترجع
            if context:
                system_instruction = (
                    "أنت مستشار قانوني متخصص في المناقصات السعودية. أجب على سؤال المستخدم بناءً على النصوص المستخرجة من الكراسة أدناه فقط.\n"
                    "إذا لم تجد الإجابة في النص المرفق، أبلغ المستخدم بوضوح أن المعلومة غير موجودة في الكراسة المرفوعة.\n\n"
                    f"نصوص الكراسة المرفقة:\n{context}"
                )
            else:
                system_instruction = (
                    "أنت مستشار قانوني متخصص في المناقصات والأحكام السعودية. أجب على سؤال المستخدم بوضوح وبناءً على الأنظمة الرسمية."
                )

            # 2. تحضير الرسائل بتنسيق Qwen ChatML مع مراعاة السجل والتاريخ
            formatted_messages = [f"<|im_start|>system\n{system_instruction}<|im_end|>"]
            
            # تجميع آخر رسالتين فقط للحفاظ على طول النافذة وحماية الأداء
            history_messages = st.session_state.messages[-3:-1]
            for msg in history_messages:
                formatted_messages.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            
            formatted_messages.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            full_prompt = "\n".join(formatted_messages)

            # 3. التوليد عبر المحرك
            stream = llm(
                full_prompt,
                max_tokens=512,
                temperature=0.1,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True
            )

            response_container = st.empty()
            full_response = ""

            for chunk in stream:
                token = chunk["choices"][0]["text"]
                full_response += token
                response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
