# import os
# import streamlit as st
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download

# # --- 1. ضبط إعدادات الصفحة ---
# st.set_page_config(
#     page_title="مساعد كراسات الشروط والمناقصات الذكي",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- 2. تحسين مظهر الواجهة ودعم الاتجاه العربي (RTL) ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
#     html, body, [class*="css"] {
#         font-family: 'Cairo', sans-serif;
#         direction: rtl;
#         text-align: right;
#     }
#     .stChatMessage {
#         direction: rtl;
#         text-align: right;
#     }
#     .stMarkdown p {
#         text-align: right;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # # --- 3. دالة تحميل النموذج مع التخزين المؤقت (Caching) ---
# @st.cache_resource
# def load_llm():
#     model_repo = "foly884/qwen-saudi-tender-gguf"
#     # تعديل اسم الملف ليكون مطابقاً للموجود في مستودعك بالضبط
#     model_file = "qwen2.5-7b-instruct.Q4_K_M.gguf"
#     local_path = os.path.join("qwen_tender_model_gguf", model_file)
    
#     # التحقق من وجود الملف محلياً أولاً
#     if os.path.exists(local_path):
#         model_path = local_path
#     elif os.path.exists(model_file):
#         model_path = model_file
#     else:
#         hf_token = os.getenv("HF_TOKEN")
#         model_path = hf_hub_download(
#             repo_id=model_repo,
#             filename=model_file,
#             token=hf_token
#         )
        
#     llm = Llama(
#         model_path=model_path,
#         n_ctx=2048,
#         n_threads=4,
#         verbose=False
#     )
#     return llm
# # @st.cache_resource
# # def load_llm():
# #     model_repo = "foly884/qwen-saudi-tender-gguf"
# #     model_file = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
# #     local_path = os.path.join("qwen_tender_model_gguf", model_file)
    
# #     # التحقق من وجود الملف محلياً أولاً
# #     if os.path.exists(local_path):
# #         model_path = local_path
# #     elif os.path.exists(model_file):
# #         model_path = model_file
# #     else:
# #         hf_token = os.getenv("HF_TOKEN")ظظ
# #         model_path = hf_hub_download(
# #             repo_id=model_repo,
# #             filename=model_file,
# #             token=hf_token
# #         )
        
# #     llm = Llama(
# #         model_path=model_path,
# #         n_ctx=2048,
# #         n_threads=4,
# #         verbose=False
# #     )
# #     return llm

# # --- 4. تحميل النموذج والتعامل مع أخطاء الاتصال ---
# try:
#     llm = load_llm()
#     is_connected = True
#     error_msg = None
# except Exception as e:
#     is_connected = False
#     error_msg = str(e)

# # --- 5. الشريط الجانبي (Sidebar) ---
# with st.sidebar:
#     st.header("⚙️ حالة النظام")
#     if is_connected:
#         st.success("🟢 Llama-CPP Direct Engine Connected")
#     else:
#         st.error("🔴 Connection Failed")
#         st.caption(f"الخطأ: {error_msg}")

#     st.markdown("---")
#     st.header("📊 معلومات النموذج")
#     st.write("**Model:** Tender-AI (Qwen2.5-7B-Instruct)")
#     st.write("**Quantization:** GGUF (Q4_K_M)")

#     st.markdown("---")
#     st.header("🛠️ الأدوات المستخدمة")
#     st.markdown("""
#     * Unsloth + QLoRA
#     * Llama-CPP Python
#     * Streamlit Chat
#     """)

# # --- 6. الواجهة الرئيسية ---
# st.title("📄 مساعد كراسات الشروط والمناقصات الذكي")
# st.caption("نموذج ذكاء اصطناعي مدرب دقيقاً لمعالجة وتحليل مستندات المناقصات الرسمية باللغة العربية")

# # تهيئة سجل المحادثة
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {
#             "role": "assistant", 
#             "content": "مرحباً بك! أنا جاهز للرد على استفساراتك حول كراسة الشروط والمناقصة. كيف يمكنني مساعدتك اليوم؟"
#         }
#     ]

# # عرض سجل الرسائل السابقة
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # --- 7. معالجة الإدخال والتوليد المباشر ---
# if prompt := st.chat_input("اطرح سؤالك حول الشروط، الضمانات، أو المهل الزمنية..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)

#     with st.chat_message("assistant"):
#         if is_connected:
#             system_prompt = (
#                 "أنت خبير ومستشار قانوني متخصص في نظام المناقصات والمشتريات الحكومية. "
#                 "أجب على أسئلة المستخدم بدقة وبشكل مباشر ومحدد بناءً على اللوائح التنفيذية والشروط الرسمية، "
#                 "وتجنب إعادة صياغة السؤال كإجابة."
#             )
            
#             full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
#             # توليد الاستجابة بتقنية Streaming لقراءة الإجابة فور كتابتها
#             stream = llm(
#                 full_prompt,
#                 max_tokens=512,
#                 temperature=0.1,
#                 top_p=0.9,
#                 repeat_penalty=1.2,
#                 presence_penalty=0.2,
#                 stop=["<|im_end|>"],
#                 stream=True
#             )
            
#             response_container = st.empty()
#             full_response = ""
            
#             for chunk in stream:
#                 token = chunk["choices"][0]["text"]
#                 full_response += token
#                 response_container.write(full_response + "▌")
                
#             response_container.write(full_response)
#             st.session_state.messages.append({"role": "assistant", "content": full_response})
#         else:
#             st.error("تعذر الاتصال بالنموذج. يرجى التأكد من مسار الملف أو التوكن الخاص بـ Hugging Face.")



# import os
# import tempfile
# import streamlit as st
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download
# from rag_engine import SimpleRAGEngine

# # --- 1. ضبط إعدادات الصفحة ---
# st.set_page_config(
#     page_title="مساعد المناقصات الذكي مع RAG",
#     page_icon="📄",
#     layout="wide"
# )

# # --- 2. تحسين مظهر الواجهة لدعم العربية (RTL) ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
#     html, body, [class*="css"] {
#         font-family: 'Cairo', sans-serif;
#         direction: rtl;
#         text-align: right;
#     }
#     .stChatMessage { direction: rtl; text-align: right; }
#     .stMarkdown p { text-align: right; }
#     </style>
# """, unsafe_allow_html=True)

# # --- 3. تحميل النموذج ومحرك RAG باستعمال Caching ---
# @st.cache_resource
# def load_rag_engine():
#     return SimpleRAGEngine()

# @st.cache_resource
# def load_llm():
#     model_repo = "foly884/qwen-saudi-tender-gguf"
#     model_file = "qwen2.5-7b-instruct.Q4_K_M.gguf"
#     local_path = os.path.join("qwen_tender_model_gguf", model_file)
    
#     if os.path.exists(local_path):
#         model_path = local_path
#     elif os.path.exists(model_file):
#         model_path = model_file
#     else:
#         hf_token = os.getenv("HF_TOKEN")
#         model_path = hf_hub_download(
#             repo_id=model_repo,
#             filename=model_file,
#             token=hf_token
#         )
        
#     return Llama(
#         model_path=model_path,
#         n_ctx=4096,
#         n_threads=4,
#         verbose=False
#     )

# rag = load_rag_engine()
# llm = load_llm()

# # --- 4. الشريط الجانبي ورفع الملفات ---
# with st.sidebar:
#     st.header("📂 رفع كراسة الشروط")
#     uploaded_file = st.file_uploader("اختر ملف المناقصة (PDF أو TXT)", type=["pdf", "txt"])
    
#     if uploaded_file is not None:
#         if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
#             with st.spinner("جاري معالجة الملف وتحليله..."):
#                 # حفظ الملف المؤقت لقراءته
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
#                     tmp_file.write(uploaded_file.read())
#                     tmp_path = tmp_file.name
                
#                 num_chunks = rag.add_document(tmp_path)
#                 os.remove(tmp_path) # حذف الملف المؤقت
                
#                 st.session_state.processed_file = uploaded_file.name
#                 st.success(f"تم تحليل الملف بنجاح! ({num_chunks} أجزاء)")

# # --- 5. الواجهة الرئيسية واستقبال الأسئلة ---
# st.title("📄 مساعد كراسات الشروط والمناقصات الذكي")
# st.caption("اسأل عن أي بند أو شروط داخل الكراسة المرفوعة")

# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "مرحباً بك! قم برفع ملف كراسة الشروط من القائمة الجانبية وابدأ بطرح استفساراتك."}
#     ]

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# if prompt := st.chat_input("اطرح سؤالك حول الملف المرفوع..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)

#     with st.chat_message("assistant"):
#         # استرجاع النص المرتبط بالسؤال
#         context = rag.retrieve_context(prompt)
        
#         if context:
#             system_prompt = (
#                 "أنت مستشار قانوني متخصص. أجب على سؤال المستخدم بناءً على النصوص المستخرجة من الكراسة أدناه فقط.\n"
#                 "إذا لم تجد الإجابة في النص المرفق، أبلغ المستخدم بوضوح أن المعلومة غير موجودة في الكراسة المرفوعة.\n\n"
#                 f"نصوص الكراسة المرفقة:\n{context}"
#             )
#         else:
#             system_prompt = (
#                 "أنت مستشار قانوني متخصص. أجب على سؤال المستخدم بوضوح وبناءً على الأنظمة الرسمية."
#             )

#         full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

#         stream = llm(
#             full_prompt,
#             max_tokens=512,
#             temperature=0.0,
#             stop=["<|im_end|>"],
#             stream=True
#         )

#         response_container = st.empty()
#         full_response = ""

#         for chunk in stream:
#             token = chunk["choices"][0]["text"]
#             full_response += token
#             response_container.write(full_response + "▌")

#         response_container.write(full_response)
#         st.session_state.messages.append({"role": "assistant", "content": full_response})



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
        n_ctx=4096,      # تقليل سياق الذاكرة من 4096 إلى 2048 لتسريع الاستجابة
        n_threads=6,     # زِد عدد الـ Threads حسب أنوية المعالج لديك (مثلاً 6 أو 8)
        n_batch=512,     # تسريع معالجة الـ Prompt
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
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
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
            
            if context:
                system_prompt = (
                    "أنت مستشار قانوني متخصص. أجب على سؤال المستخدم بناءً على النصوص المستخرجة من الكراسة أدناه فقط.\n"
                    "إذا لم تجد الإجابة في النص المرفق، أبلغ المستخدم بوضوح أن المعلومة غير موجودة في الكراسة المرفوعة.\n\n"
                    f"نصوص الكراسة المرفقة:\n{context}"
                )
            else:
                system_prompt = (
                    "أنت مستشار قانوني متخصص. أجب على سؤال المستخدم بوضوح وبناءً على الأنظمة الرسمية."
                )

            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            stream = llm(
                full_prompt,
                max_tokens=512,
                temperature=0.1,
                
                stop=["<|im_end|>"],
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