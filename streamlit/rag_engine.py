import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

class SimpleRAGEngine:
    def __init__(self, collection_name="tender_documents"):
        # استخدام نموذج متعدد اللغات خفيف وممتاز للغة العربية
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # قاعدة بيانات متجهات تنشأ في الذاكرة (In-Memory)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def extract_text_from_file(self, file_path: str) -> str:
        """استخراج النص من ملف PDF أو TXT"""
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError("نوع الملف غير مدعوم، يرجى استخدام .pdf أو .txt")
            
        return text

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list:
        """تقسيم النص إلى أجزاء متداخلة لضمان اكتمال السياق"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def add_document(self, file_path: str):
        """قراءة الملف وتقطيعه ثم إضافته إلى ChromaDB"""
        raw_text = self.extract_text_from_file(file_path)
        chunks = self.chunk_text(raw_text)
        
        # إنشاء معرفات فريدة لكل جزء
        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
        
        # إضافة النصوص للـ Collection
        self.collection.add(
            documents=chunks,
            ids=ids
        )
        return len(chunks)

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """استرجاع المقاطع الأكثر صلة بسؤال المستخدم"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""
        return "\n---\n".join(documents)