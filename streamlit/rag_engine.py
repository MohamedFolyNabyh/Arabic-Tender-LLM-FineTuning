import os
import uuid
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker


class SimpleRAGEngine:
    def __init__(
        self,
        collection_name: str = "tender_documents",
        db_path: str = "./chroma_db",
    ):
        # 1. نموذج الـ Embedding المعزز للغة العربية
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. حفظ دائِم على القرص الصلب (Persistent Storage) بدلاً من الذاكرة فقط
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        # 3. محرك Docling عالي الدقة للـ PDF والجداول
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        self.converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.chunker = HybridChunker()

    def process_and_chunk_file(self, file_path: str) -> List[str]:
        """استخراج وتقطيع النص مع المحافظة على الهيكل والجداول"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            conv_res = self.converter.convert(str(path))
            doc_chunks = list(self.chunker.chunk(conv_res.document))
            chunks = [self.chunker.serialize(c) for c in doc_chunks if self.chunker.serialize(c).strip()]
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # تقطيع بسيط للملفات النصية السادة
            words = text.split()
            chunks = [" ".join(words[i : i + 300]) for i in range(0, len(words), 250)]
        else:
            raise ValueError("نوع الملف غير مدعوم، يرجى استخدام .pdf أو .txt")

        return chunks

    def add_document(self, file_path: str) -> int:
        """إضافة المستند لـ ChromaDB مع UUID و Metadatas"""
        chunks = self.process_and_chunk_file(file_path)
        if not chunks:
            return 0

        file_name = Path(file_path).name

        # توليد UUID فريد لكل Chunk لمنع تداخل البيانات عند رفع ملفات متعددة
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [{"source": file_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        return len(chunks)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """استرجاع المقاطع الأكثر صلة بالسؤال"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        return "\n\n---\n\n".join(documents)
