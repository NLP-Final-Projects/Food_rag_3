# Persian Food RAG System

![Food](https://github.com/NLP-Final-Projects/Food_rag_3/blob/main/assets/Tahchin.jpg)

## Abstract

We present a Persian, food-domain retrieval-augmented generation (RAG) system that combines a dual-modality retriever with a lightweight generator. Building on our prior corpus and an added Kaggle recipe collection (1,737 entries; 1,393 unique dishes), we expand the index with web-sourced photos of dishes \emph{and} systematically collected images of key ingredients to strengthen image-grounded queries. The retriever pairs a Persian text encoder (500glot) with a fine-tuned CLIP vision--text encoder (vision-fa-clip/SajjadAyoubi) trained with a multi-positive contrastive objective to handle multiple instructions per dish. Canonicalized Persian texts and cross-modal embeddings enable answering both text-only and image+text questions by retrieving pertinent evidence and conditioning the generator.
