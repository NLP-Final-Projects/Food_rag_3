# Project Overview: A Multimodal RAG System for Persian Cuisine

The primary goal of this project was to develop an intelligent system capable of answering complex questions about Persian food using both text and images. This is a **multimodal** task because it requires the system to understand and reason over different types of data (modalities). To address this, we designed and implemented a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline.

### The Core Idea: An "Open-Book Exam" for AI

Standard AI models, even powerful ones, may lack deep, specialized knowledge on niche topics like regional Persian cuisine. A RAG system solves this problem by giving the AI an "open-book exam." Instead of relying solely on its internal memory, the system first performs a high-speed search through a dedicated knowledge base to find relevant information. This retrieved context is then provided to a powerful generator model, which uses it to formulate a highly accurate and contextually grounded answer.

### The Pipeline Explained

Our system is broken down into three main stages: Data Preparation, Retrieval, and Generation.

#### 1. Data Preparation: Building the Knowledge Base

The foundation of any RAG system is a high-quality knowledge base. We began by gathering a large dataset consisting of images of Persian dishes and corresponding JSON files containing detailed descriptions, ingredients, and preparation methods. This raw data was carefully processed and cleaned to create a central "document store" (`docstore.parquet`). A critical step in this phase was **deduplication**, where identical passages were removed to ensure the knowledge base was clean and efficient.

#### 2. The Retrieval System: Finding the Right Information

This is the core of the RAG architecture and is responsible for finding the most relevant documents for a given query. Because our system is multimodal, it has two distinct pathways for retrieval:

* **Text Retrieval:** For text-only questions (e.g., "What are the ingredients in Ghorme Sabzi?"), we used a **Glot-500** sentence transformer, fine-tuned specifically to understand the nuances of Persian text.
* **Image Retrieval:** For questions involving an image, we fine-tuned a **CLIP model**. CLIP is a powerful dual-encoder model designed to map both images and text into a shared "meaning space" (embedding space). By fine-tuning it on our Persian food dataset, we trained it to become an expert at recognizing a dish from an image and finding its corresponding textual description. When a user provides an image, the model converts it into a vector, which is then used to find the most similar text descriptions in our knowledge base.

To make the search process nearly instantaneous, we used **FAISS** (Facebook AI Similarity Search) to create a highly optimized index of all the document vectors. This allows the system to search through thousands of documents in milliseconds.

#### 3. The Generation System: Formulating the Answer

Once the top-ranking documents are retrieved, their text is combined with the user's original question and passed to a powerful Large Multimodal Model (LMM), such as **Gemini Pro** or **LLaVA**. This LMM acts as the "reasoning brain" of the pipeline. It reads the retrieved context and the question, synthesizes the information, and generates a final, coherent, and accurate answer.

#### 4. Evaluation

To measure the effectiveness of our pipeline, we benchmarked it on a custom Visual Question Answering (VQA) dataset. We compared the performance of the full RAG system against a baseline where the LMM answered questions without any retrieved context. This allowed us to quantify the significant improvement in accuracy provided by the retrieval augmentation.

### Final Outcome and Availability

The result of this project is a complete, end-to-end multimodal RAG system capable of answering detailed questions about Persian cuisine. A key contribution is the fine-tuned CLIP model, which is specialized for understanding Persian food imagery. This model is now publicly available for others to use in their own projects and can be accessed on the Hugging Face Hub.

**The final fine-tuned model is available at:**
[https://huggingface.co/Arshiaizd/MCLIP_FA_FineTuned/tree/main](https://huggingface.co/Arshiaizd/MCLIP_FA_FineTuned/tree/main)
