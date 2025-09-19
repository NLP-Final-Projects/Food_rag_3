# Persian Food RAG System

![Food](https://github.com/NLP-Final-Projects/Food_rag_3/blob/main/assets/Tahchin.jpg)

## Abstract

We present a Persian, food-domain retrieval-augmented generation (RAG) system that combines a dual-modality retriever with a lightweight generator. Building on our prior corpus and an added Kaggle recipe collection (1,737 entries; 1,393 unique dishes), we expand the index with web-sourced photos of dishes \emph{and} systematically collected images of key ingredients to strengthen image-grounded queries. The retriever pairs a Persian text encoder (500glot) with a fine-tuned CLIP vision--text encoder (vision-fa-clip/SajjadAyoubi) trained with a multi-positive contrastive objective to handle multiple instructions per dish. Canonicalized Persian texts and cross-modal embeddings enable answering both text-only and image+text questions by retrieving pertinent evidence and conditioning the generator.


<div align="center">

# Persian Culinary RAG: Multimodal Retrieval and Generation for Text–Image Food Queries

**Authors:** Sadegh Mohammadian | Arshia Izadyari | Mohammad Hossein Eslami

Fateme Asgari | Ali Rahimi Akbar | Mohammad Mahdi Vahedi

</div>

## Abstract

We present a Persian, food-domain retrieval-augmented generation (RAG) system that combines a dual-modality retriever with a lightweight generator. Building on our prior corpus and an added Kaggle recipe collection (1,737 entries; 1,393 unique dishes), we expand the index with web-sourced photos of dishes *and* systematically collected images of key ingredients to strengthen image-grounded queries. The retriever pairs a Persian text encoder (Glot-500) with a fine-tuned CLIP vision–text encoder (vision-fa-clip/SajjadAyoubi) trained with a multi-positive contrastive objective to handle multiple instructions per dish. Cross-modal embeddings enable answering both text-only and image+text questions by retrieving pertinent evidence and conditioning the generator. On held-out multiple-choice sets, the RAG setup improves performance for ingredient-triggered and image-grounded queries with a lighter generator, while gains are mixed for a stronger generator.

## Methods Summary

Our system is built on a Retrieval-Augmented Generation (RAG) architecture tailored for the Persian culinary domain. The core methodology can be summarized as follows:

1.  **Shared Representation Learning**: We map both images and text passages into a single, shared vector space. This is achieved using two main encoders:
    * A **CLIP-style vision tower** processes images.
    * A **transformer-based text encoder** (Glot-500), fine-tuned on our culinary dataset, processes textual information.
    Both encoders produce L2-normalized embeddings of the same dimension, allowing for direct comparison via cosine similarity.

2.  **Retrieval and Evidence Pooling**: We use FAISS (`IndexFlatIP`) for efficient similarity searches.
    * For **text-only queries**, the question is encoded and searched against an index of all text passages.
    * For **image queries**, the image is encoded and searched against the same text passage index (image-to-text retrieval).
    The top-k retrieved documents form an evidence pool that grounds the final answer.

3.  **RAG Prompt Construction**: Short snippets from the retrieved documents are extracted and compiled into a compact evidence block. This block is injected directly into the prompt given to the generative model, providing it with relevant context to formulate an answer.

4.  **Ingredient-Aware Training**: A key innovation in our approach is the use of **ingredient photos**. During training, images of a dish's main ingredients are treated as additional positive examples alongside the final plated dish. This encourages the model to create a more holistic, set-level concept of each dish, making the retriever more robust to queries that are based on raw ingredients.

## Dataset Properties

We constructed a novel multimodal dataset of Iranian foods from scratch. The process and properties are outlined below:

* **Data Collection**: We began by scraping a wide range of information about Iranian dishes from the internet using tools like Selenium and BeautifulSoup. The collected data included ingredients, preparation instructions, city of origin, and cultural context.

* **LLM-Powered Cleaning and Structuring**: The raw, unstructured scraped data was processed using a Large Language Model (GPT). The LLM standardized the information, resolved inconsistencies, and formatted it into a clean, uniform JSON structure.

* **Passage and Question Generation**: For each dish, we used another LLM (Gemma) to generate a fluent, descriptive passage. These passages form the knowledge base for our retriever. To create training data for the retriever, we prompted an LLM to generate at least five question-answer pairs that could be answered directly from each passage.

* **Multimodal Enrichment**: The dataset was enhanced with visual data. For each dish, we collected a set of representative images, including:
    * The final, plated dish.
    * Photos of its three main ingredients.
    This unique feature allows our system to understand dishes from both their final appearance and their core components.

* **Corpus Size**: The final dataset contains **1,737 total entries**, covering **1,393 unique dishes**.

* **Preprocessing Philosophy**: We deliberately avoided aggressive text preprocessing techniques like lemmatization or stemming. This decision was made to preserve the full linguistic richness of the text, ensuring that the RAG system has access to all potentially valuable information during retrieval.
```eof
