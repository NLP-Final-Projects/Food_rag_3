
<div align="center">

# Persian Culinary RAG: Multimodal Retrieval and Generation for Text–Image Food Queries

**Mohammad Hossein Eslami,Arshia Izadyari,Sadegh Mohammadian,Fateme Asgari,Ali RahimiAkbar,Mohammad Mahdi Vahedi**


</div>

## Abstract

We present a Persian, food-domain retrieval-augmented generation (RAG) system that combines a dual-modality retriever with a lightweight generator. Building on our prior corpus and an added Kaggle recipe collection (1{,}737 entries; 1{,}393 unique dishes), we expand the index with web-sourced photos of dishes \emph{and} systematically collected images of key ingredients to strengthen image-grounded queries. The retriever pairs a Persian text encoder (Glot-500) with a fine-tuned CLIP vision--text encoder (vision-fa-clip/SajjadAyoubi) trained with a multi-positive contrastive objective to handle multiple instructions per dish. Cross-modal embeddings enable answering both text-only and image+text questions by retrieving pertinent evidence and conditioning the generator. On held-out multiple-choice sets, the RAG setup improves performance for ingredient-triggered and image-grounded queries with a lighter generator, while gains are mixed for a stronger generator.

## Methods Summary

Our system is built on a Retrieval-Augmented Generation (RAG) architecture tailored for the Persian culinary domain. At its core is a shared representation learning module where both images and text passages are mapped into a unified vector space using a CLIP-style vision tower and a fine-tuned Glot-500 text encoder, respectively. For any given query, whether text or image-based, we leverage these embeddings to perform an efficient similarity search against a FAISS index of our entire culinary text corpus. The top-ranked documents are then pooled to form an evidence block, which is injected directly into the prompt for the generative model to produce a contextually grounded answer. A key innovation in our methodology is an ingredient-aware training strategy; by treating images of a dish's main ingredients as additional positive examples during fine-tuning, we encourage the model to develop a more holistic concept of each dish, thereby enhancing the retriever's robustness, especially for ingredient-based queries.

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


## Evaluation and Results
We evaluated our system on a custom set of 50 multiple-choice questions (30 text-only, 20 image-based). Our key finding was the differential impact of RAG on generators of varying strengths.


