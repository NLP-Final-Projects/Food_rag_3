# Food QA Dataset

This repository contains structured datasets for **food-related question answering** tasks.
It includes both **text-only data** and **image–text paired data**.

---

## 📂 Repository Structure

```
.
├── text_only/
│   └── passages_questions.json
│
├── image_text/
│   └── food_dataset.json
│
├── food_images_URL.txt
└── README.md
```

### `text_only/`

* **`passages_questions.json`**
  Each entry is a pair consisting of:

  * `passage`: a paragraph of text about food.
  * `question`: a question generated from the passage.

Example:

```json
{
  "passage": "باقلوای اردبیل یکی از شیرینی‌های سنتی اردبیل است...",
  "question": "باقلوای اردبیل مربوط به کدام شهر است؟"
}
```

---

### `image_text/`

* **`food_dataset.json`**
  Each entry contains:

  * `title`: the name of the food.
  * `passage`: a descriptive paragraph about the food.
  * `folder_path`: the relative path to a folder containing images of the food.

Example:

```json
{
  "title": "تهچین گرمسار",
  "passage": "تهچین گرمسار یکی از غذاهای سنتی ایران است...",
  "folder_path": "./final_foods/تهچین_گرمسار"
}
```

⚠️ **Note:**
Due to GitHub storage limits, only **one image folder** is included in this repo.
For the full dataset of food images, see **`food_images_URL.txt`**, which contains a Kaggle link to the complete image dataset.

---

## 📦 Data Access

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/food-qa-dataset.git
   cd food-qa-dataset
   ```

2. Download the complete food image dataset from Kaggle using the link in:

   ```
   food_images_URL.txt
   ```

3. Place the downloaded image folders in the same structure as referenced by the `folder_path` fields in `food_dataset.json`.

---

## 🔧 Use Cases

* **Text-only QA tasks** (e.g., passage–question answering, reading comprehension).
* **Multimodal QA tasks** (e.g., using both text and food images).
* **Dataset extensions** for food classification, image–text retrieval, or multimodal reasoning.

---

## 📜 License

This dataset is released for **research and educational purposes only**.
Please cite this repository if you use it in your work.

---
