# CLIP Fine-Tuning for Persian Food VQA

This project contains the code to fine-tune a CLIP-based text encoder on a dataset of Persian food images and their corresponding descriptions. The goal is to create a model that can effectively embed both images and text into a shared representation space for retrieval tasks.

## Project Structure

- `config.py`: A centralized file for all configurations, including file paths, model names, and training hyperparameters.
- `train.py`: The main entry point to start the data preparation and model fine-tuning process.
- `src/`: Contains the core source code for the model and training logic.
  - `dataset.py`: Defines the PyTorch `Dataset` for loading image-text pairs.
  - `encoders.py`: Contains wrapper classes for the CLIP vision and text models.
  - `trainer.py`: The main `FaCLIPTextTrainer` class that orchestrates the training loop.
- `utils/`: Contains utility scripts, primarily for data processing.
  - `data_prep.py`: Functions to process the raw JSON/image data and build the `docstore.parquet`.

## How to Run

1.  **Prepare Your Data**:
    - Place your raw food dataset (the folders containing images and `.json` files) into a single directory.

2.  **Install Dependencies**:
    ```bash
    pip install torch pandas transformers faiss-cpu pillow tqdm
    ```

3.  **Configure the Project**:
    - Open `config.py`.
    - Set the `base_data_root` variable to the path of the directory from Step 1.
    - (Optional) Adjust other hyperparameters like `epochs`, `batch_size`, etc., as needed.

4.  **Run Training**:
    - Execute the main training script from your terminal:
    ```bash
    python train.py
    ```
    - You can also override the settings from `config.py` using command-line arguments for quick experiments:
    ```bash
    python train.py --epochs 5 --learning_rate 2e-5 --output_dir "models/my_finetuned_clip"
    ```

This process will first scan and process your raw data to create a `docstore.parquet` and a training CSV file in the `data/` directory. It will then proceed with fine-tuning the model. The final model, tokenizer, and the corresponding FAISS index will be saved to the directory specified by `output_dir` in your configuration.
