import torch

class Config:
    per_option_ctx: int = 5
    max_text_len: int = 512
    docstore_path: str = "indexes/docstore.parquet"
    glot_model_hf: str = "Arshiaizd/Glot500-FineTuned"
    mclip_text_model_hf: str = "Arshiaizd/MCLIP_FA_FineTuned"
    clip_vision_model: str = "SajjadAyoubi/clip-fa-vision"
    glot_index_out: str = "indexes/I_glot_text_fa.index"
    clip_index_out: str = "indexes/I_clip_text_fa.index"

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
