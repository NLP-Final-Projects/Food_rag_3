import pandas as pd
import torch
from config import Config
from encoders import Glot500Encoder, FaTextEncoder, FaVisionEncoder
from retrievers import Glot500Retriever, TextIndexRetriever

class RAGSystem:
    def __init__(self, cfg: Config):
        self.docstore = pd.read_parquet(cfg.docstore_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glot_enc = Glot500Encoder(cfg.glot_model_hf)
        self.glot_ret = Glot500Retriever(self.glot_enc, self.docstore, cfg.glot_index_out)
        txt_enc = FaTextEncoder(cfg.mclip_text_model_hf, device, cfg.max_text_len)
        self.mclip_ret = TextIndexRetriever(txt_enc, self.docstore, cfg.clip_index_out)
        self.vision = FaVisionEncoder(cfg.clip_vision_model, device)
