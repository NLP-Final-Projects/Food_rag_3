from typing import List
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class Glot500Encoder:
    def __init__(self, model_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.st_model = SentenceTransformer(model_id, device=str(self.device))

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.st_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

class FaTextEncoder:
    def __init__(self, model_id: str, device: torch.device, max_len: int):
        self.device, self.max_len = device, max_len
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode_numpy(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            toks = self.tok(
                texts[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**toks)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                x = out.pooler_output
            else:
                x = (out.last_hidden_state * toks.attention_mask.unsqueeze(-1)).sum(1) / toks.attention_mask.sum(1).clamp(min=1)
            x_norm = x / x.norm(p=2, dim=1, keepdim=True)
            vecs.append(x_norm.detach().cpu().numpy())
        return np.vstack(vecs).astype(np.float32)

class FaVisionEncoder:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode(self, img: Image.Image) -> np.ndarray:
        img = ImageOps.exif_transpose(img).convert("RGB")
        batch = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**batch)
        v = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
        v_norm = v / v.norm(p=2, dim=1, keepdim=True)
        return v_norm[0].detach().cpu().numpy().astype(np.float32)
