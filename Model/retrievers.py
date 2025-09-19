from typing import List, Tuple
import os
import numpy as np
import pandas as pd
import faiss
from encoders import Glot500Encoder, FaTextEncoder

class BaseRetriever:
    def __init__(self, docstore: pd.DataFrame, index_path: str):
        self.docstore, self.index_path = docstore.reset_index(drop=True), index_path
        if os.path.isfile(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            raise FileNotFoundError(f"Index file not found at {self.index_path}.")

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        D, I = self.index.search(query_vec[None, :].astype(np.float32), k)
        return list(zip(I[0].tolist(), D[0].tolist()))

class Glot500Retriever(BaseRetriever):
    def __init__(self, encoder: Glot500Encoder, docstore: pd.DataFrame, index_path: str):
        super().__init__(docstore, index_path)
        self.encoder = encoder

    def topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        qv = self.encoder.encode([query], batch_size=1)[0]
        return self.search(qv, k)

class TextIndexRetriever(BaseRetriever):
    def __init__(self, text_encoder: FaTextEncoder, docstore: pd.DataFrame, index_path: str):
        super().__init__(docstore, index_path)
        self.encoder = text_encoder
