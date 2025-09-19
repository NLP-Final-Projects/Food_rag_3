from typing import List, Tuple
import pandas as pd

class Utils:
    @staticmethod
    def build_context_block(hits: List[Tuple[int, float]], docstore: pd.DataFrame, count: int, max_chars=350) -> str:
        if not hits:
            return "No relevant documents found."
        lines = []
        for i, score in hits[:count]:
            row = docstore.iloc[i]
            txt = str(row["passage_text"])
            txt = (txt[:max_chars] + "…") if len(txt) > max_chars else txt
            lines.append(f"- [doc:{row['id']}] {txt}")
        return "\n".join(lines)
