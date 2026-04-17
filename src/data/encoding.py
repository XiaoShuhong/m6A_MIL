from __future__ import annotations
 
import numpy as np
 
class SequenceEncoder:

    def __init__(
        self,
        method: str = "dnabert2",
        seq_len: int = 501,
        max_tokens: int = 256,
    ):
        self.method = method
        self.seq_len = seq_len
        self.max_tokens = max_tokens
 
        if method == "dnabert2":
            self._tokenizer = None  # 延迟加载
        elif method == "onehot":
            pass
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    @property
    def output_type(self) -> str:
        """返回编码输出的类型标识, 供 collate_fn 判断."""
        return self.method

    def encode_single(self, seq: str) -> dict | np.ndarray:
       
        if self.method == "dnabert2":
            return self._dnabert2_encode_single(seq)
        else:
            return _one_hot_encode(seq)
        
    def encode_batch(self, sequences: list[str]) -> dict | np.ndarray:

        if self.method == "dnabert2":
            return self._dnabert2_encode_batch(sequences)
        else:
            return np.stack([_one_hot_encode(s) for s in sequences], axis=0)
        
    
    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True,
                
            )
        
        return self._tokenizer
    
    def _dnabert2_encode_single(self, seq: str) -> dict:
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="np",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0).astype(np.int64),
            "attention_mask": tokens["attention_mask"].squeeze(0).astype(np.int64),
        }
    def _dnabert2_encode_batch(self, sequences: list[str]) -> dict:
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="np",
        )
        return {
            "input_ids": tokens["input_ids"].astype(np.int64),          # (N, L)
            "attention_mask": tokens["attention_mask"].astype(np.int64), # (N, L)
        }
 

 
_BASE_TO_IDX = np.full(128, -1, dtype=np.int8)
_BASE_TO_IDX[65] = 0   # A
_BASE_TO_IDX[97] = 0   # a
_BASE_TO_IDX[84] = 1   # T
_BASE_TO_IDX[116] = 1  # t
_BASE_TO_IDX[67] = 2   # C
_BASE_TO_IDX[99] = 2   # c
_BASE_TO_IDX[71] = 3   # G
_BASE_TO_IDX[103] = 3  # g
 
 
def _one_hot_encode(seq: str) -> np.ndarray:
    """
    A=[1,0,0,0], T=[0,1,0,0], C=[0,0,1,0], G=[0,0,0,1], N=[0,0,0,0]
    Returns: (len(seq), 4) float32
    """
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    indices = _BASE_TO_IDX[seq_bytes]
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    valid = indices >= 0
    arr[valid, indices[valid]] = 1.0
    return arr