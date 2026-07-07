from dataclasses import dataclass 
from typing import Tuple

@dataclass
class ModelConfig:
    model_dim: int = 1280
    num_layers: int = 20
    num_heads: int = 16
    patch_len: int = 32
    horizon_len: int = 128
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    dropout: float = 0.0
    rope_max_period: float = 10000.0
    rms_eps: float = 1e-6
    @property
    def head_dim(self) -> int:
        assert self.model_dim % self.num_heads == 0
        return self.model_dim // self.num_heads
    @property
    def num_outputs(self) -> int:
        return 1 + len(self.quantiles)  # mean (point) + quantiles