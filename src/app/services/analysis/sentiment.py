from sentimentpl.models import SentimentPLModel
import sacremoses # this import is needed for the inner workings of sentimentpl

from contextlib import ExitStack
import importlib
import torch
from torch import nn
from transformers import XLMTokenizer, RobertaModel
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import io

from data import trained_models

class SentimentAnalyzer(SentimentPLModel):
    def __init__(self, filename: str):
        super().__init__()
        f = io.BytesIO(importlib.resources.read_binary(trained_models, filename))
        self.fc.load_state_dict(torch.load(f))
        self.eval()

    def analyze(self, text) -> float:
        result = self(text)
        return result.item()

