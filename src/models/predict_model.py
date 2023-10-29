import warnings
import re
import sys
import os
import transformers
import numpy as np
import pandas as pd

from typing import Optional
from pathlib import Path
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# ignore warnings
warnings.filterwarnings('ignore')

AMOUNT_OF_PAIRS = 50000
EPOCHS = 1
VAL_RATIO = 0.1
MAX_LENGTH = 75
MIN_TOX = 0.75
MODEL_CHECKPOINT = "t5-small"
BATCH_SIZE = 25
SEED = 432

# Support types
ModelTokenizer = transformers.models.t5.tokenization_t5_fast.T5TokenizerFast
ModelType = transformers.models.t5.modeling_t5.T5ForConditionalGeneration


def get_root_path() -> Path:
    """
    Get the project's root path.

    Returns:
        Path: Path to the project's root directory.
    """
    candidates = ['../..', '..', '.']
    for candidate in candidates:
        readme_candidate = Path(candidate) / 'README.md'
        readme_candidate = readme_candidate.resolve()
        if not readme_candidate.exists():
            continue
        with open(readme_candidate) as readme:
            if not readme.readline().startswith('## PMLDL ASSIGNMENT 1'):
                continue
            return Path(candidate).resolve()
    raise FileNotFoundError(f'Run {__file__} from project root or src/data or src')


def de_toxification(tr_model: ModelType, inference_request: str, tokenizer: ModelTokenizer) -> str:
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = tr_model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)


def start_loop(model_cktp_path: Path):
    # loading the model and run inference for it
    print('Loading...')
    model_tokenizer: ModelTokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_cktp_path)
    model.eval()
    model.config.use_cache = False

    print('Print text, use /q to exit')
    while True:
        print('> ', end='')
        text = input()

        if text == '/q':
            break

        res = de_toxification(model, text, model_tokenizer)
        print(f'> {res}')


def main():
    root_path = get_root_path()
    model_cktp_path = root_path / 'models' / 'pretrained.pt'

    start_loop(model_cktp_path)
    print('Buy...')


if __name__ == '__main__':
    main()
