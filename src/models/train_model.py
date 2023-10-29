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


class DeToxificationDataset(Dataset):
    """
    Dataset class for DeToxification text data.

    Args:
        dataframe (pd.DataFrame): Pandas DataFrame containing text data.
        tokenizer (ModelTokenizer): Hugging Face Transformers tokenizer.
        max_length (int): Maximum sequence length.
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: ModelTokenizer,
                 max_length: int):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reference = self.dataframe['reference'].values
        self.translation = self.dataframe['translation'].values

    def __getitem__(self, index) -> transformers.BatchEncoding:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            transformers.BatchEncoding: Encoded model inputs.
        """
        inputs = self.reference[index]
        targets = self.translation[index]
        model_inputs = self.tokenizer.__call__(inputs, max_length=self.max_length, truncation=True)
        labels = self.tokenizer.__call__(targets, max_length=self.max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __len__(self) -> int:
        return len(self.dataframe)


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


def load_data(data_path: Path) -> pd.DataFrame:
    print('Loading raw data...')
    raw_data = pd.read_csv(data_path, sep='\t', index_col=False)
    raw_data = raw_data[raw_data.columns[1:]]
    return raw_data


def to_train_and_val(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation datasets based on toxicity thresholds.

    Args:
        raw_data (pd.DataFrame): Raw data in a Pandas DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """
    print('Split data to train & val datasets')
    raw_data = raw_data[
        (raw_data['ref_tox'] >= MIN_TOX) &
        (raw_data['trn_tox'] <= 1 - MIN_TOX)
    ]

    raw_data['id'] = pd.RangeIndex(0, len(raw_data))
    train_split, val_split = train_test_split(
        range(raw_data[raw_data['id'] < AMOUNT_OF_PAIRS]['id'].max() + 1),
        test_size=VAL_RATIO,
        random_state=420
    )
    train_dataframe = raw_data[raw_data['id'].isin(train_split)]
    val_dataframe = raw_data[raw_data['id'].isin(val_split)]

    return train_dataframe, val_dataframe


# simple postprocessing for text
def postprocess_text(preds: list[str], labels: list[str]):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def train(data_path: Path, model_cktp_path: Path, batch_size: Optional[int], epochs: Optional[int]):
    """
    Finetune a sequence-to-sequence model.

    Args:
        data_path (Path): Path to the TSV data file.
        model_cktp_path (Path): Path to save the trained model.
        batch_size (Optional[int]): Batch size for training.
        epochs (Optional[int]): Number of training epochs.
    """
    # Set values
    batch_size = batch_size or BATCH_SIZE
    epochs = epochs or BATCH_SIZE
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    transformers.set_seed(42)
    raw_data = load_data(data_path)
    train_dataframe, val_dataframe = to_train_and_val(raw_data)

    print('Creating dataset')
    model_tokenizer: ModelTokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_dataset = DeToxificationDataset(train_dataframe, model_tokenizer, MAX_LENGTH)
    val_dataset = DeToxificationDataset(val_dataframe, model_tokenizer, MAX_LENGTH)

    print('Loading tokenizer & model')
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    args = transformers.Seq2SeqTrainingArguments(
        f"{MODEL_CHECKPOINT}-finetuned-de-toxification",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        report_to=['tensorboard'],
        seed=SEED,
    )
    data_collator = transformers.DataCollatorForSeq2Seq(model_tokenizer, model=model)
    metric = load_metric("sacrebleu")

    # compute metrics function to pass to trainer
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = model_tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, model_tokenizer.pad_token_id)
        decoded_labels = model_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != model_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    print('Train...')
    # instead of writing train loop we will use Seq2SeqTrainer
    trainer = transformers.Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=model_tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # saving model
    print('Save model...')
    trainer.save_model(model_cktp_path.as_posix())
    print('Done!')


def main():
    args = [] if len(sys.argv) == 1 else sys.argv[1:]
    batch_size = None
    epochs = None

    if '--help' in args and len(args) == 1:
        print('Usage: python3 train_model.py --batch_size=<optional int> --epochs=<optional int>')
        exit(0)

    for arg in args:
        batch_res = re.search(r'--batch_size=[0-9]+', arg)
        epochs_res = re.search(r'--epochs=[0-9]+', arg)

        if all(x is None for x in [batch_res, epochs_res]):
            print(f'Unknown parameter: {arg}')
            exit(1)

        if batch_size is not None and batch_res:
            print('Repeatable parameter --batch_size')
            exit(1)
        if epochs is not None and epochs_res:
            print('Repeatable parameter --epochs')
            exit(1)

        if batch_res:
            batch_size = int(arg[len('--batch_size='):])
        if epochs_res:
            epochs = int(arg[len('--epochs='):])

    root_path = get_root_path()
    data_path = root_path / Path('data/raw/filtered.tsv')
    model_cktp_path = root_path / 'models' / 'pretrained.pt'

    train(data_path, model_cktp_path, batch_size, epochs)


if __name__ == '__main__':
    main()
