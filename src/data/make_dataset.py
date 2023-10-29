import zipfile
import time
import pandas as pd
import torchtext
import sys

from pathlib import Path
from typing import Callable, Optional
from download_from_url import download_url
from transformers import BertTokenizer
from tqdm import tqdm

RAW_ZIP_NAME = 'filtered_paranmt.zip'
RAW_URL_FILE = 'filtered_url.txt'
RAW_FILE_NAME = 'filtered.tsv'
INTERMEDIATE_FILE_NAME = 'intermediate.tsv'
BERT_FILE_NAME = 'bert.tsv'


def get_root_path() -> Path:
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


def get_raw_url(data_path: Path) -> str:
    raw_url_file = data_path / 'raw' / RAW_URL_FILE
    with open(raw_url_file) as url_file:
        return url_file.readline()


def download_raw(raw_url: str, save_dir: Path, filename: str = RAW_ZIP_NAME) -> Path:
    print('Downloading data')
    time.sleep(0.5)
    download_url(raw_url, save_dir / filename)
    return save_dir / filename


def unzip_raw(raw_zip: Path, path_to: Path):
    print(f'Extracting {raw_zip.name}')
    with zipfile.ZipFile(raw_zip, 'r') as zip_ref:
        zip_ref.extractall(path_to)


def check_presence_of_raw_data(root_path: Path, force_rewrite: bool = False) -> Path:
    data_path = root_path / 'data'

    raw_path = data_path / 'raw'
    raw_data_file = raw_path / RAW_FILE_NAME
    raw_zip = root_path / RAW_ZIP_NAME
    raw_url = get_raw_url(data_path)

    if raw_data_file.exists() and not force_rewrite:
        print(f'{raw_data_file.name} is present')
        return raw_data_file

    if raw_zip.exists() and not force_rewrite:
        print(f'{raw_zip.name} is present')
        unzip_raw(raw_zip, raw_path)
        return raw_data_file

    download_raw(raw_url, root_path)
    unzip_raw(raw_zip, raw_path)
    return raw_data_file


def apply_tokenizer(raw: pd.DataFrame, tokenizer: Callable) -> pd.DataFrame:
    refs = raw['reference'].values
    trns = raw['translation'].values

    new_refs = []
    new_trns = []

    for data in tqdm(zip(refs, trns), total=len(refs)):
        new_refs.append(tokenizer(data[0]))
        new_trns.append(tokenizer(data[1]))

    raw['reference'] = new_refs
    raw['translation'] = new_trns
    return raw


def to_intermediate(raw_data_file: Path,
                    root_path: Path,
                    force_rewrite: bool = False,
                    tokenizer: Optional[Callable] = None,
                    intermediate_file_name: Optional[str] = None):
    if tokenizer is None:
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    if intermediate_file_name is None:
        intermediate_file_name = INTERMEDIATE_FILE_NAME

    intermediate_path = root_path / 'data' / 'interim' / intermediate_file_name
    if intermediate_path.exists() and not force_rewrite:
        print(f'{intermediate_path.name} is present')
        return

    print('Constructing intermediate representation...')
    raw = pd.read_csv(raw_data_file, delimiter='\t')
    raw = raw[raw.columns[1:]]

    # Lower
    print('To lower')
    raw['reference'] = raw['reference'].str.lower()
    raw['translation'] = raw['translation'].str.lower()

    # Remove garbage
    print('Removing garbage symbols')
    garbage = [
        '^', '&', '*', '%', '@', '#', '$', '_', '+', '-', '=', '>', '<', ':', '~'
    ]
    for symbol in garbage:
        raw['reference'] = raw['reference'].str.replace(symbol, '')
        raw['translation'] = raw['translation'].str.replace(symbol, '')
        print(f"'{symbol}' has been removed")

    raw['reference'] = raw['reference'].str.replace('...', '.')
    raw['translation'] = raw['translation'].str.replace('...', '.')
    print("Triple dots has been replaced by single ones")

    # Tokenize
    print('Tokenize')
    # raw['reference'] = raw['reference'].apply(tokenizer)
    # raw['translation'] = raw['translation'].apply(tokenizer)
    raw = apply_tokenizer(raw, tokenizer)

    raw.to_csv(intermediate_path, sep='\t', index=False)
    print('Done')


def main():
    args = [] if len(sys.argv) == 1 else sys.argv[1:]
    force_rewrite = False
    tokenizer = None
    intermediate_file_name = None

    if '--rewrite' in args:
        force_rewrite = True

    if '--bert' in args:
        intermediate_file_name = BERT_FILE_NAME
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = lambda x: bert_tokenizer.tokenize(x)

    root_path = get_root_path()
    raw_data_file = check_presence_of_raw_data(root_path, force_rewrite)
    to_intermediate(raw_data_file, root_path, force_rewrite, tokenizer, intermediate_file_name)


if __name__ == '__main__':
    main()
