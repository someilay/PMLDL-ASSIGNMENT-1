import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchtext

from typing import Optional
from ast import literal_eval
from pathlib import Path
from matplotlib.ticker import PercentFormatter
from wordcloud import WordCloud

INTERMEDIATE_FILE_NAME = 'intermediate.tsv'
RAW_FILE_NAME = 'filtered.tsv'
MIN_TOX = 0.75


def get_root_path() -> Path:
    """
    Get the root path of the project.

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


def plot_dist(values: pd.Series,
              title: str,
              x_label: str,
              step: float = 0.05,
              values_range: Optional[tuple[float, float]] = None,
              save_to: Optional[Path] = None):
    save_file = Path(title.lower().replace(' ', '_') + '.png')
    if save_to:
        save_file = save_to / save_file
    print(f'Creating {save_file.name}')
    if not values_range:
        values_range = (0, 1)
    values = values.to_numpy()
    bins = np.arange(values_range[0], values_range[1] + step, step)

    plt.figure(figsize=(12, 4))
    plt.axes().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.hist(values, bins=bins, weights=np.ones_like(values) / float(len(values)))
    plt.xticks(bins)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(save_file)


def plot_distributions(tox_data: pd.DataFrame, save_to: Path):
    plot_dist(tox_data.similarity, 'Similarity distribution', 'Similarity', save_to=save_to)
    plot_dist(
        tox_data.lenght_diff, 'Relative length difference distribution',
        'Relative length difference', save_to=save_to
    )
    plot_dist(
        tox_data.ref_tox, 'Reference toxicity distribution',
        'Reference toxicity', step=0.25, save_to=save_to
    )
    plot_dist(
        tox_data.trn_tox, 'Translation toxicity distribution',
        'Translation toxicity', step=0.25, save_to=save_to
    )

    ref_tokens = tox_data.reference.apply(len)
    ref_tokens_max = ref_tokens.max()
    ref_tokens_min = ref_tokens.min()
    ref_tokens_step = int((ref_tokens_max - ref_tokens_min) / 10)

    trn_tokens = tox_data.translation.apply(len)
    trn_tokens_max = trn_tokens.max()
    trn_tokens_min = trn_tokens.min()
    trn_tokens_step = int((trn_tokens_max - trn_tokens_min) / 10)

    plot_dist(
        ref_tokens, 'Reference amount of tokens distribution', 'Reference amount of tokens',
        step=ref_tokens_step, values_range=(ref_tokens_min, ref_tokens_max)
    )
    plot_dist(
        trn_tokens, 'Translation amount of tokens distribution', 'Translation amount of tokens',
        step=trn_tokens_step, values_range=(trn_tokens_min, trn_tokens_max)
    )


def main():
    root_path = get_root_path()
    intermediate_path = root_path / 'data' / 'interim' / INTERMEDIATE_FILE_NAME
    raw_path = root_path / 'data' / 'raw' / RAW_FILE_NAME
    vis_path = root_path / 'src' / 'visualization'

    print('Loading dataset...')
    raw_data = pd.read_csv(raw_path, sep='\t')
    tox_data = pd.read_csv(intermediate_path, sep='\t',
                           converters={'reference': literal_eval, 'translation': literal_eval})
    plot_distributions(tox_data, vis_path)

    word_cloud = WordCloud(
        max_font_size=75,
        max_words=250,
        background_color="white",
        width=1200,
        height=600,
    )
    raw_data = raw_data[
        (raw_data['ref_tox'] >= MIN_TOX) &
        (raw_data['trn_tox'] <= 1 - MIN_TOX)
    ]

    print('Generating world cloud for toxic sentences!')
    res: WordCloud = word_cloud.generate(' '.join(raw_data.reference.values.tolist()))
    res.to_file((vis_path / 'toxic_words_cloud.png').as_posix())

    print('Generating world cloud for usual sentences!')
    res: WordCloud = word_cloud.generate(' '.join(raw_data.translation.values.tolist()))
    res.to_file((vis_path / 'usual_words_cloud.png').as_posix())


if __name__ == '__main__':
    main()
