import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# create logger
# module_logger = logging.getLogger('embeddings_benchmark.utils')

BPE_MODELS = ['gpt2', 'roberta']
RATING_IDX = 2
RATING_IDX_SIMLEX = 3


def iter_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def load_embeddings_benchmark(
        path: str,
        tokenizer,
) -> List[Tuple[int, int, float]]:
    """

    Args:
        path: path to dataset file
        tokenizer: an instance of `transformers.AutoTokenizer` from HuggingFace's Transformer

    Returns:
        List[Tuple[int, int, float]]: list of tuples of word indices and their corresponding rating.
    """
    # module_logger.info(f"Loading data from:\t{path}")
    rating_idx = RATING_IDX_SIMLEX if 'SimLex' in path else 2
    examples = []

    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            items = line.split()
            if "MEN" in path:
                items = [x.strip().split('-')[0] for x in items[:2]] + [items[2]]
            items = [x.strip() for x in items]
            if any([model_name in tokenizer.name_or_path
                    for model_name in BPE_MODELS]):
                # for tokenization consistency,
                # see https://github.com/huggingface/transformers/issues/1196 for details
                items = [" " + x for x in items]
            w1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(items[0]))
            w2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(items[1]))
            rating = float(items[rating_idx])
            if is_valid_example([w1, w2], tokenizer.unk_token_id):
                examples.append((w1, w2, rating))

    return examples


def is_valid_example(example: List[List[int]], unk_token_id: int) -> bool:
    """

    Args:
        example: list of token ids from transformers tokenizer
        unk_token_id: tokenizer specific unknown token id

    Returns:
        bool: True if each word is tokenized as a single token
            (is not split into sub-word tokens by tokenizer) and is not an unknown token
            else False
    """
    if len(example[0]) == 1 and len(example[1]) == 1 \
            and not any(unk_token_id in x for x in example):
        return True
    return False


def lookup_embeddings(
        embeddings: np.ndarray,
        examples: List[Tuple[int, int, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        embeddings:
        examples:

    Returns:
        (np.ndarray, np.ndarray): embedded examples
    """
    left = np.squeeze(embeddings[np.array([x[0] for x in examples])])
    right = np.squeeze(embeddings[np.array([x[1] for x in examples])])
    return left, right


def inner_product_similarity(embeddings: np.ndarray, examples: List[Tuple[int, int, float]]) -> np.ndarray:
    left, right = lookup_embeddings(embeddings, examples)
    # module_logger.debug(f"left shape: {left.shape}; right shape: {right.shape};")
    # below is equivalent to np.diag(np.matmul(left, right.T))
    return np.einsum('ij,ij->i', left, right)


def cosine_similarity(embeddings: np.ndarray, examples: List[Tuple[int, int, float]]) -> np.ndarray:
    left, right = lookup_embeddings(embeddings, examples)
    left_norms = np.linalg.norm(left, axis=1)
    right_norms = np.linalg.norm(right, axis=1)
    return np.einsum('ij,ij->i', left, right) / (left_norms * right_norms)


def eval_pearsonr(scores, examples) -> Tuple[float, float]:
    gold = np.array([e[2] for e in examples])
    cor, p = pearsonr(scores, gold)
    return cor.item(), p.item()


def to_latex_table(path_to_json: str, path_to_latex: str):
    with open(path_to_json, "r") as fp:
        results_dict = json.load(fp)
    df = pd.DataFrame.from_records(results_dict)
    with open(path_to_latex, 'w') as tf:
        tf.write(df.to_latex(float_format=lambda x: '%10.2f' % x))
