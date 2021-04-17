import argparse
import math
import json
import os
from datetime import datetime

import numpy as np
from transformers import AutoModel
from post_processing import all_but_the_top, remove_mean

np.random.seed(4)

MODELS = [
    'roberta-large', 'roberta-base',
    'bert-base-uncased', 'bert-large-uncased',
    'bert-base-cased', 'bert-large-cased',
    'gpt2', 'gpt2-medium'
]


def isotropy(embeddings: np.ndarray) -> float:
    """
    Computes isotropy score.
    Defined in Section 5.1, equations (7) and (8) of the paper.

    Args:
        embeddings: word vectors of shape (n_words, n_dimensions)

    Returns:
        float: isotropy score
    """
    min_z = math.inf
    max_z = -math.inf

    eigen_values, eigen_vectors = np.linalg.eig(np.matmul(embeddings.T, embeddings))
    for i in range(eigen_vectors.shape[1]):
        z_c = np.matmul(embeddings, np.expand_dims(eigen_vectors[:, i], 1))
        z_c = np.exp(z_c)
        z_c = np.sum(z_c)
        min_z = min(z_c, min_z)
        max_z = max(z_c, max_z)

    return round((min_z / max_z).item(), 4)


def run(models_list, out_path):
    results = {}

    for model_name in models_list:
        model = AutoModel.from_pretrained(model_name)
        embeddings = model.get_input_embeddings().weight.detach().numpy()
        d_ = math.ceil(embeddings.shape[1] / 100)

        # raw
        iso_raw = isotropy(embeddings=embeddings)
        # centered
        iso_centered = isotropy(embeddings=remove_mean(embeddings=embeddings, scale=False))
        # post-processed with d_ = embedding_dim / 100 as proposed in Mu et al.
        iso_post_processed = isotropy(embeddings=all_but_the_top(embeddings, top_d=d_, use_fix=True))

        results[model_name] = {
            "raw": iso_raw,
            "centered": iso_centered,
            "post_processed_fixed": iso_post_processed
        }

        print(f"{model_name}::\t"
              f"raw: {iso_raw};\t"
              f"centered: {iso_centered};\t"
              f"post_processed_fixed: {iso_post_processed}")

    # write out
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=4)


if __name__ == "__main__":
    output_file = os.path.join(os.getcwd(),
                               'experiments/isotropy/',
                               '{}.json'.format(datetime.now().strftime("%d%m%Y%H%M%S")))

    parser = argparse.ArgumentParser(description='Isotropy experiment.')
    parser.add_argument('--model', type=str, default='',
                        help='Name of a pre-trained model from HuggingFace. '
                             'If none provided, runs eval on each model in: {}'.format(' '.join(MODELS)))
    parser.add_argument('--output', type=str,
                        default='', help='Output json file to save the results. '
                                         'By default stored in ../experiments/isotropy/dmYHMS.json')
    args = parser.parse_args()
    models = [args.model] if len(args.model) > 0 else MODELS
    output_file = args.output if len(args.output) > 0 else output_file

    run(models, out_path=output_file)
