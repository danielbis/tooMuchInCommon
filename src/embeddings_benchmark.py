import argparse
import os
from collections import OrderedDict
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

from post_processing import all_but_the_top, remove_mean, identity
from utils import (
    iter_files,
    load_embeddings_benchmark,
    cosine_similarity,
    inner_product_similarity,
    eval_pearsonr
)

MODELS = ['gpt2', 'gpt2-medium',
          'bert-base-cased', 'bert-large-cased',
          'roberta-base', 'roberta-large']

PATH_TO_DATA_DIR = "datasets/"
DATASET_PATH_MAP = OrderedDict(
    men=os.path.join(PATH_TO_DATA_DIR, 'MEN_dataset_lemma_form_full'),
    rw=os.path.join(PATH_TO_DATA_DIR, 'rw.txt'),
    simlex999=os.path.join(PATH_TO_DATA_DIR, 'SimLex-999.txt'),
    wordsim_sim=os.path.join(PATH_TO_DATA_DIR, 'wordsim_similarity_goldstandard.txt'),
    wordsim_rel=os.path.join(PATH_TO_DATA_DIR, 'wordsim_relatedness_goldstandard.txt')
)

POST_PROCESSING_FUNCS = [
    ('original', identity, {}),
    ('centered', remove_mean, {'scale': False}),
    ('centered_scaled', remove_mean, {'scale': True}),
    ('post_process', all_but_the_top, {'top_d': 2, 'use_fix': False}),
    ('post_process_fix', all_but_the_top, {'top_d': 2, 'use_fix': True})
]


def run_eval(embeddings, examples):
    inner_prod_cor, _ = eval_pearsonr(
        inner_product_similarity(embeddings, examples),
        examples
    )
    cosine_cor, _ = eval_pearsonr(
        cosine_similarity(embeddings, examples),
        examples
    )
    return round(inner_prod_cor * 100., 2), round(cosine_cor * 100., 2)


def _average(list2d):
    flat = [item for sublist in list2d for item in sublist]
    return (round(sum(flat[::2]) / float(len(flat[::2])), 2),
            round(sum(flat[1::2]) / float(len(flat[1::2])), 2))


def main(models_list, post_processing_funcs, dataset_path_map, out_path):
    results = {}
    for model_name in models_list:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embeddings = model.get_input_embeddings().weight.detach().numpy()

        for func_name, func, fargs in post_processing_funcs:
            print(f"{model_name}, {func_name}")
            fargs['embeddings'] = embeddings
            processed_embeddings = func(**fargs)
            row_dict = {}
            for dataset_name, dataset_path in dataset_path_map.items():
                examples = load_embeddings_benchmark(path=dataset_path, tokenizer=tokenizer)
                inner_prod_cor, cosine_cor = run_eval(
                    embeddings=processed_embeddings,
                    examples=examples
                )
                row_dict[dataset_name] = (cosine_cor, inner_prod_cor)
                print(f"{model_name:20}, {func_name:24}, {dataset_name:16},"
                      f" cosine_sim: {cosine_cor:6}; inner_prod: {inner_prod_cor:6}")
            row_dict['average'] = _average(row_dict.values())
            results[f"{model_name}+{func_name}"] = row_dict

    # write out
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=4)


if __name__ == "__main__":
    output_file = os.path.join(os.getcwd(),
                               'experiments/benchmarks/',
                               '{}.json'.format(datetime.now().strftime("%d%m%Y%H%M%S")))

    parser = argparse.ArgumentParser(description='Embeddings benchmarks.')
    parser.add_argument('--model', type=str, default='',
                        help='Name of a pre-trained model from HuggingFace. '
                             'If none provided, runs eval on each model in: {}'.format(' '.join(MODELS)))
    parser.add_argument('--output', type=str,
                        default='', help='Output json file to save the results. '
                                         'By default stored in ../experiments/benchmarks/dmYHMS.json')

    args = parser.parse_args()
    models = [args.model] if len(args.model) > 0 else MODELS
    output_file = args.output if len(args.output) > 0 else output_file

    main(
        models_list=models,
        post_processing_funcs=POST_PROCESSING_FUNCS,
        dataset_path_map=DATASET_PATH_MAP,
        out_path=output_file
    )
