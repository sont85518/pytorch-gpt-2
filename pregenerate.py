from argparse import ArgumentParser
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
from tqdm import tqdm

import os
import re
import logging
import json
import string

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def tokenize_and_encode(output_dir, output_text_dir):
    list_files = os.listdir(output_text_dir)
    max_len = 0
    file_id = 0
    for f in list_files:
        logger.info("Load data from {}...".format(f))
        f_tmp = open(os.path.join(output_text_dir, f), 'r', encoding='utf-8')
        file_sample = []
        obj = json.load(f_tmp)
        for o in tqdm(obj, desc="Tokenize and encode dataset: ", unit=" sample"):
            if isinstance(o, str):
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(o))
            else:
                print(o)
                continue
            max_len = max(len(ids) + 3, max_len)
            #file_sample.append(ids)
        f_tmp.close()
        #if not len(file_sample) == 0:
        #    write_data(output_dir, file_sample, file_id)
        #    file_id += 1
    return max_len

def write_text_data(dataset, num_file_lines, output_path):
    for count, idx in tqdm(enumerate(range(0, len(dataset), num_file_lines))):
        file_name = "train_{}.txt".format(count)
        file = open(os.path.join(output_path, file_name), 'w', encoding='utf-8')
        file_data = dataset[idx:idx + num_file_lines]
        logger.info("Write dataset to {}:".format(file_name))
        json.dump(file_data, file)
        file.close()
    logger.info("Write data finished!")

def load_data(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    f = open(dataset_path, encoding='utf_8').readlines()
    output = []
    for line in tqdm(f, desc="Reading dataset: ", unit=" line"):
        if line.strip() == "":
            continue
        output.append(line.strip())
    logger.info("Sort dataset...")
    output.sort(key=len, reverse=True)
    return output


def write_data(output_path, dataset, file_id):
    file_name = "train_{}.txt".format(file_id)
    f = open(os.path.join(output_path, file_name), 'w', encoding='utf-8')
    logger.info("Write dataset to {}:".format(file_name))
    json.dump(dataset, f)
    f.close()
    logger.info("Write data finished!")


def write_data_config(output_path, config):
    with open(os.path.join(output_path, "dataset_config.json"), "w", encoding="utf-8") as f_config:
        json.dump(config, f_config)
        f_config.close()
    logger.info("Write config finished!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument('--train_corpus', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--output_text_dir', type=Path, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--num_file_lines', type=int, default=500000)
    parser.add_argument('--reduce_memory', action='store_true',
                        help='Reduce memory usage for large datasets by keeping data on disc rather than in memory')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='The number of workers to use to write the files')
    parser.add_argument('--epochs_to_generate', type=int, default=3,
                        help='Number of epochs of data to pregenerate')

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError('Cannot use multiple workers while reducing memory')
    logger.info("Loading model...")
    special_tokens = ['_start_']
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    logger.info("Encoding dataset...")
    train_dataset = load_data(args.train_corpus)
    #write_text_data(train_dataset, args.num_file_lines, args.output_text_dir)
    data_size = len(train_dataset)

    train_dataset=None
    input_length = tokenize_and_encode(args.output_dir, args.output_text_dir)

    # Compute the max input length for the Transformer
    logger.info("Preparing dataset...")
    max_length = model.config.n_positions // 2 - 2
    print(max_length)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    print(input_length)
    num_of_batch = data_size//args.train_batch_size
    print(num_of_batch)
    print(data_size)

    logger.info("Writing dataset...")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    config = {'max_length': max_length,
              'input_lenght': input_length,
              'special_tokens': special_tokens,
              'num_of_sample': data_size,
              'num_of_batch': num_of_batch}
    write_data_config(args.output_dir, config)
