import os
import sys
import yaml
from tqdm import tqdm
from base_provider import LLMProvider
from factory import get_provider
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random

random.seed(1)
logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-dir", type=str, help="directory for the input")
    parser.add_argument("--source-name", type=str, help="file name for the input")
    parser.add_argument("--train-name", type=str, default="None", help="file name for the training set")
    parser.add_argument("--data-name", type=str, help="dataset name for the input")
    parser.add_argument("--example-dir", type=str, default="None", help="directory for the example")
    parser.add_argument("--example-name", type=str, default="None", help="file name for the example")
    parser.add_argument("--example-num", type=int, default=16, help="numebr for examples")
    parser.add_argument("--last-results", type=str, default="None", help="unfinished file")
    parser.add_argument("--write-dir", type=str, help="directory for the output")
    parser.add_argument("--write-name", type=str, help="file name for the output")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="path to the LLM provider config file (default: config.yaml)")
    
    return parser

def read_mrc_data(dir_, prefix="test"):
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_results(dir_):
    file = open(dir_, "r")
    resulst = file.readlines()
    file.close()
    return resulst

def read_examples(dir_, prefix="dev"):
    print("reading ...")
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_idx(dir_, prefix="test"):
    print("reading ...")
    file_name = os.path.join(dir_, f"{prefix}.knn.jsonl")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx

def mrc2prompt(mrc_data, data_name="CONLL", example_idx=None, train_mrc_data=None, example_num=16, last_results=None):
    print("mrc2prompt ...")

    def get_example(index):
        exampel_prompt = ""
        for idx_ in example_idx[index][:example_num]:
            context = train_mrc_data[idx_]["context"]
            context_list = context.strip().split()
            labels = ""

            last_ = 0
            for span_idx in range(len(train_mrc_data[idx_]["start_position"])):
                start_ = train_mrc_data[idx_]["start_position"][span_idx]
                end_ = train_mrc_data[idx_]["end_position"][span_idx] + 1
                if labels != "":
                    labels += " "
                if last_ == start_:
                    labels += "@@" + " ".join(context_list[start_:end_]) + "##"
                else:
                    labels += " ".join(context_list[last_:start_]) + " @@" + " ".join(context_list[start_:end_]) + "##"
                last_ = end_

            if labels != "" and last_ != len(context_list):
                labels += " "
            labels += " ".join(context_list[last_:])

            exampel_prompt += f"The given sentence: {context}\n"
            exampel_prompt += f"The labeled sentence: {labels}\n"
        return exampel_prompt
        
    results = []
    for item_idx in tqdm(range(len(mrc_data))):

        if last_results is not None and last_results[item_idx].strip() != "FRIDAY-ERROR-ErrorType.unknown":
            continue

        item_ = mrc_data[item_idx]
        context = item_["context"]
        origin_label = item_["entity_label"]
        transfered_label, sub_prompt = FULL_DATA[data_name][origin_label]
        prompt_label_name = transfered_label[0].upper() + transfered_label[1:]
        prompt = f"You are an excellent linguist. Within the OntoNotes5.0 dataset, the task is to label {transfered_label} entities that {sub_prompt}. Below are some examples, and you should make the same prediction as the examples.\n"

        prompt += get_example(index=item_idx)

        prompt += f"The given sentence: {context}\nThe labeled sentence:"

        results.append(prompt)
    
    return results

def ner_access(llm_provider: LLMProvider, ner_pairs, batch=16):
    print("tagging ...")
    results = []
    start_ = 0
    pbar = tqdm(total=len(ner_pairs))
    while start_ < len(ner_pairs):
        end_ = min(start_+batch, len(ner_pairs))
        results = results + llm_provider.complete(ner_pairs[start_:end_])
        pbar.update(end_-start_)
        start_ = end_
    pbar.close()
    return results

def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(line.strip()+'\n')
    file.close()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Resolve config path relative to repo root (one level up from this script)
    config_path = args.config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        config_path = os.path.join(repo_root, config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    llm_provider = get_provider(config)

    ner_test = read_mrc_data(args.source_dir, prefix=args.source_name)
    mrc_train = read_mrc_data(dir_=args.source_dir, prefix=args.train_name)
    example_idx = read_idx(args.example_dir, args.example_name)

    last_results = None
    if args.last_results != "None":
        last_results = read_results(dir_=args.last_results)

    prompts = mrc2prompt(mrc_data=ner_test, data_name=args.data_name, example_idx=example_idx, train_mrc_data=mrc_train, example_num=args.example_num, last_results=last_results)
    results = ner_access(llm_provider=llm_provider, ner_pairs=prompts, batch=4)
    write_file(results, args.write_dir, args.write_name)
