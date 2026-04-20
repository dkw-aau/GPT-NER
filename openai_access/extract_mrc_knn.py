from simcse import SimCSE
import argparse
import json
import numpy as np
import os
import faiss
import random


def read_feature(dir_, prefix):
    info_file = json.load(open(os.path.join(dir_, f"{prefix}.start_word_feature_info.json")))
    features = np.memmap(os.path.join(dir_, f"{prefix}.start_word_feature.npy"), 
                         dtype=np.float32,
                         mode="r",
                         shape=(info_file["entity_num"], info_file["hidden_size"]))
    index_file = []
    file = open(os.path.join(dir_, f"{prefix}.start_word_feature_index.json"), "r")
    for line in file:
        index_file.append(int(line.strip()))
    file.close()
    return info_file, features, index_file

def read_mrc_data(dir_, prefix):
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

def compute_mrc_knn(test_info, test_features, train_info, train_features, train_index, knn_num=32):
    quantizer = faiss.IndexFlatIP(train_info["hidden_size"])
    index = quantizer
    index.add(train_features.astype(np.float32))
    # 10 is a default setting in simcse
    index.nprobe = min(10, train_info["entity_num"])

    actual_knn = min(knn_num, index.ntotal)
    top_value, top_index = index.search(test_features.astype(np.float32), actual_knn)

    sum_ = 0
    vis_index = {}
    for idx_, value in enumerate(train_index):
        if value == 0:
            continue
        for i in range(sum_, value+sum_):
            vis_index[i] = idx_
        sum_ += value

    example_idx = [[vis_index[int(i)] for i in top_index[idx_]] for idx_ in range(test_info["entity_num"])]
    example_value = [[float(value) for value in top_value[idx_]] for idx_ in range(test_info["entity_num"])]

    return example_idx, example_value

def compute_simcse_knn(test_mrc_data, train_mrc_data, knn_num, test_index=None,
                       model_name="princeton-nlp/sup-simcse-roberta-large"):
    sim_model = SimCSE(model_name)

    train_sentence = {}
    train_sentence_index = {}
    for idx_, item in enumerate(train_mrc_data):
        label = item["entity_label"]
        context = item["context"]
        # if len(item["start_position"]) == 0:
        #     if label not in train_sentence:
        #         train_sentence[label] = []
        #         train_sentence_index[label] = []
        #     train_sentence[label].append(context)
        #     train_sentence_index[label].append(idx_)
        if label not in train_sentence:
            train_sentence[label] = []
            train_sentence_index[label] = []
        train_sentence[label].append(context)
        train_sentence_index[label].append(idx_)
    
    train_index = {}
    for key, _ in train_sentence.items():
        embeddings = sim_model.encode(train_sentence[key], batch_size=128, normalize_to_unit=True, return_numpy=True)
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        index = quantizer
        index.add(embeddings.astype(np.float32))
        # 10 is a default setting in simcse
        index.nprobe = min(10, len(train_sentence[key]))

        train_index[key] = index

    example_idx = []
    example_value = []

    if test_index is None:
        for idx_ in range(len(test_mrc_data)):
            context = test_mrc_data[idx_]["context"]
            label = test_mrc_data[idx_]["entity_label"]

            actual_knn = min(knn_num, train_index[label].ntotal)
            embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
            top_value, top_index = train_index[label].search(embedding.astype(np.float32), actual_knn)

            example_idx.append([train_sentence_index[label][int(i)] for i in top_index[0]])
            example_value.append([float(value) for value in top_value[0]])
        
        return example_idx, example_value

    for idx_, sub_index in enumerate(test_index):
        if sub_index != 0:
            continue
        context = test_mrc_data[idx_]["context"]
        label = test_mrc_data[idx_]["entity_label"]

        actual_knn = min(knn_num, train_index[label].ntotal)
        embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
        top_value, top_index = train_index[label].search(embedding.astype(np.float32), actual_knn)

        example_idx.append([train_sentence_index[label][int(i)] for i in top_index[0]])
        example_value.append([float(value) for value in top_value[0]])
    
    return example_idx, example_value

def combine_full_knn(test_index, mrc_knn_index, simcse_knn_index):
    results = []
    mrc_idx = 0
    simcse_idx = 0
    for idx_, num in enumerate(test_index):
        if num == 0:
            results.append(simcse_knn_index[simcse_idx])
            simcse_idx += 1
        else:
            knn_num = len(mrc_knn_index[mrc_idx])
            span_ = int(knn_num // num)
            if span_ * num != knn_num:
                span_ += 1
            sub_results = []
            for sub_idx in range(mrc_idx, mrc_idx+num):
                sub_results = sub_results + mrc_knn_index[sub_idx][:span_]
            sub_results = sub_results[:knn_num]
            results.append(sub_results)
            mrc_idx += num
    
    return results

def random_knn(test_mrc_data, train_mrc_data, knn_num):
    train_sentence = {}
    train_sentence_index = {}
    for idx_, item in enumerate(train_mrc_data):
        label = item["entity_label"]
        context = item["context"]

        if label not in train_sentence:
            train_sentence[label] = []
            train_sentence_index[label] = []
        train_sentence[label].append(context)
        train_sentence_index[label].append(idx_)

    example_idx = []

    for idx_ in range(len(test_mrc_data)):
        context = test_mrc_data[idx_]["context"]
        label = test_mrc_data[idx_]["entity_label"]

        random.shuffle(train_sentence_index[label])

        example_idx.append(train_sentence_index[label][:knn_num])
    
    return example_idx, None

def write_file(dir_, data):
    file = open(dir_, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False)+'\n')
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute SimCSE KNN for MRC data')
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Directory containing the MRC data files')
    parser.add_argument('--test-name', type=str, default='test',
                        help='Filename suffix for the test split (default: test)')
    parser.add_argument('--train-name', type=str, default='train',
                        help='Filename suffix for the train split (default: train)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to write the output KNN index file (.jsonl)')
    parser.add_argument('--knn-num', type=int, default=32,
                        help='Number of nearest neighbours to retrieve (default: 32)')
    parser.add_argument('--model-name', type=str,
                        default='princeton-nlp/sup-simcse-roberta-large',
                        help='SimCSE model name or local path')
    args = parser.parse_args()

    test_mrc_data = read_mrc_data(dir_=args.source_dir, prefix=args.test_name)
    train_mrc_data = read_mrc_data(dir_=args.source_dir, prefix=args.train_name)
    index_, value_ = compute_simcse_knn(
        test_mrc_data=test_mrc_data,
        train_mrc_data=train_mrc_data,
        knn_num=args.knn_num,
        model_name=args.model_name,
    )
    write_file(dir_=args.output, data=index_)