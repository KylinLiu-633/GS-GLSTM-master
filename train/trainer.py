import argparse
import sys
import json
import os
import random

import torch
from torch.utils import data
from utils import get_data_from_fof, get_data_from_file, padding_3d
from data_preprocessing import get_dataset_from_instances, collect_data
from data_preprocessing import word_mapping, char_mapping, edge_mapping
from network import GsGLstm


from allennlp.modules.elmo import Elmo, batch_to_ids

weights = "../data/pretrained_embedding/elmo_finetuned_matsci/elmo_weights.hdf5"
options = "../data/pretrained_embedding/elmo_finetuned_matsci/elmo_options.json"

# Set the pretrained embedder
embedder = Elmo(options, weights, 2, dropout=0)  # The word representation's dimension of EMLO is 1024


class Dataset(data.Dataset):

    # Set the property of Dataset

    def __init__(self, source_data):
        self.data = source_data
        self.num_total_seqs = len(self.data)

    def __getitem__(self, item):
        lemmas = self.data[item][0]
        lemmas_idx = self.data[item][1]
        lemmas_char_idx = self.data[item][2]
        in_node = self.data[item][3]
        in_label_idx = self.data[item][4]
        out_node = self.data[item][5]
        out_label_idx = self.data[item][6]
        entity_indexs = self.data[item][7]
        truth_tags = self.data[item][8]

        return lemmas, lemmas_idx, lemmas_char_idx, in_node, in_label_idx, out_node, out_label_idx, entity_indexs, truth_tags

    def __len__(self):
        return self.num_total_seqs


def collate_fn(data):
    #  lemmas, lemmas_idx, lemmas_char_idx, in_node, in_label_idx, out_node, out_label_idx, entity_indexs, truth_tags
    #  count the number of nodes and char_nodes
    def merge(datas):
        # print("datas", datas)
        lengths = [len(x) for x in datas]
        character_ids = batch_to_ids(datas)
        embeddings = embedder(character_ids)
        input_datas = embeddings['elmo_representations'][0].double()
        sentence_len = len(input_datas[0])
        return input_datas, lengths, sentence_len

    def merge_lemmas_id(datas, sequence_len=0):
        lengths = [len(x) for x in datas]
        padded_seqs = torch.zeros(len(datas), sequence_len).long()
        for i, seq in enumerate(datas):
            end = lengths[i]
            padded_seqs[i, :end] = torch.tensor(seq[:end])
        padded_seqs = torch.unsqueeze(padded_seqs, 2)
        return padded_seqs

    def get_mask_list(datas, batch, sequence_len=0, dim_3=0):
        mask_list = []
        for inst in datas:
            mask = []
            for item in inst:
                mask.append([1 for _ in item])
            mask_list.append(mask)
        mask_list = padding_3d(mask_list, batch, sequence_len, dim_3)
        return mask_list

    # data.sort(key=lambda x: len(x[0]), reverse=True)

    lemmas, lemmas_idx, lemmas_char_idx, in_node, in_label_idx, out_node, out_label_idx, entity_indexs, truth_tags = zip(*data)
    batch = len(lemmas)
    lemmas, node_num, sentence_len = merge(lemmas)
    lemmas_idx = merge_lemmas_id(lemmas_idx, sentence_len)

    # get the mask array for in_node, out_node, and entity
    in_node_mask = get_mask_list(in_node, batch, sentence_len)
    out_node_mask = get_mask_list(out_node, batch, sentence_len)
    entity_mask = get_mask_list(entity_indexs, batch)

    if lemmas_char_idx[0] is not None:
        lemmas_chars = padding_3d(lemmas_char_idx)
    else:
        lemmas_chars = None
    in_nodes = padding_3d(in_node).long()
    in_labels = padding_3d(in_label_idx).long()
    out_nodes = padding_3d(out_node).long()
    out_labels = padding_3d(out_label_idx).long()
    entity_indexs = padding_3d(entity_indexs)

    assert in_node_mask.shape == in_nodes.shape
    assert in_node_mask.shape == in_labels.shape
    assert out_node_mask.shape == out_nodes.shape
    assert out_node_mask.shape == out_labels.shape
    assert entity_mask.shape == entity_indexs.shape

    return node_num, lemmas, lemmas_idx, lemmas_chars, in_nodes, in_labels, out_nodes, out_labels, entity_indexs, \
           truth_tags, in_node_mask, out_node_mask, entity_mask


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_default_dtype(torch.double)

    # Get configuration

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config_file_song.json", help="The file path of configuration")

    options, unparsed = parser.parse_known_args()

    if options.config_path is not None:
        print("Loading the configuration from " + options.config_path)

        with open(options.config_path, "r") as f_in:
            config_dict = json.load(f_in)
            options.__dict__.update(config_dict)

    if not options.__dict__.get("infile_format"):
        options.__dict__["infile_format"] = "fof"

    sys.stdout.flush()  # clear console buff

    print("Configuration:")
    print(options)
    options.word_vec_dim = 1024

    #  Save configuration to log files

    model_dir = options.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = options.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = log_dir + "/song.{}".format(options.suffix) + ".log"
    print("The path of log file: " + log_file_path)
    log_file = open(log_file_path, "wt")
    log_file.write(str(options) + "\n")
    log_file.flush()

    config_dict = vars(options)
    with open(log_dir + "/song.{}".format(options.suffix) + "_config.json", "w") as f_out:
        json.dump(config_dict, f_out, indent=4)

    #  Read in data and separate them into training part and development part

    print("Loading training set...")
    if options.infile_format == "fof":
        train_set, len_node, len_in_node, len_out_node, entity_type = get_data_from_fof(options)
    else:
        train_set, len_node, len_in_node, len_out_node, entity_type = get_data_from_file(options.train_path, options)

    random.shuffle(train_set)
    dev_set = train_set[:200]
    train_set = train_set[200:]

    print('Number of training samples:' + str(len(train_set)))
    print('Number of development samples:' + str(len(dev_set)))

    print("Number of node: " + str(len_node) + ", while max allowed is " + str(options.max_node_num))
    print("Number of parent node: " + str(len_in_node) + ", truncated to " + str(options.max_in_node_num))
    print("Number of child node: " + str(len_out_node) + ", truncated to " + str(options.max_out_node_num))
    print("Number of entity type: " + str(entity_type) + ", truncated to " + str(options.entity_type))

    # Build dictionary and mapping of words, characters, edges

    words, chars, edges = collect_data(train_set)
    print('Number of words:' + str(len(words)))
    print('Number of characters:' + str(len(chars)))
    print('Number of edges:' + str(len(edges)))

    dict_word, word_to_id, id_to_word = word_mapping(words)
    dict_char, char_to_id, id_to_char = char_mapping(chars)
    dict_edge, edge_to_id, id_to_edge = edge_mapping(edges)

    options.word_to_id = word_to_id
    options.char_to_id = char_to_id
    options.edge_to_id = edge_to_id

    train_set = get_dataset_from_instances(train_set, word_to_id, char_to_id, edge_to_id, options)
    dev_set = get_dataset_from_instances(dev_set, word_to_id, char_to_id, edge_to_id, options)

    #  Build dataloader of training set and development set

    batch_size = options.batch_size

    # train_dataset = Dataset(train_set)
    # train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
    #                                collate_fn=collate_fn, num_workers=0, drop_last=True)
    #
    dev_dataset = Dataset(dev_set)
    dev_loader = data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, num_workers=0, drop_last=True)

    _model = GsGLstm(options)
    _model = _model.to(device)


    with torch.no_grad():
        for batch_idx, (node_num, lemmas, lemmas_idx, lemmas_chars, in_nodes, in_labels, out_nodes, out_labels,
                        entity_indexs, truth_tags, in_node_mask, out_node_mask, entity_mask) in enumerate(dev_loader):
            if batch_idx > 0:
                break
            graph_rep, node_cell, node_hidden = _model(node_num, lemmas, lemmas_idx, lemmas_chars, in_nodes, in_labels,
                                                       out_nodes, out_labels, entity_indexs, truth_tags,
                                                       in_node_mask, out_node_mask, entity_mask, options)
            print(graph_rep, node_cell, node_hidden)

            break




    










