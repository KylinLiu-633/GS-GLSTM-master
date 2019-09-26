import re
import json
import codecs
import torch


def read_file(file_path):
    lines = []
    with open(file_path, "rt") as f_in:
        for line in f_in:
            lines.append(line.strip())

    return lines


def get_data_from_file(file_path, options):
    words = []
    lemmas = []
    pos_tags = []
    in_nodes = []
    in_labels = []
    out_nodes = []  # [batch, node, neighbor]
    out_labels = []  # [batch, node, neighbor]
    entity_indexs = []  # [batch, 3, neighbor]
    truth_tags = []

    if options.binary_classification:
        dict_rel = {'resistance or non-response': 1, 'sensitivity': 1, 'response': 1, 'resistance': 1, 'None': 0}
    else:
        dict_rel = {'resistance or non-response': 1, 'sensitivity': 2, 'response': 3, 'resistance': 4, 'None': 0}

    len_words = 0
    len_in_nodes = 0
    len_out_nodes = 0
    entity_size = 0

    with codecs.open(file_path, 'rU', 'utf-8') as f_in:
        for inst in json.load(f_in):
            word = []
            lemma = []
            pos_tag = []
            if options.only_single_sent and len(inst['sentences']) > 1:
                continue
            for sent in inst['sentences']:
                for node in sent['nodes']:
                    word.append(node['label'])
                    lemma.append(node['lemma'])
                    pos_tag.append(node['postag'])
            len_words = max(len_words, len(word))
            words.append(word)
            lemmas.append(lemma)
            pos_tags.append(pos_tags)

            in_node = [[k, ] for k, _ in enumerate(word)]
            in_label = [['self', ] for k, _ in enumerate(word)]
            out_node = [[k, ] for k, _ in enumerate(word)]
            out_label = [['self', ] for k, _ in enumerate(word)]

            for sent in inst['sentences']:
                for node in sent['nodes']:
                    idx = node['index']
                    for arc in node['arcs']:
                        j = arc['toIndex']
                        l = arc['label']
                        l = l.split('::')[0]
                        l = l.split('_')[0]
                        l = l.split('(')[0]
                        if j == -1 or l == '':
                            continue
                        in_node[j].append(idx)
                        in_label[j].append(l)
                        out_node[idx].append(j)
                        out_label[idx].append(l)

            for k in in_node:
                len_in_nodes = max(len_in_nodes, len(k))
            for k in out_node:
                len_out_nodes = max(len_out_nodes, len(k))
            in_nodes.append(in_node)
            in_labels.append(in_label)
            out_nodes.append(out_node)
            out_labels.append(out_label)

            entity_index = []
            for k in inst['entities']:
                entity_index.append(k['indices'])
                entity_size = max(entity_size, len(k['indices']))

            assert len(entity_index) == options.entity_type
            entity_indexs.append(entity_index)
            truth_tags.append(dict_rel[inst['relationLabel'].strip()])
        lemmas = lemmas if options.word_format == 'lemma' else words

        return zip(lemmas, pos_tags, in_nodes, in_labels, out_nodes, out_labels, entity_indexs, truth_tags), len_words, \
               len_in_nodes, len_out_nodes, entity_size


def get_data_from_fof(options):
    all_paths = read_file(options.train_path)
    all_instances = []
    len_words = 0
    len_in_nodes = 0
    len_out_nodes = 0
    entity_size = 0
    for cur_path in all_paths:
        print(cur_path)
        cur_instances, cur_words, cur_in_nodes, cur_out_nodes, cur_entity_size = get_data_from_file(cur_path, options)
        all_instances.extend(cur_instances)
        len_words = max(len_words, cur_words)
        len_in_nodes = max(len_in_nodes, cur_in_nodes)
        len_out_nodes = max(len_out_nodes, cur_out_nodes)
        entity_size = max(entity_size, cur_entity_size)

    return all_instances, len_words, len_in_nodes, len_out_nodes, entity_size


def padding_3d(input_t, dim_0=0, dim_1=0, dim_2=0):
    if dim_0 == 0:
        dim_0 = len(input_t)
    if dim_1 == 0:
        dim_1 = max(len(input_t[i]) for i in input_t)
    if dim_2 == 0:
        for i in input_t:
            dim_2 = max(dim_2, max(len(input_t[k]) for k in i))

    res = torch.zeros(dim_0, dim_1, dim_2)

    dim_0 = min(dim_0, len(input_t))

    for i in range(dim_0):
        temp_i = input_t[i]
        new_dim_1 = dim_1
        new_dim_1 = min(new_dim_1, len(temp_i))
        for j in range(new_dim_1):
            temp_j = temp_i[j]
            new_dim_2 = dim_2
            new_dim_2 = min(new_dim_2, len(temp_j))
            res[i, j, :new_dim_2] = temp_j[:new_dim_2]

    return res




