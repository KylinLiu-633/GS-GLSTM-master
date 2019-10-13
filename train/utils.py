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
        dim_1 = max(len(i) for i in input_t)
    if dim_2 == 0:
        for i in input_t:
            dim_2 = max(dim_2, max(len(k) for k in i))

    res = torch.full((dim_0, dim_1, dim_2), 0.)

    dim_0 = min(dim_0, len(input_t))

    for i in range(dim_0):
        temp_i = input_t[i]
        new_dim_1 = dim_1
        new_dim_1 = min(new_dim_1, len(temp_i))
        for j in range(new_dim_1):
            temp_j = temp_i[j]
            new_dim_2 = dim_2
            new_dim_2 = min(new_dim_2, len(temp_j))
            res[i, j, :new_dim_2] = torch.tensor(temp_j[:new_dim_2])

    return res


def create_dictionary(item_list, add_unk=False):
    # Build a dictionary for a list
    assert type(item_list) is list
    dictionary = {}
    for item in item_list:
        if item not in dictionary:
            dictionary[item] = 1
        else:
            dictionary[item] += 1
    if add_unk:
        dictionary['UNK'] = 1
    return dictionary


def create_mapping(dictionary):
    # Build a mapping to dictionary 按照頻率降序排序 x[0] is key x[1] is the count_num of this word
    sorted_items = sorted(dictionary.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    return item_to_id, id_to_item


def word_mapping(words):
    dictio = create_dictionary(words, add_unk=True)
    word_to_id, id_to_word = create_mapping(dictio)
    return dictio, word_to_id, id_to_word


def char_mapping(chars):
    dictio = create_dictionary(chars, add_unk=True)
    char_to_id, id_to_char = create_mapping(dictio)
    return dictio, char_to_id, id_to_char


def edge_mapping(edges):
    dictio = create_dictionary(edges, add_unk=True)
    edge_to_id, id_to_edge = create_mapping(dictio)
    return dictio, edge_to_id, id_to_edge


def collect_data(instances):
    words = set()
    chars = set()
    edges = set()
    for (lemmas, pos_tags, in_nodes, in_labels, out_nodes, out_labels, entity_indexs, truth_tags) in instances:
        words.update(lemmas)
        for w in lemmas:
            if w.isspace() == False:
                chars.update(w)
        for edge in in_labels:
            edges.update(edge)
        for edge in out_labels:
            edges.update(edge)
    words = list(words)
    chars = list(chars)
    edges = list(edges)
    return words, chars, edges


def get_dataset_from_instances(all_instances, word_to_id, char_to_id, edge_to_id, options):

    instances = []
    for (lemmas, pos_tags, in_nodes, in_labels, out_nodes, out_labels, entity_indexs, truth_tags) in all_instances:

        # Ignore the passages were too long
        if options.max_node_num != -1 and len(lemmas) > options.max_node_num:
            continue

        in_node = [x[:options.max_in_node_num] for x in in_nodes]
        in_label = [x[:options.max_in_node_num] for x in in_labels]
        out_node = [x[:options.max_out_node_num] for x in out_nodes]
        out_label = [x[:options.max_out_node_num] for x in out_labels]

        lemmas_idx = [word_to_id[x] if word_to_id.get(x) else edge_to_id.get("UNK") for x in lemmas]
        lemmas_char_idx = None

        if options.with_char:
            lemmas_char_idx = [[char_to_id[x] if word_to_id.get(x) else word_to_id.get("UNK") for x in s] for s in lemmas]

        in_label_idx = []
        for edges in in_label:
            in_label_idx.append([edge_to_id[x] if edge_to_id.get(x) else edge_to_id.get("UNK") for x in edges])
        out_label_idx = []
        for edges in out_label:
            out_label_idx.append([edge_to_id[x] if edge_to_id.get(x) else edge_to_id.get("UNK") for x in edges])

        instances.append([lemmas, lemmas_idx, lemmas_char_idx, in_node, in_label_idx, out_node, out_label_idx,
                          entity_indexs, truth_tags])

    return instances










