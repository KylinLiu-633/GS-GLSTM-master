

def create_dictionary(item_list):
    # Build a dictionary for a list
    assert type(item_list) is list
    dictionary = {}
    for item in item_list:
        if item not in dictionary:
            dictionary[item] = 1
        else:
            dictionary[item] += 1
    return dictionary


def create_mapping(dictionary):
    # Build a mapping to dictionary 按照頻率降序排序 x[0] is key x[1] is the count_num of this word
    sorted_items = sorted(dictionary.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    return item_to_id, id_to_item


def word_mapping(words):
    dictio = create_dictionary(words)
    word_to_id, id_to_word = create_mapping(dictio)
    return dictio, word_to_id, id_to_word


def char_mapping(chars):
    dictio = create_dictionary(chars)
    char_to_id, id_to_char = create_mapping(dictio)
    return dictio, char_to_id, id_to_char


def edge_mapping(edges):
    dictio = create_dictionary(edges)
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

        lemmas_idx = [word_to_id[x] for x in lemmas]
        lemmas_char_idx = None

        if options.with_char:
            lemmas_char_idx = char_to_id(lemmas)

        in_label_idx = []
        for edges in in_label:
            in_label_idx.append([edge_to_id[x] for x in edges])
        out_label_idx = []
        for edges in out_label:
            out_label_idx.append([edge_to_id[x] for x in edges])

        instances.append((lemmas_idx, lemmas_char_idx, in_node, in_label_idx, out_node, out_label_idx,
                          entity_indexs, truth_tags))

    return instances







