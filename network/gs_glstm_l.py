import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np



class GsGLstmL(nn.Module):
    """
    Args:

    Shape:
        - Input:

            lemmas.shape (batch, sentence_len, word_vec_dim)
            lemmas_idx.shape (batch, sentence_len, 1)
            in_nodes.shape (batch, sentence_len, max_in_node_num of this batch)
            in_edges.shape (batch, sentence_len, max_in_node_num of this batch)
            in_node_mask.shape (batch, sentence_len, max_in_node_num of this batch)
            out_nodes.shape (batch, sentence_len, max_out_node_num of this batch)
            out_edges.shape (batch, sentence_len, max_out_node_num of this batch)
            out_node_mask.shape (batch, sentence_len, max_out_node_num of this batch)
            entity_indexs.shape (batch, entity_num, max len of entities of this batch)
            entity_mask.shape (batch, entity_num, max len of entities of this batch)

        - Output:

    Attributes:
        w_in_i_gate: the learnable weights of the nodes for input_gate of shape (g_hidden_dim, g_hidden_dim)
        u_in_i_gate: the learnable weights of the edges for input_gate of shape (g_hidden_dim)
        w_out_i_gate: the learnable weights of the nodes for input_gate of shape (g_hidden_dim, g_hidden_dim)
        u_out_i_gate: the learnable weights of the edges for input_gate of shape (g_hidden_dim)
        b_i_gate: the learnable bias of the input_gate of shape (g_hidden_dim)

        w_in_o_gate: the learnable weights of the nodes for output_gate of shape (g_hidden_dim, g_hidden_dim)
        u_in_o_gate: the learnable weights of the edges for output_gate of shape (g_hidden_dim)
        w_out_o_gate: the learnable weights of the nodes for output_gate of shape (g_hidden_dim, g_hidden_dim)
        u_out_o_gate: the learnable weights of the edges for output_gate of shape (g_hidden_dim)
        b_o_gate: the learnable bias of the output_gate of shape (g_hidden_dim)

        w_in_f_gate: the learnable weights of the nodes for forget_gate of shape (g_hidden_dim, g_hidden_dim)
        u_in_f_gate: the learnable weights of the edges for forget_gate of shape (g_hidden_dim)
        w_out_f_gate: the learnable weights of the nodes for forget_gate of shape (g_hidden_dim, g_hidden_dim)
        u_out_f_gate: the learnable weights of the edges for forget_gate of shape (g_hidden_dim)
        b_f_gate: the learnable bias of the forget_gate of shape (g_hidden_dim)

        w_in_cell: the learnable weights of the nodes for cell itself of shape (g_hidden_dim, g_hidden_dim)
        u_in_cell: the learnable weights of the edges for cell itself of shape (g_hidden_dim)
        w_out_cell: the learnable weights of the nodes for cell itself of shape (g_hidden_dim, g_hidden_dim)
        u_out_cell: the learnable weights of the edges for cell itself of shape (g_hidden_dim)
        b_cell: the learnable bias of the cell itself of shape (g_hidden_dim)



    Examples:

    """

    def __init__(self, options):
        super(GsGLstmL, self).__init__()
        self.word_to_id = options.word_to_id
        self.edge_to_id = options.edge_to_id
        self.char_to_id = options.char_to_id
        self.device = options.device

        self.g_hidden_dim = options.g_hidden_dim
        if options.attention_type == 'hidden':
            self.encoder_dim = self.g_hidden_dim
        elif options.attention_type == 'hidden_cell':
            self.encoder_dim = self.g_hidden_dim * 2
        elif options.attention_type == 'hidden_embed':
            self.encoder_dim = self.g_hidden_dim * 2

        # self.encoder_dim = self.g_hidden_dim

        self.relation_num = options.relation_num
        self.entity_num = options.entity_type

        self.batch = options.batch_size
        self.word_vec_dim = options.word_vec_dim
        self.edge_dim = len(self.edge_to_id)

        self.max_node_num = options.max_node_num
        self.max_in_node_num = options.max_in_node_num
        self.max_out_node_num = options.max_out_node_num
        self.max_entity_size = options.max_entity_size
        self.layer_num = options.gslstm_layer_num
        self.edge_label_dim = options.edge_label_dim
        self.dropout_rate = options.dropout_rate

        self.in_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_in_node_num)
        self.out_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_out_node_num)

        self.w_in_i_gate = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_in_i_gate = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.w_out_i_gate = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_out_i_gate = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.b_i_gate = nn.Parameter(torch.zeros(self.layer_num, self.g_hidden_dim))

        self.w_in_o_gate = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_in_o_gate = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.w_out_o_gate = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_out_o_gate = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.b_o_gate = nn.Parameter(torch.zeros(self.layer_num, self.g_hidden_dim))

        self.w_in_f_gate = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_in_f_gate = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.w_out_f_gate = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_out_f_gate = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.b_f_gate = nn.Parameter(torch.zeros(self.layer_num, self.g_hidden_dim))

        self.w_in_cell = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_in_cell = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.w_out_cell = nn.Parameter(nn.init.normal_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.u_out_cell = nn.Parameter(nn.init.zeros_(torch.Tensor(self.layer_num, self.g_hidden_dim, self.g_hidden_dim)))
        self.b_cell = nn.Parameter(torch.zeros(self.layer_num, self.g_hidden_dim))

        self.edge_rep = nn.Linear(self.g_hidden_dim + self.edge_label_dim, self.g_hidden_dim)

        self.w_rel = nn.Parameter(nn.init.normal_(torch.Tensor(self.encoder_dim * self.entity_num, self.relation_num)))
        self.b_rel = nn.Parameter(torch.zeros(self.relation_num))

        # self.node_hidden = nn.Parameter(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim))
        # self.node_cell = nn.Parameter(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim))
        self.node_hidden = None
        self.node_cell = nn.init.zeros_(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim)).to(self.device)

        self.rep = torch.empty(self.batch, self.max_node_num, self.max_in_node_num, self.g_hidden_dim)
        self.rep = nn.Parameter(nn.init.zeros_(self.rep))
        self.active = nn.Sigmoid()

        # The output of word_embedding is word representation for nodes, where each node only includes one word
        self.word_embedding = nn.Linear(self.word_vec_dim, self.g_hidden_dim)  # 1024 => 150
        self.edge_embedding = nn.Embedding(self.edge_dim, self.edge_label_dim)  # 114 => 3
        self.embeding_dropout = nn.Dropout(p=0.3)
        self.node_mask = None  # (batch, max_node_size)
        self.graph_rep = None
        self.graph_hidden = None
        self.classifier = nn.Linear(self.g_hidden_dim, self.relation_num)
        self.seq_len = 0

    def forward(self, node_num, lemmas, lemmas_idx, lemmas_chars, in_nodes, in_labels, out_nodes, out_labels, entity_indexs, \
           truth_tags, in_node_mask, out_node_mask, entity_mask, options):

        # get node mask of whole graph
        # node_mask = torch.zeros(self.batch, self.max_node_num)
        # for b in range(self.batch):
        #     node_mask[b][:node_num[b]] = torch.ones(node_num[b])
        # self.node_mask = node_mask  # (8, 450)
        # self.node_mask = torch.tensor(node_num).to(self.device)
        # print("mask: ", self.node_mask.shape)
        self.seq_len = max(node_num)

        word_rep, in_node_rep, out_node_rep = self._get_embedding(lemmas, in_nodes, in_labels, in_node_mask, out_nodes,
                                                                  out_labels, out_node_mask)
        # word_rep, in_node_rep, out_node_rep => torch.Size([8, 450, 150]) torch.Size([3600, 150]) torch.Size([3600, 150])
        self.node_hidden = word_rep
        self.graph_hidden = word_rep
        # get graph_representation
        graph_rep, graph_cell = self._get_graph_rep(in_nodes, in_node_mask, out_nodes, out_node_mask, in_node_rep, out_node_rep)

        encoder_state = None
        if options.attention_type == 'hidden':
            encoder_state = self.graph_hidden
        elif options.attention_type == 'hidden_cell':
            encoder_state = torch.cat((self.graph_hidden, graph_cell), dim=-1)
        elif options.attention_type == 'hidden_embed':
            encoder_state = torch.cat((self.graph_hidden, self.node_hidden), dim=-1)

        entity_rep = self._get_entity_rep(encoder_state, entity_indexs)

        return entity_rep

    def _get_embedding(self, lemmas, in_nodes, in_labels, in_node_mask, out_nodes,  out_labels, out_node_mask):
        # the representation of words
        lemmas = lemmas.to(self.device)
        word_rep = self.word_embedding(lemmas)
        # word_rep = Func.dropout(word_rep, p=self.dropout_rate, training=False)
        # word_rep = self.mask_select_of_node(word_rep, self.node_mask)

        # the representation of in_edges, out_edges
        in_labels = in_labels.to(self.device)
        out_labels = out_labels.to(self.device)
        in_edge_input = self.edge_embedding(in_labels)  # (8, 137, 19, 50)
        out_edge_input = self.edge_embedding(out_labels)

        in_nodes = in_nodes.to(self.device)
        out_nodes = out_nodes.to(self.device)
        in_node_mask = in_node_mask.to(self.device)
        out_node_mask = out_node_mask.to(self.device)

        in_node_input = self._get_neighbour_rep(word_rep, in_nodes)  # (8, 137, 19, 150)
        out_node_input = self._get_neighbour_rep(word_rep, out_nodes)

        in_node_rep = torch.cat((in_edge_input, in_node_input), dim=-1).to(self.device)  # (8, 137, 19, 200)
        in_node_rep = self.mask_select_of_neighbour(in_node_rep, in_node_mask,
                                                    vec_dim=self.g_hidden_dim + self.edge_label_dim)
        in_node_rep = in_node_rep.sum(-2)

        out_node_rep = torch.cat((out_edge_input, out_node_input), dim=-1).to(self.device)
        out_node_rep = self.mask_select_of_neighbour(out_node_rep, out_node_mask,
                                                     vec_dim=self.g_hidden_dim + self.edge_label_dim)
        out_node_rep = out_node_rep.sum(-2)

        in_node_rep = self.edge_rep(in_node_rep.view((-1, self.g_hidden_dim + self.edge_label_dim)))
        in_node_rep = torch.tanh(in_node_rep)

        out_node_rep = self.edge_rep(out_node_rep.view((-1, self.g_hidden_dim + self.edge_label_dim)))
        out_node_rep = torch.tanh(out_node_rep)

        return word_rep, in_node_rep, out_node_rep

    def _get_graph_rep(self, in_nodes, in_node_mask, out_nodes, out_node_mask, in_node_rep, out_node_rep):

        graph_rep = torch.zeros(self.layer_num, self.batch, self.seq_len, self.g_hidden_dim)
        in_node_rep = in_node_rep.view((self.batch, self.max_node_num, self.g_hidden_dim))
        out_node_rep = out_node_rep.view((self.batch, self.max_node_num, self.g_hidden_dim))
        node_hidden = self.graph_hidden
        node_cell = self.node_cell

        def unit_mul(mat_1, mat_2):
            # res = torch.zeros(self.batch, self.max_node_num, self.g_hidden_dim).to(self.device)
            res = []
            for b in range(self.batch):
                for n in range(self.max_node_num):
                    res.append(torch.matmul(mat_1[b][n], mat_2[n]))
            res = torch.stack(res, dim=0)
            # res = res.view((-1, self.g_hidden_dim))
            return res

        for layer_idx in range(self.layer_num):
            in_nodes = in_nodes.to(self.device)
            in_edge_hidden = self._get_neighbour_rep(node_hidden, in_nodes)
            in_edge_hidden = self.mask_select_of_neighbour(in_edge_hidden, in_node_mask)
            in_edge_hidden = in_edge_hidden.sum(-2)
            prev_in_edge_hidden = in_edge_hidden.view((self.batch, self.max_node_num, self.g_hidden_dim))

            out_nodes = out_nodes.to(self.device)
            out_edge_hidden = self._get_neighbour_rep(node_hidden, out_nodes)
            out_edge_hidden = self.mask_select_of_neighbour(out_edge_hidden, out_node_mask)
            out_edge_hidden = out_edge_hidden.sum(-2)
            prev_out_edge_hidden = out_edge_hidden.view((self.batch, self.max_node_num, self.g_hidden_dim))

            edge_i_gate = self.active(torch.matmul(in_node_rep, self.w_in_i_gate[layer_idx])
                                      + torch.matmul(prev_in_edge_hidden, self.u_in_i_gate[layer_idx])
                                      + torch.matmul(out_node_rep, self.w_out_i_gate[layer_idx])
                                      + torch.matmul(prev_out_edge_hidden, self.u_out_i_gate[layer_idx])
                                      + self.b_i_gate[layer_idx])
            edge_o_gate = self.active(torch.matmul(in_node_rep, self.w_in_o_gate[layer_idx])
                                       + torch.matmul(prev_in_edge_hidden, self.u_in_o_gate[layer_idx])
                                       + torch.matmul(out_node_rep, self.w_out_o_gate[layer_idx])
                                       + torch.matmul(prev_out_edge_hidden, self.u_out_o_gate[layer_idx])
                                       + self.b_o_gate[layer_idx])
            edge_f_gate = self.active(torch.matmul(in_node_rep, self.w_in_f_gate[layer_idx])
                                       + torch.matmul(prev_in_edge_hidden, self.u_in_f_gate[layer_idx])
                                       + torch.matmul(out_node_rep, self.w_out_o_gate[layer_idx])
                                       + torch.matmul(prev_out_edge_hidden, self.u_out_f_gate[layer_idx])
                                       + self.b_f_gate[layer_idx])
            edge_cell_input = self.active(torch.matmul(in_node_rep, self.w_in_cell[layer_idx])
                                          + torch.matmul(prev_in_edge_hidden, self.u_in_cell[layer_idx])
                                          + torch.matmul(out_node_rep, self.w_out_cell[layer_idx])
                                          + torch.matmul(prev_out_edge_hidden, self.u_out_cell[layer_idx])
                                          + self.b_cell[layer_idx])
            edge_i_gate = edge_i_gate.view((self.batch, -1, self.g_hidden_dim))
            edge_o_gate = edge_o_gate.view((self.batch, -1, self.g_hidden_dim))
            edge_f_gate = edge_f_gate.view((self.batch, -1, self.g_hidden_dim))
            edge_cell_input = edge_cell_input.view((self.batch, -1, self.g_hidden_dim))

            edge_cell = edge_f_gate * node_cell + edge_i_gate * edge_cell_input
            edge_hidden = edge_o_gate * torch.tanh(edge_cell)

            # node_cell = self.mask_select_of_node(edge_cell, self.seq_len)
            node_cell = edge_cell
            node_hidden = self.mask_select_of_node(edge_hidden, self.seq_len)
            # print(node_hidden.shape)

            graph_rep[layer_idx] = node_hidden

        self.graph_rep = graph_rep
        self.graph_hidden = self.mask_select_of_node(node_cell, self.seq_len)
        self.node_hidden = node_hidden

        return graph_rep, node_cell

    def _get_entity_rep(self, encoder_state, entity_indexs):
        rep = self._get_entity_by_index(encoder_state, entity_indexs)
        rep = rep.mean(-2)
        rep = rep.view((self.batch, -1)).to(self.device)
        rep = Func.softmax(torch.tanh(torch.matmul(rep, self.w_rel) + self.b_rel + 1e-8), dim=-1)

        return rep

    def _get_neighbour_rep(self, node_hidden, neighbour_index):
        """
        :param node_hidden: in the shape of (batch, num_nodes, graph_hidden_dim)
        :param neighbour_index: in the shape of (batch, num_nodes, num_neighbours)
        :return: the representation in the shape of (batch, num_nodes, num_neighbour, graph_hidden_dim)
        """
        res = torch.matmul(neighbour_index.unsqueeze(-1).double(), node_hidden.unsqueeze(-2)).squeeze()
        return res

    def _get_entity_by_index(self, encoder_state, entity_indexs):
        res = torch.empty(self.batch, self.entity_num, self.max_entity_size, self.encoder_dim)
        res = nn.init.zeros_(res)
        # res = torch.matmul(entity_indexs.unsqueeze(-1).double(), encoder_state.unsqueeze(-2)).squeeze()
        for b in range(self.batch):
            for i in range(self.entity_num):
                for j in range(self.max_entity_size):
                    res[b][i][j] = encoder_state[b][entity_indexs[b][i][j]]
        return res

    def mask_select_of_neighbour(self, edge_hidden, mask_t, vec_dim=0):
        if vec_dim == 0:
            vec_dim = self.g_hidden_dim
        seq_len = self.max_node_num - len(edge_hidden[0])
        neigh_len = self.max_in_node_num - len(edge_hidden[0][0])
        zero_n = torch.zeros(self.batch, len(edge_hidden[0]), neigh_len, vec_dim).to(self.device)
        edge_hidden = torch.cat((edge_hidden, zero_n), dim=-2)
        zero_n = torch.zeros(self.batch, seq_len, self.max_in_node_num, vec_dim).to(self.device)
        edge_hidden = torch.cat((edge_hidden, zero_n), dim=1)

        return edge_hidden

    def mask_select_of_node(self, node_hidden, seq_len, vec_dim=0):
        if vec_dim == 0:
            vec_dim = self.g_hidden_dim
        tmp = torch.zeros(self.batch, seq_len, vec_dim).to(self.device)
        for b in range(self.batch):
            tmp[b] = node_hidden[b][:seq_len]

        return tmp






