import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.parameter as Param


class GsEmbedder(nn.Module):
    """
    Args:

    Shape:
        - Input:
        - Output:

    Attributes:


    Examples:

    """
    def __init__(self, options):
        super(GsEmbedder, self).__init__()
        self.batch = options.batch_size
        self.max_node_num = options.max_node_num
        self.max_in_node_num = options.max_in_node_num
        self.max_out_node_num = options.max_out_node_num

        self.nodes_num = self.batch
        self.nodes = nn.Embedding(self.batch, self.max_node_num)  # batch * max_node_num =>
        self.nodes_mask = torch.zeros(self.batch, self)
        if options.with_char:
            self.nodes_char_num = self.batch
            self.nodes_char = nn.Embedding(self.batch, self.max_node_num)  # batch * max_node_num =>

        self.in_node_index = nn.Embedding(self.max_in_node_num, self.g_hidden_num)  # batch * max_node_num * max_in_node_num =>
        self.in_node_edges = nn.Embedding(self.max_in_node_num)
        self.in_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_in_node_num)

        self.out_node_index = nn.Embedding(self.max_out_node_num)  # batch * max_node_num * max_in_node_num =>
        self.out_node_edges = nn.Embedding(self.max_out_node_num)
        self.out_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_out_node_num)




    def forward(self, *input):
        #  Return the last hidden state of encoder
        return None



class GsGLstm(nn.Module):
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
        super(GsGLstm, self).__init__()
        self.word_to_id = options.word_to_id
        self.edge_to_id = options.edge_to_id
        self.char_to_id = options.char_to_id

        self.g_hidden_dim = options.g_hidden_dim

        self.batch = options.batch_size
        self.word_vec_dim = options.word_vec_dim
        self.edge_dim = len(self.edge_to_id)

        self.max_node_num = options.max_node_num
        self.max_in_node_num = options.max_in_node_num
        self.max_out_node_num = options.max_out_node_num
        self.layer_num = options.gslstm_layer_num
        self.edge_label_dim = options.edge_label_dim

        self.in_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_in_node_num)
        self.out_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_out_node_num)

        self.w_in_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_i_gate = Param(torch.Tensor(self.g_hidden_dim))

        self.w_in_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_o_gate = Param(torch.Tensor(self.g_hidden_dim))

        self.w_in_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_f_gate = Param(torch.Tensor(self.g_hidden_dim))

        self.w_in_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_cell = Param(torch.Tensor(self.g_hidden_dim))

        self.w_edge = Param(torch.Tensor(self.word_vec_dim+self.edge_dim, self.g_hidden_dim))
        self.b_edge = Param(torch.Tensor(self.g_hidden_dim))

        self.node_hidden = Param(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim))
        self.cell = Param(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim))

        # The output of word_embedding is word representation for nodes, where each node only includes one word
        self.word_embedding = nn.Embedding(self.word_vec_dim, self.word_vec_dim / 4)  # 1024 => 256
        self.edge_embedding = nn.Embedding(self.edge_dim, self.edge_label_dim)  # 113 => 3
        self.node_mask = None  # (batch, max_node_size)
        self.encoder = None
        self.decoder = None
        self.classifier = None
        self.graph_rep = None
        self.node_rep = None
        self.graph_hidden = None
        self.graph_cell = None

    def forward(self, node_num, lemmas, lemmas_idx, lemmas_chars, in_nodes, in_labels, out_nodes, out_labels, entity_indexs, \
           truth_tags, in_node_mask, out_node_mask, entity_mask, options):

        # get node mask of whole graph
        node_mask = torch.zeros(self.batch, self.max_node_num)
        for b in range(self.batch):
            node_mask[b][:node_num[b]] = torch.ones(node_num[b])
        self.node_mask = node_mask  # (8, 450)

        word_rep, in_node_rep, out_node_rep = self._get_embedding(lemmas, in_nodes, in_labels, in_node_mask, out_nodes,
                                                                  out_labels, out_node_mask)

        # get graph_representation
        self._get_graph_rep(in_nodes, in_node_mask, out_nodes, out_node_mask, in_node_rep, out_node_rep)


        self.graph_rep = None
        self.node_rep = None
        self.graph_hidden = None
        self.graph_cell = None

        return None

    def _get_embedding(self, lemmas, in_nodes, in_labels, in_node_mask, out_nodes,  out_labels, out_node_mask):
        # the representation of words
        word_rep = self.word_embedding(lemmas)
        # (8, 137, 256) * (8, 137, 1)

        # Still working on
        word_rep = torch.masked_select(word_rep, self.node_mask)
        word_rep = torch.squeeze(word_rep)  # (8, 137, 256)

        # the representation of in_edges
        seq_len = len(lemmas[0])
        in_edge_input = torch.zeros(self.batch, seq_len, self.edge_dim)
        for i in range(self.batch):
            in_edge_input[i].scatter_(1, in_labels[i], 1)
        in_edge_input = self.edge_embedding(in_edge_input)  # (8, 137, 113) => (8, 137, 3)

        # the representation of out_edges
        out_edge_input = torch.zeros(self.batch, seq_len, self.edge_dim)
        for i in range(self.batch):
            out_edge_input[i].scatter_(1, out_labels[i], 1)
        out_edge_input = self.edge_embedding(out_edge_input)  # (8, 137, 113) => (8, 137, 3)

        in_node_input = self._get_neighbour_rep(word_rep, in_nodes)
        out_node_input = self._get_neighbour_rep(word_rep, out_nodes)

        in_node_rep = torch.cat((in_edge_input, in_node_input), dim=-1)
        in_node_rep = torch.matmul(in_node_rep, torch.unsqueeze(in_node_mask, -1))
        in_node_rep = in_node_rep.sum(-1)

        out_node_rep = torch.cat((out_edge_input, out_node_input), dim=-1)
        out_node_rep = torch.matmul(out_node_rep, torch.unsqueeze(out_node_mask, -1))
        out_node_rep = out_node_rep.sum(-1)

        in_node_rep = torch.matmul(in_node_rep.view((-1, self.word_vec_dim + self.edge_dim)), self.w_edge) + self.b_edge
        in_node_rep = Func.tanh(in_node_rep)

        out_node_rep = torch.matmul(out_node_rep.view((-1, self.word_vec_dim + self.edge_dim)), self.w_edge) + self.b_edge
        out_node_rep = Func.tanh(out_node_rep)

        return word_rep, in_node_rep, out_node_rep

    def _get_graph_rep(self, in_nodes, in_node_mask, out_nodes, out_node_mask, in_node_rep, out_node_rep):

        graph_rep = []
        node_cell = []


        for layer_idx in range(self.layer_num):
            in_edge_hidden = self._get_neighbour_rep(self.node_hidden, in_nodes)
            # (batch, num_nodes, num_neighbour, graph_hidden_dim) * (batch, max_node_num, max_in_node_num, 1)
            in_edge_hidden = torch.matmul(in_edge_hidden.transpose(-1, -2), torch.unsqueeze(in_node_mask, dim=-1))
            in_edge_hidden = torch.squeeze(in_edge_hidden)
            in_edge_hidden = torch.matmul(in_edge_hidden.transpose(-1, -2), torch.unsqueeze(self.node_mask, dim=-1))
            prev_in_edge_hidden = in_edge_hidden.view((-1, self.g_hidden_dim))

            out_edge_hidden = self._get_neighbour_rep(self.node_hidden, out_nodes)
            # (batch, num_nodes, num_neighbour, graph_hidden_dim) * (batch, max_node_num, max_in_node_num, 1)
            out_edge_hidden = torch.matmul(out_edge_hidden.transpose(-1, -2), torch.unsqueeze(out_node_mask, dim=-1))
            out_edge_hidden = torch.squeeze(out_edge_hidden)
            out_edge_hidden = torch.matmul(out_edge_hidden.transpose(-1, -2), torch.unsqueeze(self.node_mask, dim=-1))
            prev_out_edge_hidden = out_edge_hidden.view((-1, self.g_hidden_dim))

            edge_i_gate = Func.sigmoid(torch.matmul(in_node_rep, self.w_in_i_gate)
                                       + torch.matmul(prev_in_edge_hidden, self.u_in_i_gate)
                                       + torch.matmul(out_node_rep, self.w_out_i_gate)
                                       + torch.matmul(prev_out_edge_hidden, self.u_out_i_gate)
                                       + self.b_i_gate)
            edge_o_gate = Func.sigmoid(torch.matmul(in_node_rep, self.w_in_o_gate)
                                       + torch.matmul(prev_in_edge_hidden, self.u_in_o_gate)
                                       + torch.matmul(out_node_rep, self.w_out_o_gate)
                                       + torch.matmul(prev_out_edge_hidden, self.u_out_o_gate)
                                       + self.b_o_gate)
            edge_f_gate = Func.sigmoid(torch.matmul(in_node_rep, self.w_in_f_gate)
                                       + torch.matmul(prev_in_edge_hidden, self.u_in_f_gate)
                                       + torch.matmul(out_node_rep, self.w_out_o_gate)
                                       + torch.matmul(prev_out_edge_hidden, self.u_out_f_gate)
                                       + self.b_f_gate)
            edge_cell_input = Func.sigmoid(torch.matmul(in_node_rep, self.w_in_cell)
                                           + torch.matmul(prev_in_edge_hidden, self.u_in_cell)
                                           + torch.matmul(out_node_rep, self.w_out_cell)
                                           + torch.matmul(prev_out_edge_hidden, self.u_out_cell)
                                           + self.b_cell)
            edge_i_gate = edge_i_gate.view((self.batch, -1, self.g_hidden_dim))
            edge_o_gate = edge_o_gate.view((self.batch, -1, self.g_hidden_dim))
            edge_f_gate = edge_f_gate.view((self.batch, -1, self.g_hidden_dim))
            edge_cell_input = edge_cell_input.view((self.batch, -1, self.g_hidden_dim))

            edge_cell = edge_f_gate * self.node_cell + edge_i_gate * edge_cell_input
            edge_hidden = edge_o_gate * Func.tanh(edge_cell)

            node_cell = torch.matmul(edge_cell, torch.unsqueeze(self.node_mask, -1))
            node_hidden = torch.matmul(edge_hidden, torch.unsqueeze(self.node_mask, -1))

            graph_rep.append(node_hidden)

        self.node_cell = node_cell

    def _get_neighbour_rep(self, node_hidden, neighbour_index):
        """
        :param node_hidden: in the shape of (batch, num_nodes, graph_hidden_dim)
        :param neighbour_index: in the shape of (batch, num_nodes, num_neighbours)
        :return: the representation in the shape of (batch, num_nodes, num_neighbour, graph_hidden_dim)
        """

        num_neighbours = len(neighbour_index[0][0])
        neighbour_index = neighbour_index.long()
        rep = torch.zeros(self.batch, self.max_node_num, num_neighbours, self.g_hidden_dim)

        for i in range(self.batch):
            for j in range(self.max_node_num):
                for k in range(num_neighbours):
                    rep[i, j, k] = node_hidden[i][neighbour_index[i][j][k]]

        return rep





