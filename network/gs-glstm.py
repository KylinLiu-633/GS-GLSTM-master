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



class GsLstm(nn.Module):
    """
    Args:

    Shape:
        - Input:
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
        super(GsLstm, self).__init__()
        self.word_to_id = options.word_to_id
        self.edge_to_id = options.edge_to_id
        self.char_to_id = options.char_to_id

        self.g_hidden_dim = options.g_hidden_dim

        self.batch = options.batch_size
        self.max_node_num = options.max_node_num
        self.max_in_node_num = options.max_in_node_num
        self.max_out_node_num = options.max_out_node_num
        self.layer_num = options.gslstm_layer_num

        self.in_node_index = nn.Embedding(self.max_in_node_num, self.g_hidden_num)  # batch * max_node_num * max_in_node_num =>
        self.in_node_edges = nn.Embedding(self.max_in_node_num)
        self.in_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_in_node_num)

        self.out_node_index = nn.Embedding(self.max_out_node_num)  # batch * max_node_num * max_in_node_num =>
        self.out_node_edges = nn.Embedding(self.max_out_node_num)
        self.out_node_mask = torch.zeros(self.batch, self.max_node_num, self.max_out_node_num)

        self.w_in_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_i_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_in_i_gate = Param(torch.Tensor(self.g_hidden_dim))

        self.w_in_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_o_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_in_o_gate = Param(torch.Tensor(self.g_hidden_dim))

        self.w_in_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_f_gate = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_in_f_gate = Param(torch.Tensor(self.g_hidden_dim))

        self.w_in_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_in_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.w_out_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.u_out_cell = Param(torch.Tensor(self.g_hidden_dim, self.g_hidden_dim))
        self.b_in_cell = Param(torch.Tensor(self.g_hidden_dim))

        self.node_hidden = Param(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim))
        self.cell = Param(torch.Tensor(self.batch, self.max_node_num, self.g_hidden_dim))




        self.embedder = None
        self.encoder = None
        self.decoder = None
        self.classifier = None

    def forward(self, options):

        for layer_idx in range(self.layer_num):
            prev_in_edge_hidden = self._get_neighbour_rep(self.node_hidden, self.in_node_index)
            # (batch, num_nodes, num_neighbour, graph_hidden_dim) * (batch, max_node_num, max_in_node_num, 1)
            prev_in_edge_hidden = torch.matmul(prev_in_edge_hidden, torch.unsqueeze(self.in_node_mask, dim=-1))
            prev_in_edge_hidden = torch.squeeze(prev_in_edge_hidden, dim=2)


            # prev_in_edge_hidden is in shape of (-1, g_hidden_dim)



        return None


    def _get_neighbour_rep(self, node_hidden, neighbour_index):
        """

        :param node_hidden: in the shape of (batch, num_nodes, graph_hidden_dim)
        :param neighbour_index: in the shape of (batch, num_nodes, num_neighbours)
        :return: the representation in the shape of (batch, num_nodes, num_neighbour, graph_hidden_dim)
        """

        hidden_dim = len(node_hidden[0][0])
        num_nodes = len(node_hidden[0])
        batch = len(node_hidden)
        num_neighbours = len(neighbour_index[0][0])
        neighbour_index = neighbour_index.long()
        rep = torch.zeros(batch, num_nodes, num_neighbours, hidden_dim)

        for i in range(batch):
            for j in range(num_nodes):
                for k in range(num_neighbours):
                    rep[i][j][k] = node_hidden[i][neighbour_index[i][j][k]]

        return rep





