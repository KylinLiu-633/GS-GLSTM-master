import torch


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






a = torch.rand(3, 5, 6, 8)
b = torch.tensor([[[0,1,2,3,4,0], [1,1,1,3,4,15], [1,2,2,3,4,2], [1,1,2,3,4,3], [0,1,2,4,4,0]],
                  [[0,1,2,3,4,4], [1,2,1,3,4,14], [1,2,2,3,4,1], [1,1,2,3,4,1], [0,1,2,4,4,4]],
                  [[0,1,2,3,4,2], [1,3,1,3,4,13], [1,2,2,3,4,3], [1,1,2,3,4,2], [0,1,2,4,4,1]]], dtype=torch.float32)
# b = torch.rand(3, 5, 8, 10)
ff = torch.rand(3, 5, 6, 10)

c = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)

d = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)

e = torch.tensor([[1., 0.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.]])

word_rep = torch.rand(8, 10, 8)
print(word_rep.shape)
node_mask = torch.ones(8, 20)
print(node_mask.shape)
zero_t = torch.zeros(8, 1, 8)

seq_len = 20 - len(word_rep[0])
for n in range(seq_len):
    word_rep = torch.cat((word_rep, zero_t), dim=-2)
res = word_rep
print(res.shape)


# (batch, num_nodes, num_neighbour, graph_hidden_dim) * (batch, max_node_num, max_in_node_num, 1)