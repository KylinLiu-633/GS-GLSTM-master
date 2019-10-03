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
                  [[0,1,2,3,4,2], [1,3,1,3,4,13], [1,2,2,3,4,3], [1,1,2,3,4,2], [0,1,2,4,4,1]]])

d = torch.rand(8, 137, 256)
e = torch.zeros(8, 450)
for b in range(8):
    for i in range(137):
        e[b][i] = 1
res = torch.zeros(e.shape)
for b in range(8):
    for i in range(137):
        res[b, i] = d[b, i]

print(e.shape)

print(res.shape)
print(res[0][0])
print(res[0][-1])


# (batch, num_nodes, num_neighbour, graph_hidden_dim) * (batch, max_node_num, max_in_node_num, 1)