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
d = torch.zeros(8, 10, 22)
for i in range(8):
    d[i] = c

e = torch.rand(8, 15, 22, 150)
res = torch.rand(8, 15, 22, 150)
print(e[0][1])
padding_t = torch.zeros(e[0][0][0].shape)
padding_e = torch.zeros(e[0][0].shape)

for i in range(8):
    for j in range(15):
        if j < 10:
            for k in range(22):
                if d[i][j][k] == 0:
                    e[i][j][k] = padding_t
        else:
            e[i][j] = padding_e

print(e.shape)
print(e[0][1])


# (batch, num_nodes, num_neighbour, graph_hidden_dim) * (batch, max_node_num, max_in_node_num, 1)