#!/usr/bin/env python3

import json
import torch

def to_list(t):
    return [float(x) for x in t.reshape(-1).tolist()]


def main():
    refs = {}

    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    refs["matmul"] = to_list(a @ b)

    refs["add"] = to_list(torch.tensor([[1.0, -2.0], [3.5, 0.5]]) + torch.tensor([[0.5, 2.0], [-1.5, 0.25]]))
    refs["relu"] = to_list(torch.relu(torch.tensor([-1.0, 0.0, 2.0, -3.0, 4.0])))
    refs["softmax"] = to_list(torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=1))

    graph_input = torch.tensor([[1.0, 2.0, -1.0]])
    graph_weight = torch.tensor([[0.5, 1.0], [-0.5, 0.25], [1.0, -1.0]])
    graph_bias = torch.tensor([[0.1, -0.2]])
    out = torch.relu(graph_input @ graph_weight + graph_bias)
    out = torch.softmax(out, dim=1).reshape(2, 1).transpose(0, 1)
    refs["runtime_path"] = to_list(out)

    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()
