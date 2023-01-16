def count_params(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
