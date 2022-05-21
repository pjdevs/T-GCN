import torch


def self_loop_laplacian(matrix: torch.Tensor) -> torch.Tensor:
    matrix += torch.eye(matrix.size(0))
    degrees = matrix.sum(1)
    degrees_inv_sqrt = torch.pow(degrees, -1.0/2.0).flatten()
    degrees_inv_sqrt[torch.isinf(degrees_inv_sqrt)] = 0.0
    degrees_inv_sqrt_matrix = torch.diag(degrees_inv_sqrt)

    normalized_laplacian = matrix.matmul(degrees_inv_sqrt_matrix).transpose(
        0, 1).matmul(degrees_inv_sqrt_matrix)
    return normalized_laplacian
