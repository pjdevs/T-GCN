import torch
import torch.nn
import torch.utils
import torch.utils.data


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader, loss_function: torch.nn.BCEWithLogitsLoss, nb_epochs: int):
    """
    Train an PyTorch module.
    """

    model.train()

    for epoch in range(nb_epochs):
        for (data_inputs, data_labels) in data_loader:
            # Run model onn input data
            predictions = model(data_inputs)
            # predictions = predictions.squeeze(dim=1)

            # Calculate the loss
            loss = loss_function(predictions, data_labels)

            # Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
