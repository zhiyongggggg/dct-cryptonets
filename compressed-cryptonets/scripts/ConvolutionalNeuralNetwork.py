"""
This is a .py version of the ConvolutionalNeuralNetwork notebook

"""
import os, sys
import time

import numpy as np
import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model
from concrete.fhe import Configuration


torch.manual_seed(42)


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits data-set."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of 1216 MAC
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, 2, stride=1, padding=0)
        self.fc1 = nn.Linear(32, n_classes)

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x


def train_one_epoch(net, optimizer, train_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()

    net.train()
    avg_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    return avg_loss / len(train_loader)


def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data, fhe=fhe_mode)

        endidx = idx + target.shape[0]

        # Accumulate the ground truth labels
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)

    return n_correct / len(test_loader)


def main():
    # --- Load dataset ---
    X, y = load_digits(return_X_y=True)

    # The sklearn Digits data-set, though it contains digit images, keeps these images in vectors
    # so we need to reshape them to 2D first. The images are 8x8 px in size and monochrome
    X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )

    # --- Train model ---
    # Create the tiny CNN with 10 output classes
    # N_EPOCHS = 150
    N_EPOCHS = 10

    # Create a train data loader
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    # Create a test data loader to supply batches for network evaluation (test)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_dataloader = DataLoader(test_dataset)

    # Train the network with Adam, output the test set accuracy every epoch
    net = TinyCNN(10)
    losses_bits = []
    optimizer = torch.optim.Adam(net.parameters())
    for _ in tqdm(range(N_EPOCHS), desc="Training"):
        losses_bits.append(train_one_epoch(net, optimizer, train_dataloader))

    # --- Test in Concrete ---
    configuration = Configuration(
        # To enable displaying progressbar
        show_progress=True,
        # To enable showing tags in the progressbar (does not work in notebooks)
        progress_tag=True,
        # To give a title to the progressbar
        progress_title='Evaluation: ',
    )

    n_bits = 6
    q_module = compile_torch_model(
        net,
        x_train,
        rounding_threshold_bits=6,
        p_error=0.1,
        configuration=configuration,
        verbose=True,
    )

    # Run inference in FHE on a set of encrypted example
    size_set = 5
    mini_test_dataset = TensorDataset(torch.Tensor(x_test[:size_set, :]), torch.Tensor(y_test[:size_set]))
    mini_test_dataloader = DataLoader(mini_test_dataset)

    t = time.time()
    accuracy_test = test_with_concrete(
        q_module,
        mini_test_dataloader,
        use_sim=True,
    )
    elapsed_time = time.time() - t
    time_per_inference = elapsed_time / len(mini_test_dataset)
    accuracy_percentage = 100 * accuracy_test

    print(
        f"Time per inference in FHE: {time_per_inference:.2f} "
        f"with {accuracy_percentage:.2f}% accuracy"
    )


if __name__ == "__main__":
    try:
        main()
        os._exit(0)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
