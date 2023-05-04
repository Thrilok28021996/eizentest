from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.loss import _Loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


use_cuda = False
use_mps = True

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


class BiTemperedLogisticLoss(_Loss):
    def __init__(self, reduction="mean", t1=1, t2=1, label_smoothing=0.0, num_iters=5):
        super().__init__(reduction=reduction)
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    @classmethod
    def log_t(cls, u, t):
        """Compute log_t for `u`."""

        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    @classmethod
    def exp_t(cls, u, t):
        """Compute exp_t for `u`."""

        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    @classmethod
    def compute_normalization_fixed_point(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            logt_partition = torch.sum(
                cls.exp_t(normalized_activations, t), dim=-1
            ).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (
                logt_partition ** (1.0 - t)
            )

        logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(
            -1, 1
        )

        return -cls.log_t(1.0 / logt_partition, t) + mu

    @classmethod
    def compute_normalization(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return cls.compute_normalization_fixed_point(activations, t, num_iters)

    @classmethod
    def tempered_softmax(cls, activations, t, num_iters=5):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(
                torch.sum(torch.exp(activations), dim=-1)
            )
        else:
            normalization_constants = cls.compute_normalization(
                activations, t, num_iters
            )

        return cls.exp_t(activations - normalization_constants, t)

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        labels: A tensor with shape and dtype as activations.
        t1: Temperature 1 (< 1.0 for boundedness).
        t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smoothing: Label smoothing parameter between [0, 1).
        num_iters: Number of iterations to run the method.
        Returns:
        A loss tensor.
        """
        if self.label_smoothing > 0.0:
            targets = BiTemperedLogisticLoss._smooth_one_hot(
                targets, inputs.size(-1), self.label_smoothing
            )

        probabilities = self.tempered_softmax(inputs, self.t2, self.num_iters)

        temp1 = (
            self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)
        ) * targets
        temp2 = (1 / (2 - self.t1)) * (
            torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1)
        )
        loss = temp1 - temp2

        loss = loss.sum(dim=-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


loss_function = BiTemperedLogisticLoss(
    reduction="sum", t1=0.7, t2=1.3, label_smoothing=0.3
)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 6000 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("../data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1)
test_loader = torch.utils.data.DataLoader(dataset2)

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
epochs = 10
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
