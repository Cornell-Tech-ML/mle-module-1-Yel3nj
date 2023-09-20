"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import random

import minitorch


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int) -> None:
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):  # type: ignore
        middle = [h.relu() for h in self.layer1.forward(x)]  # type: ignore
        end = [h.relu() for h in self.layer2.forward(middle)]  # type: ignore
        return self.layer3.forward(end)[0].sigmoid()  # type: ignore


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.weights = []  # type: ignore
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs: list[float]) -> list[float]:
        res = []
        for j in range(len(self.bias)):
            dot_product = sum(inputs[i] * self.weights[i][j].value for i in range(len(inputs)))
            res.append(dot_product + self.bias[j].value)
        return res


def default_log_fn(epoch: int, total_loss: float, correct: int, losses: float) -> None:
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers: int) -> None:
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):  # type: ignore
        return self.model.forward(  # type: ignore
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):  # type: ignore
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))  # type: ignore

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data  # type: ignore

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)  # type: ignore
