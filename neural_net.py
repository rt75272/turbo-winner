"""
Feedforward neural network — implemented entirely from scratch.

Only Python's built-in ``math`` and ``random`` modules are used; no
external packages or libraries are required.

Architecture
------------
The network uses ReLU activations on all hidden layers and raw (linear)
outputs on the final layer.  The caller selects the action with the
highest output value (argmax).

Weights are initialised with Xavier / Glorot uniform initialisation.

The ``flatten`` / ``unflatten`` pair is used by the genetic algorithm
to treat the entire parameter vector as a genome.
"""

import copy
import math
import random


class NeuralNetwork:
    """
    Feedforward neural network with configurable depth and width.

    Parameters
    ----------
    layer_sizes:
        List of integers specifying the number of neurons in each layer,
        e.g. ``[11, 32, 16, 3]`` gives 11 inputs, two hidden layers of
        32 and 16 neurons respectively, and 3 outputs.
    """

    def __init__(self, layer_sizes: list[int]) -> None:
        self.layer_sizes: list[int] = list(layer_sizes)
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), +sqrt(...))
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            w = [
                [random.uniform(-limit, limit) for _ in range(fan_out)]
                for _ in range(fan_in)
            ]
            b = [0.0] * fan_out
            self.weights.append(w)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x: float) -> float:
        """Rectified Linear Unit: max(0, x)."""
        return x if x > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Run a forward pass through the network.

        Args:
            inputs: List of floats of length ``layer_sizes[0]``.

        Returns:
            List of raw output values (no softmax applied).
        """
        layer = list(inputs)
        last_idx = len(self.weights) - 1
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            new_layer: list[float] = []
            for j in range(len(b)):
                s = b[j]
                for k in range(len(layer)):
                    s += layer[k] * w[k][j]
                # ReLU on all but the last layer
                if idx < last_idx:
                    s = self._relu(s)
                new_layer.append(s)
            layer = new_layer
        return layer

    def get_action(self, inputs: list[float]) -> int:
        """
        Return the index of the output neuron with the highest activation.

        Args:
            inputs: Game-state vector passed directly to ``forward()``.

        Returns:
            Integer action index (argmax of outputs).
        """
        outputs = self.forward(inputs)
        best = 0
        for i in range(1, len(outputs)):
            if outputs[i] > outputs[best]:
                best = i
        return best

    # ------------------------------------------------------------------
    # Serialisation helpers for the genetic algorithm
    # ------------------------------------------------------------------

    def flatten(self) -> list[float]:
        """Return all weights and biases as a single flat list (genome)."""
        params: list[float] = []
        for w in self.weights:
            for row in w:
                params.extend(row)
        for b in self.biases:
            params.extend(b)
        return params

    def unflatten(self, params: list[float]) -> None:
        """Restore weights and biases from a flat list produced by ``flatten()``."""
        idx = 0
        for w in self.weights:
            for row in w:
                for j in range(len(row)):
                    row[j] = params[idx]
                    idx += 1
        for b in self.biases:
            for j in range(len(b)):
                b[j] = params[idx]
                idx += 1

    def clone(self) -> "NeuralNetwork":
        """Return a deep copy of this network."""
        return copy.deepcopy(self)

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return len(self.flatten())
