import math
import random


class Node:
    """
    A Node is a node in a computational graph.
    It contains the data, the children, and the operation.
    """

    def __init__(self, data, children=(), label="", op=""):
        self.data = data  # Scalar value
        self._prev = set(children)  # Previous nodes in the graph
        self._grad = 0.0  # Gradient
        self._backward = lambda: None  # Backward pass function
        # Some cosmetic properties
        self.label = label
        self._op = op

    def __repr__(self):
        return f"Node(data={self.data}, label={self.label}, grad={self._grad})"

    def __add__(self, other):
        """
        +
        """
        # Instantiate a Value object when passing in a primitive datatype
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, children=(self, other), op="+")

        def _backward():
            self._grad += 1 * out._grad
            other._grad += 1 * out._grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        """
        Allow __add__ to work with the expression 2 + Value(1) with a primitive datatype first
        """
        return self + other

    def __sub__(self, other):
        """
        -
        """
        # To compute self - other is the same as computing
        # self + (-1 * other)
        out = self + (-1 * other)
        out._op = "-"
        return out

    def __mul__(self, other):
        """
        *
        """
        # Instantiate a Value object when passing in a primitive datatype
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, children=(self, other), op="*")

        def _backward():
            self._grad += other.data * out._grad
            other._grad += self.data * out._grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        """
        Allow __mul__ to work with the expression 2 * Value(1) with a primitive datatype first
        """
        return self * other

    def __pow__(self, other):
        """
        **
        """
        # Instantiate a Value object when passing in a primitive datatype
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data**other.data, children=(self,), op="**")

        def _backward():
            self._grad += other.data * (self.data ** (other.data - 1)) * out._grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """
        /
        """
        # To compute self / other is the same as computing
        # self * (other)**-1
        out = self * other**-1
        out._op = "/"
        return out

    def exp(self):
        """
        exp(self)
        """
        out = Node(math.exp(self.data), children=(self,), op="exp")

        def _backward():

            self._grad += out._grad * math.exp(self.data)

        out._backward = _backward
        return out

    def sigmoid(self):
        out = Node(1 / (1 + math.exp(-self.data)), children=(self,), op="sigmoid")

        def _backward():
            self._grad += out.data * (1 - out.data)

        out._backward = _backward
        return out

    def tanh(self):
        out = Node(
            (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1),
            children=(self,),
            op="tanh",
        )

        def _backward():
            self._grad += out._grad * (1 - out.data**2)

        out._backward = _backward
        return out

    def relu(self):
        out = Node(self.data if self.data > 0 else 0, children=(self,), op="relu")
        return out

    def times_minus_one(self):
        out = Node(-1 * self.data, children=(self,), op="*-1")

        def _backward():
            self._grad += -1 * out._grad

        out._backward = _backward
        return out

    def plus_one(self):
        out = Node(1 + self.data, children=(self,), op="+1")

        def _backward():
            self._grad = out._grad * 1

        out._backward = _backward
        return out

    def reciprocal(self):
        out = Node(1 / self.data, children=(self,), op="1/x")

        def _backward():
            self._grad += (-1) * self.data ** (-2)

        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self._grad = 1
        # print("order of backprop: ")
        # for v in reversed(topo):
        #     print(v)
        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()


class Neuron:
    """
    A neuron is a node in a neural network.
    It takes in an x array, adds them to the w to get a weighted sum,
    adds the bias, and then applies a tanh activation function.
    """

    def __init__(self, num_dimensions):
        # Initialise the weights and bias
        self.w = [Node(random.uniform(-1, 1)) for _ in range(num_dimensions)]
        self.b = Node(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """
    A layer is a list of neurons.
    It takes in an array x, and performs the weighted sum
    for all neurons in the layer
    """

    def __init__(self, n_input_dims, n_output_dims):
        self.neurons = [Neuron(n_input_dims) for _ in range(n_output_dims)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    An MLP is a list of layers.
    It takes in an array x, and performs the weighted sum
    for all neurons in the layer, for all layers in the network
    """

    def __init__(self, n_input_dims, n_output_dims_list):
        network = [n_input_dims] + n_output_dims_list
        self.layers = [
            Layer(network[i], network[i + 1]) for i in range(len(n_output_dims_list))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def get_layers(self):
        return [layer for layer in self.layers]
