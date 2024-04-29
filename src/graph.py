import math


class Node:
    """
    A Node is a node in a computational graph.
    It contains the data, the children, and the operation.
    """

    def __init__(self, data, children=(), label="", op=""):
        self.data = data  # Scalar value
        self._prev = set(children)  # Previous nodes in the graph
        self._grad = 0.0  # Gradient
        self.backward = lambda: None  # Backward pass function
        # Some cosmetic properties
        self.label = label
        self._op = op

    def __repr__(self):
        return f"Node(data={self.data})"

    def __add__(self, other):
        """
        +
        """
        # Instantiate a Value object when passing in a primitive datatype
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, children=(self, other), op="+")
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
        return out

    def sigmoid(self):
        out = Node(1 / (1 + math.exp(-self.data)), children=(self,), op="sigmoid")
        return out

    def tanh(self):
        out = Node(
            (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1),
            children=(self,),
            op="tanh",
        )
        return out

    def relu(self):
        out = Node(self.data if self.data > 0 else 0, children=(self,), op="relu")
        return out
