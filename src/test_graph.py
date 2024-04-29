import pytest

from graph import Node


def test_add():
    a = Node(1)
    b = Node(2)
    c = a + b
    assert c.data == 3
    assert c._op == "+"


def test_twinadd():
    a = Node(1)
    c = a + a
    assert c.data == 2
    assert c._op == "+"


def test_radd():
    a = Node(1)
    b = 2
    c = b + a
    assert c.data == 3
    assert c._op == "+"


def test_sub():
    a = Node(1)
    b = Node(2)
    c = a - b
    assert c.data == -1
    assert c._op == "-"


def test_twinsub():
    a = Node(1)
    c = a - a
    assert c.data == 0
    assert c._op == "-"


def test_mul():
    a = Node(2)
    b = Node(3)
    c = a * b
    assert c.data == 6
    assert c._op == "*"


def test_rmul():
    a = Node(1)
    b = 2
    c = a * b
    assert c.data == 2
    assert c._op == "*"


def test_pow():
    a = Node(3)
    b = Node(2)
    c = a**b
    assert c.data == 9
    assert c._op == "**"


def test_truediv():
    a = Node(4)
    b = 2
    c = a / b
    assert c.data == 2
    assert c._op == "/"


def test_exp():
    a = Node(0)
    c = a.exp()
    assert c.data == 1
    assert c._op == "exp"


def test_sigmoid():
    a = Node(0)
    c = a.sigmoid()
    assert c.data == 0.5
    assert c._op == "sigmoid"


def test_tanh():
    a = Node(0)
    c = a.tanh()
    assert c.data == 0
    assert c._op == "tanh"


def test_relu():
    a = Node(-3)
    c = a.relu()
    assert c.data == 0
    assert c._op == "relu"


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
