import math
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
    a = Node(4)
    c = a.exp()
    assert c.data == math.exp(4)
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


def test_times_minus_one():
    a = Node(3)
    c = a.times_minus_one()
    assert c.data == -3
    assert c._op == "*-1"


def test_plus_one():
    a = Node(3)
    c = a.plus_one()
    assert c.data == 4
    assert c._op == "+1"


def test_reciprocal():
    a = Node(3)
    c = a.reciprocal()
    assert c.data == 1 / 3
    assert c._op == "1/x"
