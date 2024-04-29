from graph import Node


def main(a, b, c):
    # Base case
    node_a = Node(a)
    node_b = Node(b)
    node_c = Node(c)
    # To maintain purity, only have 1 function per line
    node_e = node_a * node_b
    node_d = node_e + node_c
    print(node_d)
    print(node_d._prev)
    print(node_d._op)
    print(node_e)
    print(node_e._prev)
    print(node_e._op)

    # Initialise with only primitives
    node_a = Node(a)
    node_b = b
    node_c = c

    # To maintain purity, only have 1 function per line
    node_e = node_a * node_b
    node_d = node_e + node_c
    print(node_d)
    print(node_d._prev)
    print(node_d._op)
    print(node_e)
    print(node_e._prev)
    print(node_e._op)

    # Swap the order
    node_a = Node(a)
    node_b = b
    node_c = c

    # To maintain purity, only have 1 function per line
    node_e = node_b * node_a
    node_d = node_e + node_c
    print(node_d)
    print(node_d._prev)
    print(node_d._op)
    print(node_e)
    print(node_e._prev)
    print(node_e._op)

    # Power
    node_a = Node(a)
    node_b = Node(b)

    node_e = node_a**node_b
    print(node_e)
    print(node_e._prev)
    print(node_e._op)

    # Division
    node_a = Node(a)
    node_b = Node(b)

    node_e = node_a / node_b
    print(node_e)
    print(node_e._prev)
    print(node_e._op)

    # Exponential
    node_a = Node(a)

    node_e = node_a.exp()
    print(node_e)
    print(node_e._prev)
    print(node_e._op)


if __name__ == "__main__":
    main(2, -3, 10)
