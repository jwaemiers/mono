from truss_components import Node, Element


class Truss:
    def __init__(self, nodes: list[Node], elements: list[Element]) -> None:
        self.nodes: list[Node] = nodes
        self.elements: list[Element] = elements


class TrussBuilder:
    def load_truss(self) -> None:
        raise NotImplementedError

    def build_truss(self) -> Truss:
        raise NotImplementedError
