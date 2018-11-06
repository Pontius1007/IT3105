class HexCell:
    def __init__(self):
        self.value = [0, 0]
        self.neighbour = []

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_neighbour(self):
        return self.neighbour

    def addNeighbor(self, hexcell):
        self.neighbour.append(hexcell)
