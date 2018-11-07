class HexCell:
    def __init__(self):
        self.value = [0, 0]
        self.neighbours = []
        self.position = []

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_neighbours(self):
        return self.neighbours

    def set_neighbours(self, neighbours):
        self.neighbours = neighbours
        return self.neighbours

    def addNeighbor(self, hexcell):
        self.neighbours.append(hexcell)

    def setPosition(self, position):
        self.position = position
