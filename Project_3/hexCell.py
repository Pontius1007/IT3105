class HexCell:
    def __init__(self):
        self.value = [0, 0]
        self.neighbours = []

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_neighbour(self):
        return self.neighbours

    def set_neighbours(self, neighbours):
        self.neighbours = neighbours
