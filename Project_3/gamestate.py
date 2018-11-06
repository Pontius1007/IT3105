# State manager for NIM
class GameState:
    def __init__(self, player=1, numberofpieces=10, maxremove=2):
        self.player = player

        # NIM related attributes
        self.numberOfPieces = numberofpieces
        self.maxRemovePieces = maxremove

    def __str__(self):
        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

    def get_number_of_pieces(self):
        return self.numberOfPieces

    def set_number_of_pieces(self, pieces):
        self.numberOfPieces = pieces

    def get_max_remove_pieces(self):
        return self.maxRemovePieces

    def set_max_remove_pieces(self, maxremove):
        self.maxRemovePieces = maxremove

    def get_player(self):
        return self.player

    def set_player(self, player):
        self.player = player

    def switch_player(self, player):
        if player == 1:
            return 2
        if player == 2:
            return 1

    def game_over(self):
        unvisited_1 = []
        unvisited_2 = []
        visited_1 = []
        visited_2 = []

        # checks for player 1
        for cell in hexBoard[0]:
            if cell.value == [1,0]:
                unvisited_1.append(cell)
        while len(unvisited_1):
            # checks if node is on opposite side for player 1
            for cell in hexBoard[-1]:
                if unvisited_1[0] == cell:
                    return True
            # adds unvisited neighbors
            for neighbor in unvisited_1[0].neighbors:
                if neighbor.value == [1,0] and neighbor not in visited_1 and neighbor not in unvisited_1:
                    unvisited_1.append(neighbor)
                visited_1.append(unvisited_1.pop(0))

        # checks for player 2
        for row in hexBoard
            if row[0].value == [0,1]:
                unvisited_2.append(row[0])
        while len(unvisited_2):
            # checks if node is on opposide side for player 2
            for row in hexBoard:
                if unvisited_2[0] == row[-1]:
                    return True
            # adds unvisted neighbors
            for neighbor in unvisited_2[0].neigbors:
                if neigbor.value == [0,1] and neighbor not in visited_2 and neighbor not in unvisited_2:
                    unvisited_2.append(neighbor)
                visited_2.append(unvisited_2.pop(0))

        return false





    # returns next possible nodes of a node
    def next_node_states(self):
        children = []
        for i, row in enumerate(hexBoard):
            for j, cell in enumerate(row):
                if cell.value == [0,0]:
                    tempBoard = hexBoard
                    if self.player == 1:
                        tempBoard[i][j] == [1,0]
                    else:
                        tempBoard[i][j] == [0,1]
                    children.append((GameState(player=3 - self.player, hexBoard=tempBoard, dimensions=dimensions)))
        return children
