import hexCell
import math


# State manager for NIM
class GameState:
    def __init__(self, player=1, dimensions=5):
        self.player = player

        # HEX related attributes
        self.dimensions = dimensions
        self.hexBoard = []

    def initialize_hexboard(self):
        dimensions = self.dimensions
        hexBoard = []
        for row in range(dimensions):
            row_list = []
            for element in range(dimensions):
                row_list.append(hexCell.HexCell())
            hexBoard.append(row_list)
        self.hexBoard = hexBoard

    def print_hexboard(self):
        board = self.flatten()
        loops = 0
        index = 0
        down = False
        max_length_string = self.dimensions * 3
        for i in range(0, self.dimensions * 2 - 1):
            row_string = ""
            if loops < self.dimensions and not down:
                loops += 1
            else:
                loops -= 1
                down = True
            for x in range(loops):
                if board[index] == '[0, 0]':
                    row_string += " 0 "
                elif board[index] == '[1, 0]':
                    row_string += " 1 "
                elif board[index] == '[0, 1]':
                    row_string += " 2 "
                else:
                    raise ValueError("Board state is not recognized. Board state is: ", board[index])
            centered_row = row_string.center(max_length_string)
            print(centered_row)

    def flatten(self):
        board = self.hexBoard
        new_board = []
        for row in range(len(board)):
            for element in range(len(board)):
                state_to_string = str(board[row][element].get_value())
                new_board.append(state_to_string)
        return new_board

    def __str__(self):
        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

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
