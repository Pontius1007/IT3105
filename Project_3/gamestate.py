import hexcell
import math
from copy import deepcopy


# State manager for HEX
class GameState:
    def __init__(self, player=1, hexBoard=None, dimensions=5):
        self.player = player

        # HEX related attributes
        self.hexBoard = hexBoard
        self.dimensions = dimensions

    def __str__(self):
        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

    def initialize_hexboard(self):
        dimensions = self.dimensions
        hexBoard = []
        for row in range(dimensions):
            row_list = []
            for element in range(dimensions):
                row_list.append(hexcell.HexCell())
            hexBoard.append(row_list)
        self.hexBoard = hexBoard

        # finds neighbor coordinates
        board_length = len(hexBoard) - 1
        for j, row in enumerate(self.hexBoard):
            for i, cell in enumerate(row):
                neighbours = []
                if 0 <= i - 1 <= board_length:
                    neighbours.append([i - 1, j])
                if 0 <= i - 1 <= board_length and 0 <= j + 1 <= board_length:
                    neighbours.append([i - 1, j + 1])
                if 0 <= j - 1 <= board_length:
                    neighbours.append([i, j - 1])
                if 0 <= j + 1 <= board_length:
                    neighbours.append([i, j + 1])
                if 0 <= i + 1 <= board_length and 0 <= j - 1 <= board_length:
                    neighbours.append([i + 1, j - 1])
                if 0 <= i + 1 <= board_length:
                    neighbours.append([i + 1, j])
                cell.set_neighbours(neighbours)
                cell.setPosition([i, j])

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
                    raise ValueError("Board state is not recognized. Board state is: ", board[x])
                index += 1
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

    # Converts the 3d array into a simple 1d array to be used in the NN. Last two bits denotes player
    def complex_to_simple_hexboard(self, board):
        simple_array = []
        player = self.get_player()
        for row in range(len(board)):
            for element in range(len(board)):
                for value in board[row][element].value:
                    simple_array.append(value)
        if player == 1:
            simple_array.extend((1, 0))
        elif player == 2:
            simple_array.extend((0, 1))
        else:
            raise ValueError("Error: Player not recognized")
        return simple_array

    def get_hexboard(self):
        return self.hexBoard

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
        for cell in self.hexBoard[0]:
            if cell.value == [1, 0]:
                unvisited_1.append(cell)
        while len(unvisited_1):
            current_node = unvisited_1[0]
            # adds unvisited neighbours
            for neighbour in current_node.neighbours:
                neighbour_object = self.hexBoard[neighbour[0]][neighbour[1]]
                if neighbour_object.value == [1,
                                              0] and neighbour_object not in visited_1 and neighbour_object not in unvisited_1:
                    unvisited_1.append(neighbour_object)
            visited_1.append(unvisited_1.pop(0))
            # checks if node is on opposite side for player 1
            if current_node in self.hexBoard[-1]:
                return True
        print("Board:")
        for thing in self.hexBoard:
            print([x.value for x in thing])

        # checks for player 2
        for row in self.hexBoard:
            if row[0].value == [0, 1]:
                unvisited_2.append(row[0])
        while len(unvisited_2):
            current_node = unvisited_2[0]
            # adds unvisited neighbours
            for neighbour in current_node.neighbours:
                neighbour_object = self.hexBoard[neighbour[0]][neighbour[1]]
                if neighbour_object.value == [0,
                                              1] and neighbour_object not in visited_2 and neighbour_object not in unvisited_2:
                    unvisited_2.append(neighbour_object)
            visited_2.append(unvisited_2.pop(0))
            # checks if node is on opposite side for player 2
            print("value")
            print([x[-1].value for x in self.hexBoard])
            if current_node in [x[-1] for x in self.hexBoard]:
                return True



        return False

    # returns next possible nodes of a node
    def next_node_states(self):
        children = []
        for i, row in enumerate(self.hexBoard):
            for j, cell in enumerate(row):
                if cell.value == [0, 0]:
                    temp_board = deepcopy(self.hexBoard)
                    if self.player == 1:
                        temp_board[i][j].value = [1, 0]
                    else:
                        temp_board[i][j].value = [0, 1]
                    children.append(GameState(player=3 - self.player, hexBoard=temp_board, dimensions=self.dimensions))
        return children
