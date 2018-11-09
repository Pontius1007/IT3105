import hexcell
import math
from copy import deepcopy
import pickle
import copy
import ujson


# State manager for HEX
class GameState:
    def __init__(self, player=1, hexBoard=None, dimensions=5, neighbours=None):
        self.player = player

        # HEX related attributes
        self.hexBoard = hexBoard
        self.dimensions = dimensions
        self.neighbours = neighbours

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

        neighbours_dict = {}

        # finds neighbor coordinates
        board_length = len(hexBoard) - 1
        for i, row in enumerate(self.hexBoard):
            for j, cell in enumerate(row):
                position = [i, j]
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
                neighbours_dict[str(position)] = neighbours
                cell.setPosition([i, j])
        self.neighbours = neighbours_dict

    # Prints the board in a diamond
    def print_hexboard(self):
        board = self.flatten()
        loops = 0
        index = 0
        down = False
        down_index = 0
        down_sec_index = 0
        max_length_string = self.dimensions * 3
        for i in range(0, self.dimensions * 2 - 1):
            row_string = ""
            if loops < self.dimensions and not down:
                loops += 1
            else:
                loops -= 1
                down = True
            for x in range(loops):
                # Prints from top to and including middle
                if not down:
                    if index == 0 and loops == 1:
                        row_string += self.return_hex_player_id(board[index])
                    # If its the first cells in a row
                    elif (index % (self.dimensions - 1) or index == 0) and x == 0:
                        index = (loops - 1) * self.dimensions
                        down_index = index
                        row_string += self.return_hex_player_id(board[index])
                    # Prints next cells in a row
                    else:
                        index -= (self.dimensions - 1)
                        row_string += self.return_hex_player_id(board[index])
                else:
                    if x == 0:
                        # If its the first node in a row
                        down_index += 1
                        down_sec_index = down_index
                        row_string += self.return_hex_player_id(board[down_index])
                    # Prints next cells in a row
                    else:
                        down_sec_index -= (self.dimensions - 1)
                        row_string += self.return_hex_player_id(board[down_sec_index])

            centered_row = row_string.center(max_length_string)
            print(centered_row)

    def return_hex_player_id(self, cell_state):
        if cell_state == '[0, 0]':
            return " 0 "
        elif cell_state == '[1, 0]':
            return " 1 "
        elif cell_state == '[0, 1]':
            return " 2 "
        else:
            raise ValueError("Board state is not recognized. Board state is: ", cell_state)

    # 3d to 1d array
    def flatten(self):
        board = self.hexBoard
        new_board = []
        for row in range(len(board)):
            for element in range(len(board)):
                state_to_string = str(board[row][element].value)
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
            neighbours = self.neighbours.get(str(current_node.position))
            for neighbour in neighbours:
                neighbour_object = self.hexBoard[neighbour[0]][neighbour[1]]
                if neighbour_object.value == [1,
                                              0] and neighbour_object not in visited_1 and neighbour_object not in \
                        unvisited_1:
                    unvisited_1.append(neighbour_object)
            visited_1.append(unvisited_1.pop(0))
            # checks if node is on opposite side for player 1
            if current_node in self.hexBoard[-1]:
                return True

        # checks for player 2
        for row in self.hexBoard:
            if row[0].value == [0, 1]:
                unvisited_2.append(row[0])
        while len(unvisited_2):
            current_node = unvisited_2[0]
            # adds unvisited neighbours
            neighbours = self.neighbours.get(str(current_node.position))
            for neighbour in neighbours:
                neighbour_object = self.hexBoard[neighbour[0]][neighbour[1]]
                if neighbour_object.value == [0,
                                              1] and neighbour_object not in visited_2 and neighbour_object not in \
                        unvisited_2:
                    unvisited_2.append(neighbour_object)
            visited_2.append(unvisited_2.pop(0))
            # checks if node is on opposite side for player 2
            if current_node in [x[-1] for x in self.hexBoard]:
                return True

        return False

    # returns next possible nodes of a node
    def next_node_states(self):
        children = []

        for i, row in enumerate(self.hexBoard):
            for j, cell in enumerate(row):
                if cell.value == [0, 0]:
                    # temp_board = deepcopy(self.hexBoard)
                    # TODO: MAKE THIS LINE FASTER
                    temp_board = pickle.loads(pickle.dumps(self.hexBoard, -1))
                    # temp_board = copy.copy(self.hexBoard)
                    # temp_board = ujson.loads(ujson.dumps(self.hexBoard))
                    # print("should print hERE")
                    # print(self.hexBoard)
                    # print(temp_board)
                    if self.player == 1:
                        temp_board[i][j].value = [1, 0]
                    else:
                        temp_board[i][j].value = [0, 1]
                    children.append(GameState(player=3 - self.player, hexBoard=temp_board, dimensions=self.dimensions,
                                              neighbours=self.neighbours))
        return children
