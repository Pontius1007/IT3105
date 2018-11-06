import hexcell
import math


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
        for i, row in enumerate(self.hexBoard):
            for j, cell in enumerate(row):
                neighbours = []
                if i - 1 is not None:
                    neighbours.append([i - 1, j])
                if i - 1 is not None and j + 1 is not None:
                    neighbours.append([i - 1, j + 1])
                if j - 1 is not None:
                    neighbours.append([i, j - 1])
                if j + 1 is not None:
                    neighbours.append([i, j + 1])
                if i + 1 is not None and j - 1 is not None:
                    neighbours.append([i + 1, j - 1])
                if i + 1 is not None:
                    neighbours.append([i + 1, j])
                cell.set_neighbours(neighbours)


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
            # checks if node is on opposite side for player 1
            for cell in self.hexBoard[-1]:
                if unvisited_1[0] == cell:
                    return True
            # adds unvisited neighbours
            for neighbour in unvisited_1[0].neighbours:
                neighbour_object = self.hexBoard[neighbour]
                if neighbour_object.value == [1, 0] and neighbour_object not in visited_1 and neighbour_object not in unvisited_1:
                    unvisited_1.append(neighbour_object)
                visited_1.append(unvisited_1.pop(0))

        # checks for player 2
        for row in self.hexBoard:
            if row[0].value == [0, 1]:
                unvisited_2.append(row[0])
        while len(unvisited_2):
            # checks if node is on opposite side for player 2
            for row in self.hexBoard:
                if unvisited_2[0] == row[-1]:
                    return True
            # adds unvisited neighbours
            for neighbour in unvisited_2[0].neigbors:
                neighbour_object = self.hexBoard[neighbour]
                if neighbour_object.value == [0, 1] and neighbour_object not in visited_2 and neighbour_object not in unvisited_2:
                    unvisited_2.append(neighbour_object)
                visited_2.append(unvisited_2.pop(0))

        return False

    # returns next possible nodes of a node
    def next_node_states(self):
        children = []
        for i, row in enumerate(self.hexBoard):
            print(row)
            for j, cell in enumerate(row):
                if cell.value == [0, 0]:
                    temp_board = self.hexBoard
                    if self.player == 1:
                        temp_board[i][j] = [1, 0]
                    else:
                        temp_board[i][j] = [0, 1]
                    children.append(
                        (GameState(player=3 - self.player, hexBoard=temp_board, dimensions=self.dimensions)))
        return children


gs = GameState(dimensions=3)
gs.initialize_hexboard()
board = gs.get_hexboard()
print(gs.complex_to_simple_hexboard(board))
