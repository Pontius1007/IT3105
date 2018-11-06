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
        return True if self.numberOfPieces <= 0 else False

    # returns next possible nodes of a node
    def next_node_states(self):
        children = []
        for i in range(1, min(self.numberOfPieces, self.maxRemovePieces) + 1):
            children.append(GameState(player=3 - self.player, numberofpieces=self.numberOfPieces - i,
                                      maxremove=self.maxRemovePieces))
        return children
