from math import *
import random
from copy import deepcopy


# Functions related to the node tree and keeping track of the states
class Node:
    def __init__(self, parent=None, state=None):
        self.parent = parent
        self.child_nodes = []
        # How many pieces left, number of child nodes etc
        self.state = state
        self.wins = 0
        self.visits = 1

    # def __str__(self):
    #     return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_child_nodes(self):
        child_nodes = []
        for state in self.state.next_node_states()[0]:
            child_nodes.append(Node(parent=self, state=state))
        return child_nodes

    def set_child_nodes(self, child_nodes):
        self.child_nodes = child_nodes

    def get_random_child(self):
        temp = self.child_nodes
        if not len(temp):
            temp = self.get_child_nodes()
        return random.choice(temp)

    def add_child(self, child_node):
        self.child_nodes.append(child_node)

    def get_wins(self):
        return self.wins

    def set_wins(self, wins):
        self.wins = wins

    def get_visits(self):
        return self.visits

    def set_visits(self, visits):
        self.visits = visits


# State manager for NIM
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
                row_list.append(HexCell())
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
        states = []

        for i, row in enumerate(self.hexBoard):
            for j, cell in enumerate(row):
                if cell.value == [0, 0]:
                    # add state index
                    states.append(1)

                    temp_board = deepcopy(self.hexBoard)
                    # TODO: MAKE THIS LINE FASTER
                    # temp_board = pickle.loads(pickle.dumps(self.hexBoard, -1))
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
                else:
                    states.append(0)

        return children, states


# MCTS spesific logic. Independent from NIM-code.
class MCTS:

    # returns ucb value
    def ucb(self, node, child, opposing_player):
        qsa = child.get_wins() / child.get_visits()
        usa = 2 * sqrt(log(node.get_visits()) / (1 + child.get_visits()))
        if opposing_player:
            qsa *= -1
        return qsa + usa

    # traverse from root to node using tree policy (UCB)
    def search(self, node, batch_player):
        if len(node.child_nodes) == 0:
            return node

        best_child = node
        highest_ucb = -float('inf')
        for child in node.child_nodes:
            opposing_player = node.state.get_player() != batch_player
            ucb = MCTS().ucb(node, child, opposing_player)
            if ucb > highest_ucb:
                best_child = child
                highest_ucb = ucb
        return self.search(best_child, batch_player)

    # generate some or all states of child states of a parent state
    def expand(self, node):
        if len(node.child_nodes):
            return None
        node.child_nodes = node.get_child_nodes()

    # estimates value of node using default policy
    # Rollout - picks a random child until the game is over. Returns winner
    def evaluate(self, node):
        # print("gameboard")
        # print(node.state.flatten())
        print(node.state.game_over())
        while not node.state.game_over():
            node = node.get_random_child()
        winner = node.get_state().get_player()
        return winner

    # pass evaluating of final state up the tree, updating data
    def backpropogate(self, node, winner, batch_player):
        while node is not None:
            if winner == batch_player:
                node.set_wins(node.get_wins() + 1)
            else:
                node.set_wins(node.get_wins() - 1)
            node.set_visits(node.get_visits() + 1)
            node = node.parent


class Run:
    def run(self, batch, starting_player, simulations, numberofpieces, maxremove, hex_dimensions, verbose=False,
            mix=False):

        total_wins_player1 = 0
        total_wins_player2 = 0
        mix = False
        if starting_player == 'mix':
            mix = True

        for i in range(0, batch):
            if mix:
                starting_player = random.randint(1, 2)
                print(starting_player)
            else:
                starting_player = starting_player

            root_node = Node(parent=None,
                             state=GameState(player=starting_player,
                                             dimensions=hex_dimensions))
            root_node.state.initialize_hexboard()
            batch_player = starting_player

            game_over = False

            while not game_over:

                batch_node = Run().find_move(root_node, simulations, batch_player)

                next_move = None
                highest_ratio = -float('inf')
                lowest_ratio = float('inf')
                current_player = batch_node.get_state().get_player()

                print("")
                print("")
                print("")
                for child in batch_node.child_nodes:
                    ratio = float(child.get_wins()) / float(child.get_visits())

                    children = [x for x in child.state.hexBoard]
                    new_children = []
                    for thing in children:
                        for another in thing:
                            new_children.append(another)

                    print("Child " + str([x.value for x in new_children]) + " had ratio " + str(
                        ratio) + " with wins/visits " + str(
                        child.get_wins()) + " / " + str(child.get_visits()))

                    if current_player == batch_player:
                        if ratio > highest_ratio:
                            highest_ratio = ratio
                            next_move = child
                    else:
                        if ratio < lowest_ratio:
                            lowest_ratio = ratio
                            next_move = child
                if verbose:
                    print("Player " + str(current_player) + " selected " + str(
                        batch_node.state.get_number_of_pieces() - next_move.state.get_number_of_pieces()) + " pieces."
                          + " There are " + str(next_move.state.get_number_of_pieces()) + " pieces left.")

                # root_node = Node(state=GameState(player=(3 - current_player),
                #                                  dimensions=hex_dimensions))
                # root_node.state.initialize_hexboard()
                # root_node.state.hexBoard = next_move.state.hexBoard

                root_node = next_move
                root_node.state.print_hexboard()

                if root_node.get_state().game_over():
                    winner = 3 - root_node.get_state().get_player()
                    if verbose:
                        print("Player " + str(winner) + " wins.")
                        print("")
                    if winner == 1:
                        total_wins_player1 += 1
                    if winner == 2:
                        total_wins_player2 += 1
                    game_over = True
        print("")
        print("Player 1" + " won " + str(total_wins_player1) + " times out of " + str(batch) + " batches." + " (" + str(
            100 * total_wins_player1 / batch) + "%)")
        print("Player 2" + " won " + str(total_wins_player2) + " times out of " + str(batch) + " batches." + " (" + str(
            100 * total_wins_player2 / batch) + "%)")

    def find_move(self, node, simulations, batch_player):
        move_node = node
        player = node.get_state().get_player()
        print("player", player)

        print("THIS IS WHAT IT IS SUPPOSED TO BE", move_node.state.flatten())

        for i in range(0, simulations):

            # this searches through tree based on UCT value
            best_node = MCTS().search(move_node, batch_player)
            print(best_node.state.flatten())

            # expands the node with children if there are possible states
            MCTS().expand(best_node)

            # if node was expanded, choose a random child to evaluate
            if len(best_node.get_child_nodes()) > 0:
                best_node = random.choice(best_node.get_child_nodes())

            # simulates winner
            winner = MCTS().evaluate(best_node)

            # traverses up tree with winner
            MCTS().backpropogate(best_node, winner, batch_player)

        return move_node


class HexCell:
    def __init__(self):
        self.value = [0, 0]
        self.position = []

    def setPosition(self, position):
        self.position = position


Run().run(batch=1, starting_player=1, simulations=50, numberofpieces=14, maxremove=3, verbose=False, hex_dimensions=2)
