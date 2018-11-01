from math import *
import random


# Functions related to the node tree and keeping track of the states
class Node:
    def __init__(self, parent=None, state=None):
        self.parent = parent
        self.child_nodes = []
        # How many pieces left, number of child nodes etc
        self.state = state
        self.wins = 0
        self.visits = 1

    def __str__(self):
        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

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
        for state in self.state.next_node_states():
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
        return True if self.numberOfPieces <= 0 else False

    # returns next possible nodes of a node
    def next_node_states(self):
        children = []
        for i in range(1, min(self.numberOfPieces, self.maxRemovePieces) + 1):
            children.append(GameState(player=3-self.player, numberofpieces=self.numberOfPieces - i, maxremove=self.maxRemovePieces))
        return children


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
    def evaluate(self, node):
        while not node.state.game_over():
            node = node.get_random_child()
        winner = 3 - node.get_state().get_player()
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
    def run(self, batch, starting_player, simulations, numberofpieces, maxremove):

        total_wins = 0

        for i in range(0, batch):
            if starting_player == 'mix':
                starting_player = random.randint(1,2)

            root_node = Node(parent=None, state=GameState(player=starting_player, numberofpieces=numberofpieces, maxremove=maxremove))
            batch_player = starting_player

            game_over = False

            while not game_over:
                
                batch_node = Run().find_move(root_node, simulations, batch_player)

                next_move = None
                highest_ratio = -float('inf')
                lowest_ratio = float('inf')
                current_player = batch_node.get_state().get_player()


                for child in batch_node.child_nodes:
                    ratio = float(child.get_wins())/float(child.get_visits())

                    if current_player == batch_player:
                        if ratio > highest_ratio:
                            highest_ratio = ratio
                            next_move = child
                    else:
                       if ratio < lowest_ratio:
                            lowest_ratio = ratio
                            next_move = child

                root_node = Node(state=GameState(player=(3 - current_player), numberofpieces=next_move.get_state().get_number_of_pieces(), maxremove=maxremove))
                
                if root_node.get_state().game_over():
                    winner = 3 - root_node.get_state().get_player()
                    if starting_player == winner:
                        total_wins += 1
                    game_over = True

        print("Won " + str(total_wins) + " times out of " + str(batch) + " batches.")


    def find_move(self, node, simulations, batch_player):
        move_node = node
        player = node.get_state().get_player()
        opposing_player = player != batch_player

        for i in range(0, simulations):

            # this searches through tree based on UCT value
            best_node = MCTS().search(move_node, batch_player)

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


Run().run(batch=10, starting_player=1, simulations=1000, numberofpieces=12, maxremove=3)
