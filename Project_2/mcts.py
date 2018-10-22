from math import *
import random


# Functions related to the node tree and keeping track of the states
class Node:
    def init(self, parent=None, state=None):
        self.parent = parent
        self.child_nodes = []
        self.visits = 1
        self.wins = 0
        # How many pieces left, number of child nodes etc
        self.state = state

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_child_nodes(self):
        return self.child_nodes

    def set_child_nodes(self, child_nodes):
        self.child_nodes = child_nodes

    def get_random_child(self):
        return self.child_nodes[random.randint(0, len(self.child_nodes) - 1)]

    def add_child(self, child_node):
        self.child_nodes.append(child_node)


# MCTS spesific logic. Independent from NIM-code.
# TODO: Finish Expand, evalue and backprop.
# TODO: Create a function for finding the next move. Using all of the functions below to determine the right choice
class MCTS:

    # traverse from root to node using tree policy (UCB shit)
    def search(self, node):
        if len(node.child_nodes) == 0:
            return node
        best_child = None
        highest_ucb = 0
        for child in node.child_nodes:
            ucb = child.wins / child.visits + 2 * sqrt(log(node.visits) / child.visits)
            if ucb > highest_ucb:
                best_child = child
        return self.search(best_child)

    # generate some or all states of child states of a parent state
    def expand(self, parent, children):
        for child in children:
            node = Node(parent)
            parent.child_nodes.append(node)

    # estimates value of node using default policy
    def evaluate(self):
        return None

    # pass evaluating of final state up the tree, updating data
    def backpropogate(self):
        return None


# State manager for NIM
class GameState:
    def __init__(self, player=1, numberofpieces=10, maxremove=2):
        self.player = player
        self.wins = 0
        self.visits = 1

        # NIM related attributes
        self.numberOfPieces = numberofpieces
        self.maxRemovePieces = maxremove

    def get_wins(self):
        return self.wins

    def set_wins(self, wins):
        self.wins = wins

    def get_visits(self):
        return self.visits

    def set_visits(self, visits):
        self.visits = visits

    def get_number_of_pieces(self):
        return self.numberOfPieces

    def set_number_of_pieces(self, pieces):
        self.numberOfPieces = pieces

    def get_max_remove_pieces(self):
        return self.maxRemovePieces

    def set_max_remove_pieces(self, maxremove):
        self.maxRemovePieces = maxremove

    def game_over(self):
        return True if self.numberOfPieces == 0 else False



class Run:
    def run(self):
        return None
