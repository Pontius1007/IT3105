from math import *

class Node:
    def init(self, parent = None, state = None):
        self.parent = parent
        self.child_nodes = []
        self.visits = 1
        self.wins = 0
        self.state = state

    def add_child(self, state):
        child = Node(state, self)


class MCTS:

    # traverse from root to node using tree policy (UCB shit)
    def search(self, node):
        if len(node.child_nodes) == 0:
            return node
        best_child = None
        highest_ucb = 0
        for child in node.child_nodes:
            ucb = child.wins/child.visits + 2 * sqrt(log(node.visits)/child.visits)
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

class NimState:

    def nim(self, n, k):
        pieces = n
        if k < 1:
            raise Exception("k cannot be lower than 1")
        maximum_removal = k
        while pieces > 0:
            return None


class Run:
    def run(self):
        return None

