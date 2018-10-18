from math import *

class Node:
    def init(self, parent, visits):
        self.parent = parent
        self.visits = visits
        self.wins = 0

    def add_child(self, state):
        child = Node(state, self)

# traverse from root to node using tree policy (UCB shit)
def search(node):
    ucb = node.parent.wins/node.parent.visits + 2 * sqrt(log(node.visits)/node.parent.visits)
    return None

# generate some or all states of child states of a parent state
def expand():
    return None

# estimates value of node using default policy
def evaluate():
    return None

# pass evaluating of final state up the tree, updating data
def backpropogate():
    return None


