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
