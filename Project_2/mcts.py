from math import *
import random


# Functions related to the node tree and keeping track of the states
class Node:
    def __init__(self, parent=None, state=None):
        self.parent = parent
        self.child_nodes = []
        # How many pieces left, number of child nodes etc
        self.state = state

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
        return self.child_nodes

    def set_child_nodes(self, child_nodes):
        self.child_nodes = child_nodes

    def get_random_child(self):
        return self.child_nodes[random.randint(0, len(self.child_nodes) - 1)]

    def add_child(self, child_node):
        self.child_nodes.append(child_node)

    #TODO Return the child with highest UTC score, if we need it.


# MCTS spesific logic. Independent from NIM-code.
# TODO: Finish Expand, evalue and backprop.
# TODO: Create a function for finding the next move. Using all of the functions below to determine the right choice
# TODO: Do we need to elaborate on the UCB search algorithm?
class MCTS:

    # returns ucb value
    def ucb(self, node, child):
        return child.get_state().get_wins() / child.get_state().get_visits() + 2 * sqrt(log(node.get_state().get_visits()) / child.get_state().get_visits())

    # traverse from root to node using tree policy (UCB shit)
    def search(self, node):
        if len(node.get_child_nodes()) == 0:
            return node
        best_child = None
        highest_ucb = -999999
        for child in node.child_nodes:
            ucb = MCTS().ucb(node, child)
            if ucb > highest_ucb:
                best_child = child
        return self.search(best_child)

    # generate some or all states of child states of a parent state
    def expand(self, node):
        possible_moves = node.get_state().next_state_moves()
        for move in possible_moves:
            child_node = Node(parent=node, state=move)
            node.add_child(child_node)
        return node

    # estimates value of node using default policy
    def evaluate(self, node):
        simulated_node = node
        state = simulated_node.get_state()

        while not state.game_over():
            state = state.play_random_move(state.next_state_moves())
            state.set_player(state.switch_player(state.get_player()))
        
        winner = 3 - state.get_player()
        return winner
        

    # pass evaluating of final state up the tree, updating data
    def backpropogate(self, node, winner):
        unupdated_node = node
        while unupdated_node is not None:
            visits = unupdated_node.get_state().get_visits()
            unupdated_node.get_state().set_visits(visits + 1)

            if unupdated_node.get_state().get_player() == winner:
                wins = visits = unupdated_node.get_state().get_wins()
                unupdated_node.get_state().set_wins(wins + 1)

            unupdated_node = unupdated_node.get_parent()


# State manager for NIM
class GameState:
    def __init__(self, player=1, numberofpieces=10, maxremove=2):
        self.player = player
        self.wins = 0
        self.visits = 1

        # NIM related attributes
        self.numberOfPieces = numberofpieces
        self.maxRemovePieces = maxremove

    def __str__(self):
        return ' ,  '.join(['{key} = {value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

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

    #TODO Needs testing
    def next_state_moves(self):
        max_states = self.maxRemovePieces
        current_player = self.player
        all_possible_states = []

        for i in range(1, max_states + 1):
            if (self.numberOfPieces - i) >= 0:
                all_possible_states.append(GameState(player=current_player, numberofpieces=self.numberOfPieces-i,
                                                 maxremove=max_states))
            else:
                break
        return all_possible_states

    def play_random_move(self, next_state_moves):
        return next_state_moves[random.randint(0, len(next_state_moves) - 1)]


class Run:
    def run(self, batch, starting_player, simulations, numberofpieces, maxremove):
        root_node = Node(parent=None, state=GameState(player=1, numberofpieces=10, maxremove=3))
        for simulation in range(0, simulations):
            print("")
            print("")
            print("")
            print("ITERATION")
            print(root_node)
            print("length of root: " + str(len(root_node.get_child_nodes())))
            best_node = MCTS().search(root_node)
            print(best_node)
            print("length: " + str(len(best_node.get_child_nodes())))
            # it doesn't even enter this loop
            while len(best_node.get_child_nodes()) > 0:
                print("DOES IT GO INSIDE OF HERE")
                best_node = MCTS().search(best_node)
                print("Best node set to: " + str(best_node))
                print("WAS IT DONE")
            MCTS().expand(best_node)
            print("node EXPANDED")
            print(("length: ") + str(len(best_node.get_child_nodes())))

            for child in best_node.get_child_nodes():
                wins = 0                
                for i in range(0, simulations):
                    winner = MCTS().evaluate(child)
                    if winner == 1:
                        wins += 1
                wins = float(float(wins)/float(simulations))
                MCTS().backpropogate(child, wins)



        # while not start_node.get_state().game_over():
        #     player = start_node.get_state().get_player()



Run().run(batch=10, starting_player="mix", simulations=30, numberofpieces=10, maxremove=3)
