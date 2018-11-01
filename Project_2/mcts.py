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
        # temp = self.get_state().next_state_moves()
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

    def next_node_states(self):

        children = []
        for i in range(1, min(self.numberOfPieces, self.maxRemovePieces) + 1):
            children.append(GameState(player=3-self.player, numberofpieces=self.numberOfPieces - i, maxremove=self.maxRemovePieces))
        return children

    def play_random_move(self, next_state_moves):
        return next_state_moves[random.randint(0, len(next_state_moves) - 1)]


# MCTS spesific logic. Independent from NIM-code.
class MCTS:

    # returns ucb value
    def ucb(self, node, child):
        qsa = child.get_wins() / child.get_visits()
        new_qsa = qsa * -1
        # print("old: " + str(qsa) + " new: " + str(new_qsa))
        usa = 1 * sqrt(log(node.get_visits()) / (1 + child.get_visits()))
        return qsa + usa

    # traverse from root to node using tree policy (UCB)
    def search(self, node):
        if len(node.child_nodes) == 0:
            return node
        best_child = None
        highest_ucb = -float('inf')
        for child in node.child_nodes:
            ucb = MCTS().ucb(node, child)
            if ucb > highest_ucb:
                best_child = child
                highest_ucb = ucb
        return self.search(best_child)

        # tempNode = node
        # while tempNode is not None:
        #     while(len(tempNode.children)):
        #         tempNode = tempNode.
        #         ucb = MCTS().ucb(node, )

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
    def backpropogate(self, node, winner, player):
        unupdated_node = node
        if winner == 1:
            while unupdated_node is not None:
                visits = unupdated_node.get_visits()
                unupdated_node.set_visits(visits + 1)

                wins = unupdated_node.get_wins()
                unupdated_node.set_wins(wins + 1)
                unupdated_node = unupdated_node.get_parent()

        if winner == 2:
            while unupdated_node is not None:
                visits = unupdated_node.get_visits()
                unupdated_node.set_visits(visits + 1)

                wins = unupdated_node.get_wins()
                unupdated_node.set_wins(wins - 1)
                unupdated_node = unupdated_node.get_parent()


class Run:
    def run(self, batch, starting_player, simulations, numberofpieces, maxremove):

        total_wins = 0

        for i in range(0, batch):
            if starting_player == 'mix':
                starting_player = random.randint(1,2)

            root_node = Node(parent=None, state=GameState(player=starting_player, numberofpieces=numberofpieces, maxremove=maxremove))
            root_player = root_node.get_state().get_player()
            game_over = False

            while not game_over:
                print("")
                print("")
                print("")

                
                batch_node = Run().find_move(root_node, simulations)

                print("Move node:")
                print(batch_node)
                print("Pieces left: " + str(batch_node.get_state().get_number_of_pieces()))
                print("Children:")

                next_move = None
                highest_ratio = -float('inf')
                lowest_ratio = float('inf')
                current_player = batch_node.get_state().get_player()


                for child in batch_node.child_nodes:
                    ratio = float(child.get_wins())/float(child.get_visits())

                    if current_player == 1:
                        print("Player: " + str(current_player) + " Ratio: " + str(ratio) + " highest: "
                          + str(highest_ratio) + "    WINS:" + str(child.get_wins()) + " VISITS:" + str(child.get_visits() - 1) + "    pieces left: " + str(child.get_state().get_number_of_pieces()))
                        if ratio > highest_ratio:
                            highest_ratio = ratio
                            next_move = child
                    else:
                        print("Player: " + str(current_player) + " Ratio: " + str(ratio) + " lowest: "
                          + str(lowest_ratio) + "    WINS:" + str(child.get_wins()) + " VISITS:" + str(child.get_visits() - 1) + "    pieces left: " + str(child.get_state().get_number_of_pieces()))
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


    def find_move(self, node, simulations):
        move_node = node
        player = node.get_state().get_player()

        for i in range(0, simulations):

            # this chooses the node with a random policy instead of UCT policy
            # temp_node = move_node
            # best_node = temp_node


            # this searches through tree based on UCT value
            best_node = MCTS().search(move_node)

            # while len(temp_node.get_child_nodes()) > 0:
            #     best_node = random.choice(temp_node.get_child_nodes())
            #     temp_node = best_node


            # expands the node with children if there are possible states
            MCTS().expand(best_node)

            # if node was expanded, choose a random child to evaluate
            if len(best_node.get_child_nodes()) > 0:
                best_node = random.choice(best_node.get_child_nodes())

            # simulates winner
            winner = MCTS().evaluate(best_node)

            # traverses up tree with winner
            MCTS().backpropogate(best_node, winner, player)

        return move_node


Run().run(batch=1, starting_player=1, simulations=100, numberofpieces=6, maxremove=3)
