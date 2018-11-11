from math import *


# MCTS specific logic. Independent from NIM-code.
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
        while not node.state.game_over():
            node = node.get_random_child()
        winner = 3 - node.get_state().get_player()
        return winner

    def ANET_evaluate(self, ANET, simple_board_state):
        ANET_pred = ANET.do_prediction(simple_board_state)


    # pass evaluating of final state up the tree, updating data
    def backpropogate(self, node, winner, batch_player):
        while node is not None:
            if winner == batch_player:
                node.set_wins(node.get_wins() + 1)
            else:
                node.set_wins(node.get_wins() - 1)
            node.set_visits(node.get_visits() + 1)
            node = node.parent
