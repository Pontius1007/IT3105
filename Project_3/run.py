import random
import node
import gamestate
import mcts
import cProfile
from input_run_handler import *


class Ann():
    # Class for keeping track of the case manager and the models.
    def set_cman(self, cman):
        self.cman = cman

    def get_cman(self):
        return self.cman

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


class Run:
    def run(self, batch, starting_player, simulations, dimensions, verbose=False):

        total_wins_player1 = 0
        total_wins_player2 = 0
        mix = False

        # ANN variables
        ann = Ann()
        irh = InputRunHandler(ann)

        if starting_player == 'mix':
            mix = True

        for i in range(0, batch):
            if mix:
                starting_player = random.randint(1, 2)
                print(starting_player)
            else:
                starting_player = starting_player

            root_node = node.Node(parent=None,
                                  state=gamestate.GameState(player=starting_player, dimensions=dimensions))
            root_node.state.initialize_hexboard()

            batch_player = starting_player

            game_over = False

            while not game_over:
                print("")
                print("")
                print("")
                print("Move")

                batch_node = Run().find_move(root_node, simulations, batch_player, irh)

                next_move = None
                highest_ratio = -float('inf')
                lowest_ratio = float('inf')
                current_player = batch_node.get_state().get_player()

                print("Current player: " + str(current_player))

                for child in batch_node.child_nodes:
                    ratio = float(child.get_wins()) / float(child.get_visits())

                    if current_player == batch_player:
                        if ratio > highest_ratio:
                            highest_ratio = ratio
                            next_move = child
                    else:
                        if ratio < lowest_ratio:
                            lowest_ratio = ratio
                            next_move = child
                if verbose:
                    next_move.state.print_hexboard()

                root_node = next_move
                root_node.state.switch_player(root_node.state.get_player())
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

    def find_move(self, node, simulations, batch_player, irh):
        move_node = node

        for i in range(0, simulations):

            # this searches through tree based on UCT value
            best_node = mcts.MCTS().search(move_node, batch_player)

            # expands the node with children if there are possible states
            mcts.MCTS().expand(best_node)

            # if node was expanded, choose a random child to evaluate
            if len(best_node.get_child_nodes()) > 0:
                best_node = random.choice(best_node.get_child_nodes())

            # simulates winner. Rollout
            # TODO: Add ANN
            simple_board_state = best_node.state.complex_to_simple_hexboard(best_node.state.hexBoard)
            self.ann_rollout(irh, simple_board_state)
            winner = mcts.MCTS().evaluate(best_node)

            # traverses up tree with winner
            mcts.MCTS().backpropogate(best_node, winner, batch_player)

        return move_node

    def ann_rollout(self, irh, simple_board_state):
        irh.load_json("./config/variables.json")
        irh.run_rollout(simple_board_state)

        return 0
        # Case generator:
        # Properly use irh to set up the correct values and paramters
        # Create new function in irh to call
        # Return ANN probability
        # check for legal move and fix index accordingly
        # return winning nodes
        # Do prediction






Run().run(batch=1, starting_player=1, simulations=10, dimensions=3, verbose=False)
