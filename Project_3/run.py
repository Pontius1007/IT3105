import random
import node
import gamestate
import mcts


class Run:
    def run(self, batch, starting_player, simulations, numberofpieces, maxremove, verbose=False):

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

            root_node = node.Node(parent=None,
                                  state=gamestate.GameState(player=starting_player, numberofpieces=numberofpieces,
                                                            maxremove=maxremove))
            batch_player = starting_player

            game_over = False

            while not game_over:

                batch_node = Run().find_move(root_node, simulations, batch_player)

                next_move = None
                highest_ratio = -float('inf')
                lowest_ratio = float('inf')
                current_player = batch_node.get_state().get_player()

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
                    print("Player " + str(current_player) + " selected " + str(
                        batch_node.state.get_number_of_pieces() - next_move.state.get_number_of_pieces()) + " pieces."
                          + " There are " + str(next_move.state.get_number_of_pieces()) + " pieces left.")

                root_node = node.Node(state=gamestate.GameState(player=(3 - current_player),
                                                                numberofpieces=next_move.get_state().get_number_of_pieces(),
                                                                maxremove=maxremove))

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

        for i in range(0, simulations):

            # this searches through tree based on UCT value
            best_node = mcts.MCTS().search(move_node, batch_player)

            # expands the node with children if there are possible states
            mcts.MCTS().expand(best_node)

            # if node was expanded, choose a random child to evaluate
            if len(best_node.get_child_nodes()) > 0:
                best_node = random.choice(best_node.get_child_nodes())

            # simulates winner
            winner = mcts.MCTS().evaluate(best_node)

            # traverses up tree with winner
            mcts.MCTS().backpropogate(best_node, winner, batch_player)

        return move_node


Run().run(batch=1, starting_player=1, simulations=5000, numberofpieces=14, maxremove=3, verbose=True)
