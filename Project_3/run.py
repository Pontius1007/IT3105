import random
import node
import gamestate
import mcts
import cProfile
import tflowtools as TFT
import anet as ANET
from copy import deepcopy


class Run:
    def __init__(self, batch, starting_player, simulations, dimensions, number_of_saved_agents, verbose=False,
                 saved_folder="netsaver/saved_anet_states/"):
        self.batch = batch
        self.starting_player = starting_player
        self.simulations = simulations
        self.hex_dimensions = dimensions
        self.verbose = verbose
        self.replay_buffer = []
        self.number_of_saved_agents = number_of_saved_agents
        self.saved_folder = saved_folder

        # ANET parameter
        self.ANET_CM = ANET.Caseman(self.replay_buffer)
        self.ANET_input_dim = (self.hex_dimensions * self.hex_dimensions * 2) + 2
        self.ANET_output_dim = self.hex_dimensions * self.hex_dimensions
        self.ANET = ANET.Gann(dims=[self.ANET_input_dim, 12, 12, self.ANET_output_dim],
                              hidden_activation_function="relu",
                              optimizer="adam", lower=-0.1,
                              upper=0.1, cman=self.ANET_CM, lrate=0.01,
                              showfreq=None, mbs=10, vint=None, softmax=True,
                              cost_function='MSE', grab_module_index=[],
                              grab_type=None)

    def run(self):

        total_wins_player1 = 0
        total_wins_player2 = 0
        mix = False
        self.ANET.setupSession()
        self.ANET.error_history = []
        self.ANET.validation_history = []
        self.save_interal = self.batch / (self.number_of_saved_agents - 1)

        if self.starting_player == 'mix':
            mix = True

        # Save ANET state before training
        self.ANET.save_session_params(self.saved_folder, self.ANET.current_session, 0)
        print("Saved game after 0 episodes")
        for i in range(0, self.batch):
            if mix:
                self.starting_player = random.randint(1, 2)
                print("Starting player is: ", self.starting_player)

            # TODO: Move outside and clear in intervals
            self.replay_buffer = []

            root_node = node.Node(parent=None,
                                  state=gamestate.GameState(player=self.starting_player,
                                                            dimensions=self.hex_dimensions))
            root_node.state.initialize_hexboard()

            batch_player = self.starting_player

            game_over = False

            while not game_over:
                print("")
                print("")
                print("")
                print("Move")
                indexes = root_node.state.next_node_states()[1]
                batch_node = self.find_move(root_node, self.simulations, batch_player, indexes)

                next_move = None
                highest_visits = -float('inf')

                visit_counts = []

                for child in batch_node.child_nodes:
                    visits = child.get_visits()
                    visit_counts.append(child.get_visits())

                    # Printing
                    children = [x for x in child.state.hexBoard]
                    new_children = []
                    for thing in children:
                        for another in thing:
                            new_children.append(another)

                    print("Child " +  " had ratio " + str(
                        visits) + " with wins/visits " + str(
                        child.get_wins()) + " / " + str(child.get_visits()))

                    if visits > highest_visits:
                        highest_visits = visits
                        next_move = child
                if self.verbose:
                    next_move.state.print_hexboard()

                # add case
                case = []
                visit_distribution = []
                case.append(batch_node.state.complex_to_simple_hexboard(batch_node.state.hexBoard))
                for index in indexes:
                    if index == 1:
                        visit_distribution.append(visit_counts.pop(0) - 1)
                    else:
                        visit_distribution.append(0)

                one_hot_visit_distribution = [0] * len(visit_distribution)
                normalized_visit_distribution = []

                # generates one_hot list
                max_value = max(visit_distribution)
                max_index = visit_distribution.index(max_value)
                one_hot_visit_distribution[max_index] = 1

                # generates normalized list
                for value in visit_distribution:
                    normalized_visit_distribution.append(value / max_value)

                case.append(normalized_visit_distribution)
                self.replay_buffer.append(case)

                root_node = next_move
                # Already switching when generating kids
                # root_node.state.switch_player(root_node.state.get_player())
                root_node.state.print_hexboard()

                if root_node.get_state().game_over():
                    winner = 3 - root_node.get_state().get_player()
                    if self.verbose:
                        print("Player " + str(winner) + " wins.")
                        print("")
                    if winner == 1:
                        total_wins_player1 += 1
                    if winner == 2:
                        total_wins_player2 += 1
                    game_over = True

            # do training
            inputs = [c[0] for c in self.replay_buffer]
            targets = [c[1] for c in self.replay_buffer]
            feeder = {self.ANET.input: inputs, self.ANET.target: targets}
            gvars = self.ANET.error

            _, grabvals, _ = self.ANET.run_one_step([self.ANET.trainer], gvars, self.ANET.probes,
                                                    session=self.ANET.current_session, feed_dict=feeder,
                                                    show_interval=0)

            self.ANET.error_history.append((i, grabvals))

            # Save ANET-params
            if self.number_of_saved_agents:
                if (i + 1) % self.save_interal == 0:
                    self.ANET.save_session_params(self.saved_folder, self.ANET.current_session, (i + 1))
                    print("Saved game after ", i + 1, " episodes")

            self.ANET_CM.cases = self.replay_buffer
        print("")
        print("Player 1" + " won " + str(total_wins_player1) + " times out of " + str(
            self.batch) + " batches." + " (" + str(
            100 * total_wins_player1 / self.batch) + "%)")
        print("Player 2" + " won " + str(total_wins_player2) + " times out of " + str(
            self.batch) + " batches." + " (" + str(
            100 * total_wins_player2 / self.batch) + "%)")
        TFT.plot_training_history(self.ANET.error_history, self.ANET.validation_history, xtitle="Game",
                                  ytitle="Error",
                                  title="", fig=True)
        self.ANET.close_current_session(view=False)

    def find_move(self, node, simulations, batch_player, indexes):
        move_node = node
        print(move_node.state.get_player())

        for i in range(0, simulations):

            # move_node.state.player = 3 - move_node.state.player

            # this searches through tree based on UCT value
            best_node = mcts.MCTS().search(move_node, batch_player)

            # expands the node with children if there are possible states
            mcts.MCTS().expand(best_node)

            # if node was expanded, choose a random child to evaluate
            if len(best_node.get_child_nodes()) > 0:
                best_node = random.choice(best_node.get_child_nodes())

            # simulates winner. Rollout
            # TODO: Add ANN

            winner = mcts.MCTS().ANET_evaluate(ANET=self.ANET, node=best_node)

            # traverses up tree with winner
            mcts.MCTS().backpropogate(best_node, winner, batch_player)

        return move_node


Run(batch=1, starting_player=1, simulations=30, dimensions=2, verbose=False, number_of_saved_agents=5).run()
