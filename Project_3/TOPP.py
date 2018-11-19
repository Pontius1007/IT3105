import anet as ANET
import tensorflow as tf
import node
import gamestate
import random
import numpy as np

class Topp:
    def __init__(self, number_of_games, hex_dimensions, anet_dims, number_of_agents, save_offset, verbose,
                 saved_path="netsaver/TOPP/"):
        self.number_of_games = number_of_games
        self.hex_dimension = hex_dimensions
        self.anet_dims = anet_dims
        self.saved_path = saved_path
        self.number_of_agents = number_of_agents
        self.verbose = verbose

        self.agents = self.load_agents(save_offset)

    def load_agents(self, save_offset):
        agents = []
        for agent in range(self.number_of_agents):
            agents.append(Hex(dims=self.anet_dims, load_path=self.saved_path, save_offset=agent * save_offset))
        return agents

    # Round robin loop over all the different actors and play them up against each other. Save wins for each agent
    def play_tournament(self):
        print("Starting the tournament with", len(self.agents), "players")
        for i in range(len(self.agents) - 1):
            for j in range(len(self.agents) - 1, i, -1):
                start_player = 1
                for x in range(self.number_of_games):

                    if self.verbose:
                        print("Starting game {}. Player {} against player {}. "
                              "Starting player is {}".format(x, self.agents[i].name, self.agents[j].name, start_player))
                    self.play_game(self.agents[i], self.agents[j], start_player)
                    # Circulate starting player
                    start_player = 3 - start_player
        print("The tournament is finished and the score for each agent is: ")
        for agent in self.agents:
            print(agent.name, agent.wins)

    # Play game. A function that plays out the game between two different agents
    def play_game(self, ANET1, ANET2, start_player):
        root_node = node.Node(parent=None,
                              state=gamestate.GameState(player=start_player, dimensions=self.hex_dimension))
        root_node.state.initialize_hexboard()
        if self.verbose:
            root_node.state.print_hexboard()
        agents = [ANET1, ANET2]
        current_player = start_player
        while not root_node.state.game_over():
            best_move_node = agents[current_player - 1].find_move(root_node)
            if self.verbose:
                print("Move found by {}".format(agents[current_player-1].name))
            root_node = best_move_node
            current_player = 3 - current_player
            if self.verbose:
                root_node.state.print_hexboard()
        winner = root_node.state.get_winner()
        agents[winner - 1].wins += 1
        if self.verbose:
            print("The game is over and {} won.".format(agents[winner - 1].name))


class Hex:
    def __init__(self, dims, load_path, save_offset):
        self.ANET = None
        self.name = "agent "
        self.wins = 0
        self.save_offset = save_offset

        self.load_ANET_agent(dims=dims, load_path=load_path)

    def load_ANET_agent(self, dims, load_path):
        self.ANET = ANET.Gann(
            dims=dims,
            cman=ANET.Caseman([]),
            hidden_activation_function="relu",
            optimizer="adam",
            lower=-0.01,
            upper=0.1,
            lrate=0.01,
            showfreq=None,
            mbs=32,
            vint=None,
            softmax=True,
            cost_function='CE',
            grab_module_index=[],
            grab_type=None
        )
        self.ANET.setupSession()
        session = self.ANET.current_session
        state_vars = []
        for module in self.ANET.modules:
            vars = [module.getvar('wgt'), module.getvar('bias')]
            state_vars = state_vars + vars
        self.ANET.state_saver = tf.train.Saver(state_vars)
        self.ANET.state_saver.restore(sess=session, save_path=(load_path + "-" + str(self.save_offset)))
        self.name += str(self.save_offset)

    # Find the best move using that agent. Should just be copy paste from the ANET_predict/rollout
    def find_move(self, node):
        node_indexes = node.state.next_node_states()[1]
        simple_board_state = node.state.complex_to_simple_hexboard(node.state.hexBoard)
        ANET_pred = self.ANET.do_prediction(simple_board_state)
        best_move = []
        for i, value in enumerate(ANET_pred[0]):
            if node_indexes[i] == 1:
                best_move.append(value)
        max_value = max(best_move)
        max_index = best_move.index(max_value)
        child_nodes = node.get_child_nodes()

        # next best move
        if len(best_move) > 2:
            next_max_value = sorted(best_move)[-2]
            next_max_index = best_move.index(next_max_value)
            random_number = random.randint(0, 100)
            if random_number < 10:
                return child_nodes[next_max_index]

        return child_nodes[max_index]


def main():
    # We create an ANET agent with the same dimensions before loading from file.
    # Save offset is the offset between saves so loading is handled correctly.
    topp = Topp(number_of_games=100, hex_dimensions=3, anet_dims=[20, 32, 16, 9], number_of_agents=5, save_offset=50,
                verbose=False)
    topp.play_tournament()


if __name__ == '__main__':
    main()
