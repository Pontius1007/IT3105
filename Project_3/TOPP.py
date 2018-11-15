import anet as ANET
import tensorflow as tf


class Topp:
    def __init__(self, number_of_games, hexsize, anet_dims, number_of_agents, saved_path="netsaver/TOPP/"):
        self.number_of_games = number_of_games
        self.hexsize = hexsize
        self.anet_dims = anet_dims
        self.saved_path = saved_path
        self.number_of_agents = number_of_agents

    def load_agents(self, save_offset):
        agents = []
        for agent in range(self.number_of_agents):
            agents.append(Hex(dims=self.anet_dims, load_path=self.saved_path, save_offset=agent * save_offset))
        return agents

    # Round robin loop over all the different actors and play them up against each other. Save wins for each agent
    def play_tournament(self):
        return 0

    # Play game. A function that plays out the game between two different agents
    def play_game(self):
        return 0


class Hex:
    def __init__(self, dims, load_path, save_offset):
        self.ANET = None
        self.name = "agent"
        self.wins = 0
        self.save_offset = save_offset

        self.load_ANET_agent(dims=dims, load_path=load_path)

    def load_ANET_agent(self, dims, load_path):
        self.ANET = ANET.Gann(
            dims=dims,
            cman=ANET.Caseman([]),
            hidden_activation_function="relu",
            optimizer="adam",
            lower=-0.1,
            upper=0.1,
            lrate=0.01,
            showfreq=None,
            mbs=10,
            vint=None,
            softmax=True,
            cost_function='MSE',
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
    def find_move(self):
        return 0


topp = Topp(number_of_games=10, hexsize=3, anet_dims=[20, 12, 12, 9], number_of_agents=5)
agents = topp.load_agents(50)
for agent in agents:
    print(agent.name)


# create agent
# play tournament
# play game


# hex
# load params
# give move from state
