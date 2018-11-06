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

    # returns next possible nodes of a node
    def next_node_states(self):
        children = []
        for i in range(1, min(self.numberOfPieces, self.maxRemovePieces) + 1):
            children.append(GameState(player=3 - self.player, numberofpieces=self.numberOfPieces - i,
                                      maxremove=self.maxRemovePieces))
        return children
