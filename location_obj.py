class location:
    def __init__(self, name, state, state_short):
        self.name = name
        self.state = state
        self.state_short = state_short

    def __str__(self):
        return "Location Name: {}, State: {}.".format(self.name, self.state)