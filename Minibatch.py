
class MiniBatch(object):

    def __init__(self):
        self.prev_phis = []
        self.actions = []
        self.rewards = []
        self.phis = []
        self.bTrial_over = []