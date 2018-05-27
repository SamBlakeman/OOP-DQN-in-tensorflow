import numpy as np
from Minibatch import MiniBatch

class Memory(object):

    def __init__(self):

        self.capacity = 500000
        self.prev_phis = []
        self.phis = []
        self.actions = []
        self.rewards = []
        self.bTrial_over = []

        return

    def RecordExperience(self, prev_phi, phi, action, reward, bTrial_over):

        self.prev_phis.append(prev_phi)
        self.phis.append(phi)
        self.rewards.append(reward)
        self.bTrial_over.append(bTrial_over)

        self.actions.append(action)

        if(self.rewards.__len__() > self.capacity):
            del self.prev_phis[0]
            del self.phis[0]
            del self.actions[0]
            del self.rewards[0]
            del self.bTrial_over[0]

        return


    def GetMinibatch(self, minibatch_size):

        minibatch = MiniBatch()
        experience_indices = np.random.randint(0, self.rewards.__len__(), minibatch_size)

        prev_phis = []
        actions = []
        rewards = []
        phis = []
        bTrial_over = []

        for i in experience_indices:

            prev_phis.append(self.prev_phis[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            phis.append(self.phis[i])
            bTrial_over.append(self.bTrial_over[i])

        minibatch.prev_phis = prev_phis[0]
        minibatch.actions = np.array(actions, dtype=int)
        minibatch.rewards = np.array(rewards, dtype=float)
        minibatch.phis = phis[0]
        minibatch.bTrial_over = bTrial_over

        for i in range(rewards.__len__()-1):
            minibatch.prev_phis = np.concatenate((minibatch.prev_phis, prev_phis[i+1]), axis=0)
            minibatch.phis = np.concatenate((minibatch.phis, phis[i+1]), axis=0)

        return minibatch
