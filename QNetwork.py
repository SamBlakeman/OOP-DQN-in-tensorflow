import random
import numpy as np
from PIL import Image

from QGraph import QGraph
from Memory import Memory
from Minibatch import MiniBatch
from QTargetGraph import QTargetGraph


class QNetwork(object):

    def __init__(self, bTrain):

        # Settings
        self.directory = '/tmp/TrainedQNetwork'
        self.num_actions = 9
        self.im_height = 84
        self.im_width = 84
        self.discount_factor = 0.99
        self.minibatch_size = 32
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon_frames = 1000000
        self.replay_start_size = 50000
        self.policy_start_size = self.replay_start_size
        self.k = 4  # action repeat (frame skipping)
        self.u = 4  # update frequency
        self.m = 4  # number of frames to include in sequence
        self.c = 10000  # number of actions selected before updating the network used to generate the targets

        # Internal Variables
        self.bTrain = bTrain
        self.ki = 0
        self.ui = 0
        self.mi = 0
        self.frame = 0
        self.ci = 0
        self.sequence = []
        self.prev_phi = np.array([])
        self.phi = np.array([])
        self.epsilon_increment = (self.initial_epsilon - self.final_epsilon) / self.epsilon_frames
        self.epsilon = self.initial_epsilon
        self.action = 0
        self.reward = 0
        self.memory = Memory()
        self.minibatch = MiniBatch()
        self.targets = np.zeros(self.minibatch_size)
        self.bTrial_over = False
        self.bStartLearning = False
        self.bStartPolicy = False
        self.ti = 0

        random.seed(0)

        # Construct tensorflow graphs
        self.q_graph = QGraph(self.im_width, self.im_height, self.m, self.num_actions, self.directory)

        if (self.bTrain):
            self.q_graph.SaveGraphAndVariables()
            self.q_graph_targets = QTargetGraph(self.im_width, self.im_height, self.m, self.num_actions, self.directory)

        return

    def Update(self, pxarray, bTrial_over, reward):

        self.frame += 1
        self.ki +=1

        self.bTrial_over = bTrial_over

        if(self.bTrain):

            self.Train(pxarray=pxarray, bTrial_over=bTrial_over, reward=reward)

        else:

            self.Test(pxarray=pxarray, bTrial_over=bTrial_over)

        return

    def Train(self, pxarray, bTrial_over, reward):

        if (self.frame == self.replay_start_size):
            print('Starting Learning...')
            self.bStartLearning = True

        if (self.frame == self.policy_start_size):
            print('Starting Policy...')
            self.bStartPolicy = True

        self.reward += reward

        if (self.ki >= self.k):
            self.PreprocessSequence(pxarray)
            self.mi += 1
            self.ki = 0

            if(self.mi == self.m):
                self.prev_phi = np.copy(self.phi)
                self.reward = 0

            elif(self.mi >= self.m + 1):
                self.ui += 1
                self.StoreExperience()

                if (self.bStartLearning):
                    if(self.ui >= self.u):
                        self.SampleRandomMinibatch()
                        self.GenerateTargets()
                        self.GradientDescentStep()
                        self.ui = 0

                if (not self.bStartPolicy):
                    self.SelectRandomAction()
                else:
                    self.UpdateAction()

                self.prev_phi = np.copy(self.phi)
                self.reward = 0

        if(bTrial_over):
            self.mi = 0
            self.ki = 0
            self.ui = 0

        if (self.bStartPolicy):
            if(self.epsilon > self.final_epsilon):
                self.epsilon -= self.epsilon_increment

        return

    def Test(self, pxarray, bTrial_over):

        if (self.ki >= self.k):
            self.PreprocessSequence(pxarray)
            self.mi += 1
            self.ki = 0

            if (self.mi >= self.m):
                self.SelectAction()

        if (bTrial_over):
            self.mi = 0
            self.ki = 0

        return

    def PreprocessSequence(self, pxarray):

        img = Image.fromarray(pxarray)
        img = img.resize([self.im_width, self.im_height])
        img_grey = img.convert('LA')
        img_grey = np.array(img_grey).astype(np.uint8)
        img = img_grey[:, :, 0]

        self.sequence.append(img)

        if (self.sequence.__len__() > self.m):
            del self.sequence[0]

        self.phi = np.stack(self.sequence, axis=-1)
        self.phi = np.expand_dims(self.phi, axis=0)

        return

    def UpdateAction(self):

        # e-greedy policy
        if (random.random() <= self.epsilon):
            self.SelectRandomAction()
        else:
            self.SelectAction()

        return

    def SelectAction(self):

        action_values = np.array(self.q_graph.GetActionValues(self.phi))
        num = np.argmax(action_values)
        self.action = num

        return

    def SelectRandomAction(self):

        num = random.randint(0, self.num_actions)
        self.action = num

        return

    def StoreExperience(self):

        self.memory.RecordExperience(self.prev_phi, self.phi, self.action, self.reward, self.bTrial_over)

        return

    def SampleRandomMinibatch(self):

        self.minibatch = self.memory.GetMinibatch(self.minibatch_size)

        return

    def GenerateTargets(self):

        self.ci += 1

        if (self.ti == 0):
            self.example_minibatch = self.minibatch

        # refresh target network
        if(self.ci >= self.c):
            print('Loading New Target Graph...')
            self.ci = 0
            self.q_graph.SaveGraphAndVariables()
            self.q_graph_targets = QTargetGraph(self.im_width, self.im_height, self.m, self.num_actions, self.directory)


        self.targets = self.GetTargets(self.minibatch)

        if(self.ti % 2500 == 0):
            print('Example Actions:')
            print(self.example_minibatch.actions)

            print('Example Targets:')
            print(self.GetTargets(self.example_minibatch))

            print('Phi Example Action Values Q:')
            print(self.q_graph.GetActionValues(self.example_minibatch.prev_phis))

        self.ti += 1

        return

    def GetTargets(self, minibatch):

        targets = np.zeros(minibatch.rewards.__len__())

        action_values = np.array(self.q_graph_targets.GetActionValues(minibatch.phis))
        action_values = np.squeeze(action_values)
        max_action_values = np.amax(action_values, axis=1)


        for i in range(targets.__len__()):

            if (not minibatch.bTrial_over[i]):
                targets[i] = minibatch.rewards[i] + (max_action_values[i] * self.discount_factor)
            else:
                targets[i] = minibatch.rewards[i]

        return targets

    def GradientDescentStep(self):

        self.q_graph.GradientDescentStep(self.minibatch.prev_phis, self.minibatch.actions, self.targets)

        return

    def SaveGraphAndVariables(self):

        self.q_graph.SaveGraphAndVariables()

        return

    def LoadGraphAndVariables(self):

        self.q_graph.LoadGraphAndVariables()

        return