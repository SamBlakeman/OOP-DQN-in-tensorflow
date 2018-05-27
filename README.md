Deep Q-Network

OOP tensorflow implementation of the model described in the below paper from Google DeepMind:

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Call the QNetwork constructor in QNetwork.py to initialise the model. The boolean bTrain dictates whether the model is trained from new or loaded from a previous session (if loaded then no more training occurs).

To use the model, call the update method every frame from whichever game you are simulating. Provide the update method with a numpy array of the screen pixel values, a boolean indicating whether the game trial has finished and a reward value such as -1, 0 or 1. The update method will then train the DQN if bTrain == True and return the action number chosen by the agent. 

This implementation has been adapted from some other work of mine in order to be stand alone so please let me know if there are any issues. Further documentation coming soon!

Below is a brief description of the settings variables in QNetwork.py

self.directory = '/tmp/TrainedQNetwork' # Where the tensorflow graph will be saved (or loaded from if bTrain == False)
self.num_actions = 9 # The number of available actions to the agent
self.im_height = 84 # The re-scaled height, in pixel values, of the input image
self.im_width = 84 # The re-scaled width, in pixel values, of the input image
self.discount_factor = 0.99 # The discount factor for future rewards
self.minibatch_size = 32 # The number of examples used for each gradient descent update
self.initial_epsilon = 1.0 # Starting probability for selecting a random action
self.final_epsilon = 0.1 # Final probability for selecting a random action
self.epsilon_frames = 1000000 # The number of frames over which to decay epsilon towards its final value
self.replay_start_size = 50000 # The number of frames before training begins
self.policy_start_size = self.replay_start_size # The number of frames before epsilon decay begins
self.k = 4  # The number of frames between each frame seen by the agent (action repeats in between)
self.u = 4  # THe number of frames seen by the agent before a gradient descent update is performed
self.m = 4  # The number of frames to include in a single state
self.c = 10000  # The number of actions selected before updating the target network
