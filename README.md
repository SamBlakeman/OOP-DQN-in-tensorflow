DQN

OOP tensorflow implementation of the model used in the below paper from Google DeepMind:

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Call the update method every frame from whichever game you are simulating. Provide the update method with a numpy array of screen pixel values, a boolean indicating whether the game trial has finished and a reward value such as -1, 0 or 1.

This implementation has been adapted from some other work of mine in order to be stand alone so please let me know if there are any issues. Further documentation coming soon!
