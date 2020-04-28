# Playing Atari Games with DQN

This project was developed as a final project for the Trends of AI course at ULB.

In order to run the project you need:
* Python 3.7
* Install python libraries: Thensorflow, Gym, Numpy.
* Jupyter Notebook

It consists on three scripts:
* *Graphs.ipynb* : it receives a trained-model.pck file containing an array of rewards and it plots the average rewards over time.
* *index.ipynb*: it contains the DQN script that trains the agent.
* *prod_model.ipynb*: it receives a ```trained-model.meta``` file that contains the already trained model meta information and it plays the Atari game using this model.

# References
This project uses the following excellent sources:

- Sudharsan Ravichandiran. (2018). Hands-On Reinforcement Learning with Python.
- Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.
- Ryan Wong. (2018). Reinforcement Learning: An Introduction to the Concepts, Applications and Code.
- Chris Nicholson. (2018). A Beginner’s Guide to Deep Reinforcement Learning.
- [Simoninithomas github tutorial](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb)

Made with love: Niklaus Geisser and Joan Gerard :green_heart:
