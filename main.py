import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import pickle
import random
from datetime import datetime

from dqn import DQN

"""
Placeholders
A placeholder is a way to input information into a TensorFlow computation graph. Think of placeholders as the input nodes 
through which information enters Tensor‐ Flow. The key function used to create placeholders is tf.placeholder (Example 3-4).
Example 3-4. Create a TensorFlow placeholder
>>> tf.placeholder(tf.float32, shape=(2,2))
<tf.Tensor 'Placeholder:0' shape=(2, 2) dtype=float32>
We will use placeholders to feed datapoints x and labels y to our regression and classi‐ fication algorithms.
"""


def preprocess_observation(obs):
    color = np.array([210, 164, 74]).mean()
    # Crop and resize the image
    img = obs[1:176:2, ::2]

    # Convert the image to greyscale
    img = img.mean(axis=2)

    # Improve image contrast
    img[img == color] = 0

    # Next we normalize the image from -1 to +1
    img = (img - 128) / 128 - 1

    return img.reshape(88, 80, 1)


def q_network(X, name_scope):
    # Initialize layers
    initializer = tf.contrib.layers.variance_scaling_initializer()

    with tf.variable_scope(name_scope) as scope:
        # initialize the convolutional layers
        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME',
                         weights_initializer=initializer)
        tf.summary.histogram('layer_1', layer_1)

        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME',
                         weights_initializer=initializer)
        tf.summary.histogram('layer_2', layer_2)

        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME',
                         weights_initializer=initializer)
        tf.summary.histogram('layer_3', layer_3)

        # Flatten the result of layer_3 before feeding to the fully connected layer
        flat = flatten(layer_3)

        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        tf.summary.histogram('fc', fc)

        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output', output)

        # Vars will store the parameters of the network such as weights
        vars = {v.name[len(scope.name):]: v for v in
                tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        return vars, output


def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action


"""
Next, we define a function called sample_memories for sampling experiences from the memory. Batch size is the number of experience sampled from the memory.
"""


def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0], mem[:, 1], mem[:, 2], mem[:, 3], mem[:, 4]


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    n_outputs = env.action_space.n

    tf.reset_default_graph()

    """
    Next we define a function called epsilon_greedy for performing epsilon greedy policy. In epsilon greedy policy we 
    either select the best action with probability 1 - epsilon or a random action with probability epsilon.

    We use decaying epsilon greedy policy where value of epsilon will be decaying over time as we don't want to explore forever. 
    So over time our policy will be exploiting only good actions.
    """

    epsilon = 0.5
    eps_min = 0.05
    eps_max = 1.0
    eps_decay_steps = 500000

    """
    Now, we initialize our experience replay buffer of length 20000 which holds the experience.
    We store all the agent's experience i.e (state, action, rewards) in the experience replay buffer and we sample from 
    this minibatch of experience for training the network.
    """

    buffer_len = 20000
    exp_buffer = deque(maxlen=buffer_len)

    """
    Now we define our network hyperparameters,
    """

    # num_episodes = 800
    num_episodes = 3000
    batch_size = 48
    input_shape = (None, 88, 80, 1)
    learning_rate = 0.001
    X_shape = (None, 88, 80, 1)
    discount_factor = 0.97

    global_step = 0
    copy_steps = 100
    steps_train = 4
    start_steps = 2000

    logdir = 'logs'
    # tf.reset_default_graph()

    # Now we define the placeholder for our input i.e game state
    X = tf.placeholder(tf.float32, shape=X_shape)

    # we define a boolean called in_training_model to toggle the training
    in_training_mode = tf.placeholder(tf.bool)

    """
    Now let us build our primary and target Q network
    """

    # # we build our Q network, which takes the input X and generates Q values for all the actions in the state
    # mainQ, mainQ_outputs = q_network(X,
    #                                  'mainQ')  # Main Q has the variables, mainQ_outputs has the output of the network
    #
    # # similarly we build our target Q network
    # targetQ, targetQ_outputs = q_network(X, 'targetQ')

    with tf.Session() as sess:
        X_action = tf.placeholder(tf.int32, shape=(None,))

        mainDQN = DQN(sess, X, X_action, n_outputs, name="main")
        targetDQN = DQN(sess, X, X_action, n_outputs, name="target")

        # define the placeholder for our action values
        # X_action = tf.placeholder(tf.int32, shape=(None,))
        Q_action = tf.reduce_sum(targetDQN.output * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True)

        # Q_action = targetDQN.get_Q_action()

        """
        Copy the primary Q network parameters to the target Q network
        """
        copy_op = [tf.assign(main_name, targetDQN.vars[var_name]) for var_name, main_name in mainDQN.vars.items()]
        copy_target_to_main = tf.group(*copy_op)

        """
        Compute and optimize loss using gradient descent optimizer
        """

        # define a placeholder for our output i.e action
        y = tf.placeholder(tf.float32, shape=(None, 1))

        # now we calculate the loss which is the difference between actual value and predicted value
        loss = tf.reduce_mean(tf.square(y - Q_action))

        # we use adam optimizer for minimizing the loss
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        loss_summary = tf.summary.scalar('LOSS', loss)
        merge_summary = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        """
        Now we start the tensorflow session and run the model,
        """
        init.run()

        # for each episode
        rewards = []
        for i in range(num_episodes):
            print("episode: ", i)
            done = False
            obs = env.reset()
            epoch = 0
            episodic_reward = 0
            actions_counter = Counter()
            episodic_loss = []
            f = False

            # while the state is not the terminal state
            while not done:

                # env.render()

                # get the preprocessed game screen
                obs = preprocess_observation(obs)

                # feed the game screen and get the Q values for each action
                actions = mainDQN.get_actions(obs)

                # get the action
                action = np.argmax(actions, axis=-1)
                # if (i + 1) % 5 == 0 and not f:
                #     print("Best action: ", action)
                actions_counter[str(action)] += 1

                # select the action using epsilon greedy policy
                action = epsilon_greedy(action, global_step)

                # now perform the action and move to the next state, next_obs, receive reward
                next_obs, reward, done, _ = env.step(action)

                # Store this transistion as an experience in the replay buffer
                exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])

                # After certain steps, we train our Q network with samples from the experience replay buffer
                if global_step % steps_train == 0 and global_step > start_steps:
                    # sample experience
                    o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)

                    # states
                    o_obs = [x for x in o_obs]

                    # next states
                    o_next_obs = [x for x in o_next_obs]

                    # next actions

                    next_act = mainDQN.get_actions(o_next_obs, False, True)

                    # reward
                    y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1 - o_done)

                    # merge all summaries and write to the file
                    mrg_summary = merge_summary.eval(
                        feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act,
                                   in_training_mode: False})
                    file_writer.add_summary(mrg_summary, global_step)

                    # now we train the network and calculate loss
                    train_loss, _ = sess.run([loss, training_op],
                                             feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act,
                                                        in_training_mode: True})
                    episodic_loss.append(train_loss)
                    if (i + 1) % 50 == 0:
                        print("episodic loss")
                        print(episodic_loss)

                # after some interval we copy our main Q network weights to target Q network
                if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                    copy_target_to_main.run()

                obs = next_obs
                epoch += 1
                global_step += 1

                episodic_reward += reward
            print('Epoch', epoch, 'Reward', episodic_reward, )
            rewards.append(episodic_reward)
            if (i + 1) % 100 == 0:
                mainDQN.save(episode=i)
                targetDQN.save(episode=i)
                pickle.dump(rewards, open('./rewards/trained-model-'+str(i)+'.pck', 'wb+'))
        print("episode {0}, global_step {1}".format(i,global_step))
