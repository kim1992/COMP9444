import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy



# TODO: HyperParameters
GAMMA =  0.9 # discount factor
INITIAL_EPSILON = 0.8 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilonw
EPSILON_DECAY_STEPS = 100 # decay period

LEARNING_RATE = 0.002
BATCH_SIZE = 64
HIDDEN_NODES = 20
REPLAY_SIZE = 10000


# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

replay_buffer = []


# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])




def build_network(c_name):
    W1 = tf.Variable(tf.random_normal(shape=[STATE_DIM, HIDDEN_NODES], seed=1), collections=c_name)
    b1 = tf.Variable(tf.random_normal(shape=[1, HIDDEN_NODES], seed=1), collections=c_name)
    fc1 = tf.nn.relu(tf.matmul(state_in, W1) + b1)

    #dueling
    # value
    W2 = tf.Variable(tf.random_normal(shape=[HIDDEN_NODES, 1], seed=1), collections=c_name)
    b2 = tf.Variable(tf.random_normal(shape=[1, 1], seed=1), collections=c_name)
    value = tf.matmul(fc1, W2) + b2

    # advantage
    W2 = tf.Variable(tf.random_normal(shape=[HIDDEN_NODES, ACTION_DIM], seed=1), collections=c_name)
    b2 = tf.Variable(tf.random_normal(shape=[1, ACTION_DIM], seed=1), collections=c_name)
    advantage = tf.matmul(fc1, W2) + b2

    network = tf.nn.relu(value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)))
    return network


with tf.variable_scope("eval_net"):
    c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
    q_values = build_network(c_name)

q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)    


# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.squared_difference(target_in, q_action))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.variable_scope("target_net"):
    c_name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
    q_target = build_network(c_name)


train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)
t_params = tf.get_collection('target_net_params')
e_params = tf.get_collection('eval_net_params')
fix_para_change = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

# update the replay_buffer with provided input in the form:
#     (state, one_hot_action, reward, next_state, done)
def update_replay_buffer(replay_buffer, state, action, reward, next_state, done):

    cache = (state, action, reward, next_state, done)
    # append to buffer
    replay_buffer.append(cache)
    # Ensure replay_buffer doesn't grow larger than REPLAY_SIZE
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)

def get_train_batch(q_values, state_in, replay_buffer):

    # Generate Batch samples for training by sampling the replay buffer
    ''''
    state_batch: Batch of state values
    action_batch: Batch of action values
    target_batch: Target batch for (s,a) pair i.e. one application of the bellman update rule.
    '''
    mini_batch = random.sample(replay_buffer, BATCH_SIZE)

    state_batch = [data[0] for data in mini_batch]
    action_batch = [data[1] for data in mini_batch]
    reward_batch = [data[2] for data in mini_batch]
    next_state_batch = [data[3] for data in mini_batch]


    if total_step < 2000 or total_step % 50 == 0:
        session.run(fix_para_change)
        
    target_batch = []
    
    Q_value_batch = q_values.eval(feed_dict={
        state_in: next_state_batch
    })
    
    #double DQN
    target_next = q_target.eval(feed_dict={
            state_in: next_state_batch
    })
    
    
    for i in range(0, BATCH_SIZE):
        sample_is_done = mini_batch[i][4]
        if sample_is_done:
            target_batch.append(reward_batch[i])
        else:
            # set the target_val to the correct Q value update
            index = np.argmax(Q_value_batch[i])
            target_val = reward_batch[i] + GAMMA * target_next[i][index]
            target_batch.append(target_val)
    return target_batch, state_batch, action_batch

def train_step(replay_buffer, state_in, action_in, \
               target_in, q_values, optimizer, loss):
    
    target_batch, state_batch, action_batch = \
        get_train_batch(q_values, state_in, replay_buffer)

    session.run([loss, optimizer], feed_dict={
        target_in: target_batch,
        state_in: state_batch,
        action_in: action_batch
    })

    

total_step = 0

# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS


    # Move through env according to e-greedy policy
    for step in range(STEP):

        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))


        update_replay_buffer(replay_buffer, state, action, reward, next_state, done)


        # Update
        state = next_state
        
#=========================================================================================================
        if len(replay_buffer) > BATCH_SIZE:
            train_step(replay_buffer, state_in, action_in, target_in, q_values, optimizer, train_loss_summary_op)
            total_step += 1
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
