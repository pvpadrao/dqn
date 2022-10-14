import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# nn.Module = Base class for all neural network modules in PyTorch
class DeepQNetwork(nn.Module):
    # lr = learning rate;
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        # we need the super() call so that the mn.Module class itself is initialised.
        # In Python superclass constructors/initialisers
        # aren't called automatically - they have to be called explicitly,
        # and that is what super() does - it works out what superclass to call.
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # our neural network
        # the * means that we are unpacking a list of elements
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        # defining optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # defining loss function
        self.loss = nn.MSELoss()
        # defining a gpu device, if we have one
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    # we need to take care about forward propagation, but we dont; need to take care of backpropagation. PyTorch will
    # do that for us.
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # no activation on the output layer for now. Only the raw values
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        # gamma = reward discount factor
        # epsilon = exploration vs. exploitation
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        # list of integer representations of available actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        # memory counter to provide us the first element available in memory to store something
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        # the position of the first unocuppied memory. Reason to use modulo (%) operation is that we will wrap up
        # to the beginning once the memory is full. We rewrite new values on top of oldest ones.
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # convert the observation to a pyTorch tensor and send it to the device. Note: our network "lives" in our
            # device. We need to send the varaibles to be computed in our device.
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            # we only need the values from the tensor. That's why we use .item()
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return  # do not learn anything

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        # action_batch does not need to be a tensor. Could be a regular numpy array
        action_batch = self.action_memory[batch]

        # we want the values of the actions that we really took. Not from actions we did not take.
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # if we would be using a target network, we would need to change the line below
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # maximum value of the next state
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        # computes dloss/dx for every parameter x
        loss.backward()
        # All optimizers implement a step() method, that updates the parameters.
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
