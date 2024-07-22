import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from .utils import SimpleEnv, Definitions

class NN(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, action_n)
        self.relu = nn.ReLU()

    def forward(self, state):
        hidden = self.linear1(state)
        hidden = self.relu(hidden)
        hidden = self.linear2(hidden)
        hidden = self.relu(hidden)
        qvalues = self.linear3(hidden)
        return qvalues

class LSTMNN(nn.Module):
    def __init__(self, state_dim, action_n, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, action_n)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, state):
        batch_size = state.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(state.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(state.device)
        out, _ = self.lstm(state, (h0, c0))
        out = out[:, -1, :]
        qvalues = self.linear(out)
        return qvalues

class TransformerNN(nn.Module):
    def __init__(self, state_dim, action_n, hidden_dim=32, num_heads=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, action_n)
        self.hidden_dim = hidden_dim

    def forward(self, state):
        state = self.embedding(state)
        state = state.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim) -> (seq_len, batch_size, hidden_dim)
        transformer_out = self.transformer(state)
        out = transformer_out[-1, :, :]
        qvalues = self.linear(out)
        return qvalues

class DQN():
    def __init__(self, action_n, model, batch_size, gamma, lr, trajectory_n, optimizer_name, loss_name, memory_size=10000, scheduler_name=None, scheduler_kwargs=None):
        self.action_n = action_n
        self.model = model
        self.epsilon = 1
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decrease = 1 / trajectory_n
        self.memory = deque(maxlen=memory_size)
        self.optimizer = Definitions.get_optimizer(optimizer_name, self.model.parameters(), lr)
        self.loss_fn = Definitions.get_loss_function(loss_name)

        if scheduler_name:
            self.scheduler = Definitions.get_scheduler(scheduler_name, self.optimizer, **scheduler_kwargs)
        else:
            self.scheduler = None

    def get_action(self, state):
        qvalues = self.model(torch.FloatTensor(state)).detach().numpy()
        prob = np.ones(self.action_n) * self.epsilon / self.action_n
        argmax_action = np.argmax(qvalues)
        prob[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return action

    def get_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in range(len(batch)):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            dones.append(batch[i][3])
            next_states.append(batch[i][4])
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        return states, actions, rewards, dones, next_states

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            states, actions, rewards, dones, next_states = self.get_batch()

            qvalues = self.model(states)
            next_qvalues = self.model(next_states)

            targets = qvalues.clone().detach()  # Detach targets from the computation graph
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_qvalues[i]).item()

            loss = self.loss_fn(qvalues, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.epsilon = max(0, self.epsilon - self.epsilon_decrease)

class DQNTrainer:
    def __init__(self, env, model, batch_size=32, gamma=0.99, lr=0.001, trajectory_n=1000, memory_size=10000, optimizer_name='adam', loss_name='mse', scheduler_name=None, scheduler_kwargs=None):
        self.env = env
        self.dqn = DQN(
            action_n=env.action_n,
            model=model,
            batch_size=batch_size,
            gamma=gamma,
            lr=lr,
            trajectory_n=trajectory_n,
            optimizer_name=optimizer_name,
            loss_name=loss_name,
            memory_size=memory_size,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs
        )

    def train(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.dqn.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.dqn.fit(state, action, reward, done, next_state)
                state = next_state
                episode_reward += reward
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {self.dqn.epsilon}")
        return rewards

if __name__ == "__main__":
    env = SimpleEnv()
    model = TransformerNN(state_dim=env.state_dim, action_n=env.action_n)
    trainer = DQNTrainer(env, model)
    trainer.train(num_episodes=100)