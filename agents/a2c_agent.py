# agents/a2c_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .agent import Agent
from .dqn_agent import preprocess_observation # Reuse preprocessing

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, h=84, w=84):
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_h = conv_output_size(conv_output_size(conv_output_size(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv_output_size(conv_output_size(conv_output_size(w, 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64
        
        self.fc_shared = nn.Linear(flattened_size, 256)
        self.actor_fc = nn.Linear(256, num_actions)
        self.critic_fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x_shared = F.relu(self.fc_shared(x))
        action_logits = self.actor_fc(x_shared)
        state_value = self.critic_fc(x_shared)
        return action_logits, state_value


class A2CAgent(Agent):
    def __init__(self, action_size, observation_shape,
                 lr=7e-4, gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5):
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C Agent using device: {self.device}")

        self.input_channels = observation_shape[0]
        self.processed_h = observation_shape[1]
        self.processed_w = observation_shape[2]

        self.network = ActorCriticNetwork(
            self.input_channels, action_size, h=self.processed_h, w=self.processed_w
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.action_logits_buffer = [] # To store logits for entropy calculation

    def choose_action(self, raw_observation):
        state_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        
        # REMOVED with torch.no_grad(): here
        action_logits, state_value = self.network(state_tensor) 
        
        m = Categorical(logits=action_logits)
        action = m.sample()
        
        self.log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        self.action_logits_buffer.append(action_logits) # Store logits
        
        return action.item()

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self, raw_next_observation=None): 
        if not self.log_probs:
            return None

        R = 0
        if not self.dones[-1] and raw_next_observation is not None:
            next_state_p_np = preprocess_observation(raw_next_observation, new_size=(self.processed_h, self.processed_w))
            next_state_tensor = torch.from_numpy(next_state_p_np).float().unsqueeze(0).to(self.device)
            with torch.no_grad(): # V(s_next) is a target, so no grad needed through it
                _, R_tensor = self.network(next_state_tensor)
            R = R_tensor.item()

        policy_returns = []
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d: R = 0
            R = r + self.gamma * R
            policy_returns.insert(0, R)
        
        policy_returns_t = torch.tensor(policy_returns, device=self.device, dtype=torch.float32)
        
        log_probs_t = torch.cat(self.log_probs) 
        state_values_t = torch.cat(self.state_values).squeeze(-1) # Squeeze the last dim (N,1) -> (N,)
        action_logits_t = torch.cat(self.action_logits_buffer)


        # Ensure consistent 1D tensors if N=1 (batch size of 1 step)
        if state_values_t.ndim == 0: state_values_t = state_values_t.unsqueeze(0)
        if policy_returns_t.ndim == 0: policy_returns_t = policy_returns_t.unsqueeze(0)
        if log_probs_t.ndim == 0: log_probs_t = log_probs_t.unsqueeze(0)
        # action_logits_t will be (N, num_actions), which is fine for Categorical

        advantages = policy_returns_t - state_values_t
        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values_t, policy_returns_t)

        # Entropy calculation
        m_entropy = Categorical(logits=action_logits_t)
        entropy_term = m_entropy.entropy().mean()

        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_term

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.action_logits_buffer = [] # Clear logits buffer
        
        return total_loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"A2C model saved to {path}")

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.train()
        print(f"A2C model loaded from {path}")