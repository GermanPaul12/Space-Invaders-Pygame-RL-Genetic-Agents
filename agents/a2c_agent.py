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
        # Shared CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_h = conv_output_size(conv_output_size(conv_output_size(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv_output_size(conv_output_size(conv_output_size(w, 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64
        
        self.fc_shared = nn.Linear(flattened_size, 256) # Smaller shared layer

        # Actor head
        self.actor_fc = nn.Linear(256, num_actions)
        # Critic head
        self.critic_fc = nn.Linear(256, 1) # Outputs a single value (state value)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        
        x_shared = F.relu(self.fc_shared(x))
        
        action_probs = F.softmax(self.actor_fc(x_shared), dim=-1)
        state_value = self.critic_fc(x_shared)
        
        return action_probs, state_value


class A2CAgent(Agent):
    def __init__(self, action_size, observation_shape, # (C, H, W)
                 lr=7e-4, gamma=0.99, entropy_coef=0.01):
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C Agent using device: {self.device}")

        self.input_channels = observation_shape[0] if observation_shape else 1
        self.network = ActorCriticNetwork(self.input_channels, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Temporary storage for one step (or a few steps for n-step A2C)
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = [] # For A2C, often updated per step or small batch

    def choose_action(self, observation): # observation is raw
        state = preprocess_observation(observation)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)
        
        m = Categorical(action_probs)
        action = m.sample()
        
        # Store for learning
        self.log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self, next_observation=None): # Call this after each step (or a few steps)
        # A2C updates can happen per step or after a small batch of steps (n-step A2C)
        # For simplicity, this is a basic 1-step A2C, expecting learn to be called after each env step.
        # If called after N steps, rewards/log_probs/state_values would be lists of N items.
        # 'next_observation' is used to bootstrap if not done.

        if not self.log_probs: # Nothing to learn yet
            return None

        # Calculate returns (Gt)
        # For 1-step A2C: R_t = r_t + gamma * V(s_{t+1}) (if not done)
        #                   R_t = r_t (if done)
        
        returns = []
        # If the last step was 'done', the return is just the final reward.
        # Otherwise, bootstrap from V(s_next).
        R = 0
        if not self.dones[-1] and next_observation is not None : # Bootstrap if not terminal
            next_state_p = preprocess_observation(next_observation)
            next_state_tensor = torch.from_numpy(next_state_p).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_val = self.network(next_state_tensor)
            R = next_val.item() # This is V(s_{t+N})

        # Discounted sum of rewards + V(s_next)
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d: # If this step was terminal, reset R (no future rewards)
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R) # Prepend to keep order

        returns = torch.tensor(returns, device=self.device).float()
        log_probs_t = torch.cat(self.log_probs) # Assuming log_probs were (1,) tensors
        state_values_t = torch.cat(self.state_values).squeeze() # Make it (N,)

        # Calculate advantage A_t = G_t - V(s_t)
        advantages = returns - state_values_t

        # Actor loss (policy gradient)
        actor_loss = -(log_probs_t * advantages.detach()).mean() # Detach advantages for actor loss

        # Critic loss (MSE of V(s_t) and G_t)
        critic_loss = F.mse_loss(state_values_t, returns)

        # Entropy bonus (for exploration)
        # For this, we need action_probs again from the stored states.
        # This is inefficient if not re-calculating.
        # A better way is to store probs or re-evaluate states from a small buffer.
        # For simplicity, let's ignore entropy for now, or assume it's small.
        # Or: if self.log_probs contains log_prob from Categorical(action_probs),
        # entropy = -(action_probs * log_action_probs).sum(-1).mean()
        # For now, omitting precise entropy calculation for brevity.
        entropy_loss = 0 # Placeholder

        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear episode/batch buffers
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        
        return total_loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"A2C model saved to {path}")

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.train() # Ensure network is in train mode
        print(f"A2C model loaded from {path}")