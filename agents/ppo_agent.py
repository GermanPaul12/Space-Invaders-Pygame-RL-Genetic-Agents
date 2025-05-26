# agents/ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .agent import Agent
from .dqn_agent import preprocess_observation # Reuse preprocessing

# Re-use ActorCriticNetwork from A2C if the structure is similar
# For PPO, the actor usually outputs logits for Categorical distribution.
class PPOActorCriticNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, h=84, w=84):
        super(PPOActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_h = conv_output_size(conv_output_size(conv_output_size(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv_output_size(conv_output_size(conv_output_size(w, 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64
        
        self.fc_shared = nn.Linear(flattened_size, 512)

        self.actor_fc = nn.Linear(512, num_actions) # Outputs logits
        self.critic_fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x_shared = F.relu(self.fc_shared(x))
        
        action_logits = self.actor_fc(x_shared) # Logits for Categorical
        state_value = self.critic_fc(x_shared)
        
        return action_logits, state_value


class PPOAgent(Agent):
    def __init__(self, action_size, observation_shape, # (C, H, W)
                 lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
                 ppo_clip=0.2, ppo_epochs=4, mini_batch_size=32,
                 entropy_coef=0.01, value_loss_coef=0.5,
                 trajectory_n_steps=128): # Steps to collect before update
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.input_channels = observation_shape[0] if observation_shape else 1
        self.network = PPOActorCriticNetwork(self.input_channels, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.trajectory_n_steps = trajectory_n_steps

        # Trajectory buffer
        self.buffer = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'dones': [], 'values': []
        }
        self.current_buffer_size = 0

    def _clear_buffer(self):
        for k in self.buffer:
            self.buffer[k] = []
        self.current_buffer_size = 0

    def choose_action(self, observation): # observation is raw
        state_p = preprocess_observation(observation)
        state_tensor = torch.from_numpy(state_p).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, state_value = self.network(state_tensor)
        
        m = Categorical(logits=action_logits)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        # Store for trajectory (will be added to buffer later)
        self._temp_action_data = {
            'state': state_p, # Store preprocessed state
            'action': action.item(),
            'log_prob': log_prob.item(),
            'value': state_value.item()
        }
        return action.item()

    def store_transition_outcome(self, reward, done, next_observation):
        # Add previous step's data along with current outcome
        self.buffer['states'].append(self._temp_action_data['state'])
        self.buffer['actions'].append(self._temp_action_data['action'])
        self.buffer['log_probs'].append(self._temp_action_data['log_prob'])
        self.buffer['values'].append(self._temp_action_data['value'])
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.current_buffer_size += 1

        # If trajectory is full, or game is done, trigger learning
        if self.current_buffer_size >= self.trajectory_n_steps or done:
            # Need the value of the 'next_observation' to calculate GAE for the last step
            last_value = 0
            if not done and next_observation is not None: # Bootstrap if not terminal
                next_state_p = preprocess_observation(next_observation)
                next_state_tensor = torch.from_numpy(next_state_p).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, last_val_tensor = self.network(next_state_tensor)
                last_value = last_val_tensor.item()
            
            loss = self.learn(last_value)
            self._clear_buffer() # Ready for new trajectory
            return loss
        return None


    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        # values include V(s_0)...V(s_{T-1}). last_value is V(s_T)
        # For loop from T-1 down to 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * (values[t+1] if t+1 < len(values) else last_value) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:len(rewards)] # Gt = At + V(st)
        return advantages, returns

    def learn(self, last_value):
        # Prepare data from buffer
        states = torch.from_numpy(np.array(self.buffer['states'])).float().to(self.device)
        actions = torch.tensor(self.buffer['actions'], device=self.device).long()
        old_log_probs = torch.tensor(self.buffer['log_probs'], device=self.device).float()
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])
        values = np.array(self.buffer['values'])

        advantages, returns = self._compute_gae(rewards, values, dones, last_value)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)

        # Normalize advantages (optional but often good)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_val = 0
        
        # PPO update epochs
        for _ in range(self.ppo_epochs):
            # Create minibatches
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                mb_indices = indices[start_idx:end_idx]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Get current policy predictions for these states
                new_action_logits, new_values = self.network(mb_states)
                m = Categorical(logits=new_action_logits)
                new_log_probs = m.log_prob(mb_actions)
                entropy = m.entropy().mean()
                new_values = new_values.squeeze()


                # Policy (Actor) loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value (Critic) loss
                critic_loss = F.mse_loss(new_values, mb_returns)
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                total_loss_val += loss.item() # Accumulate for logging

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Clip gradients
                self.optimizer.step()
        
        return total_loss_val / (self.ppo_epochs * (num_samples // self.mini_batch_size))


    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"PPO model saved to {path}")

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.train()
        print(f"PPO model loaded from {path}")