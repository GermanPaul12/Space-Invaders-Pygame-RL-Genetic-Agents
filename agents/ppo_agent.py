# agents/ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .agent import Agent
from .dqn_agent import preprocess_observation # Reuse preprocessing

# --- Constants for PPO ---
# Consider moving to config or keeping here if PPO specific
REWARD_SCALING_FACTOR = 100.0  # Example: Tune this based on your reward magnitudes

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
        self.actor_fc = nn.Linear(512, num_actions) 
        self.critic_fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x_shared = F.relu(self.fc_shared(x))
        action_logits = self.actor_fc(x_shared) 
        state_value = self.critic_fc(x_shared)
        return action_logits, state_value

class PPOAgent(Agent):
    def __init__(self, action_size, observation_shape, 
                 lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
                 ppo_clip=0.2, ppo_epochs=4, mini_batch_size=32,
                 entropy_coef=0.01, value_loss_coef=0.5,
                 trajectory_n_steps=128):
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.input_channels = observation_shape[0]
        self.processed_h = observation_shape[1]
        self.processed_w = observation_shape[2]

        self.network = PPOActorCriticNetwork(
            self.input_channels, action_size, h=self.processed_h, w=self.processed_w
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5) # eps for Adam stability
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.trajectory_n_steps = trajectory_n_steps

        self.buffer = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'dones': [], 'values': []
        }
        self.current_buffer_size = 0
        self._temp_action_data = {}

    def _clear_buffer(self):
        for k in self.buffer: self.buffer[k] = []
        self.current_buffer_size = 0

    def choose_action(self, raw_observation):
        state_p_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_p_np).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, state_value = self.network(state_tensor)
        
        # --- DEBUG: Check logits from network before Categorical ---
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print(f"PPO CHOOSE_ACTION DEBUG: NaN/Inf in action_logits from network: {action_logits}")
        # --- END DEBUG ---

        try:
            m = Categorical(logits=action_logits)
            action = m.sample()
            log_prob = m.log_prob(action)
        except ValueError as e:
            print(f"PPO CHOOSE_ACTION CRITICAL: ValueError creating Categorical. Logits: {action_logits}. Error: {e}")
            # Fallback to a random action if logits are problematic
            action = torch.randint(0, self.action_size, (1,), device=self.device)
            log_prob = torch.tensor(-1e8, device=self.device) # A very small log_prob

        self._temp_action_data = {
            'state': state_p_np, 
            'action': action.item(),
            'log_prob': log_prob.item(),
            'value': state_value.item() # state_value comes from critic
        }
        return action.item()

    def store_transition_outcome(self, reward, done, raw_next_observation):
        # Scale reward before storing
        scaled_reward = reward / REWARD_SCALING_FACTOR

        self.buffer['states'].append(self._temp_action_data['state'])
        self.buffer['actions'].append(self._temp_action_data['action'])
        self.buffer['log_probs'].append(self._temp_action_data['log_prob'])
        self.buffer['values'].append(self._temp_action_data['value']) # V(s_t)
        self.buffer['rewards'].append(scaled_reward) # Store scaled reward
        self.buffer['dones'].append(done)
        self.current_buffer_size += 1

        loss = None # Initialize loss
        if self.current_buffer_size >= self.trajectory_n_steps or done:
            last_value = 0.0 # V(s_{t+N}) or V(s_terminal)
            if not done and raw_next_observation is not None:
                next_state_p_np = preprocess_observation(raw_next_observation, new_size=(self.processed_h, self.processed_w))
                next_state_tensor = torch.from_numpy(next_state_p_np).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, last_val_tensor = self.network(next_state_tensor) # Get V(s_{t+N})
                last_value = last_val_tensor.item()
            
            loss = self.learn(last_value) # learn() returns the average loss over epochs/minibatches
            self._clear_buffer()
        return loss # Return loss (or None if buffer not full and not done)

    def _compute_gae(self, rewards_np, values_np, dones_np, last_value):
        # --- DEBUG: Check inputs to GAE ---
        if np.isnan(rewards_np).any() or np.isinf(rewards_np).any(): print(f"PPO GAE DEBUG: NaN/Inf in rewards_np input: {rewards_np}")
        if np.isnan(values_np).any() or np.isinf(values_np).any(): print(f"PPO GAE DEBUG: NaN/Inf in values_np input: {values_np}")
        if np.isnan(last_value) or np.isinf(last_value): print(f"PPO GAE DEBUG: NaN/Inf in last_value input: {last_value}")
        # --- END DEBUG ---

        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        gae = 0.0
        
        # values_np contains V(s_0)...V(s_{N-1}) from the buffer
        # last_value is the bootstrapped V(s_N) or 0 if s_{N-1} was terminal
        for t in reversed(range(len(rewards_np))):
            if dones_np[t]: # If s_t was terminal, delta considers only r_t - V(s_t)
                delta = rewards_np[t] - values_np[t]
                gae = delta # GAE starts fresh after a terminal state
            else:
                # V(s_{t+1})
                next_val = values_np[t+1] if t < len(rewards_np) - 1 else last_value
                delta = rewards_np[t] + self.gamma * next_val - values_np[t]
                gae = delta + self.gamma * self.gae_lambda * gae # If s_t was terminal, (1-dones[t]) would be 0.
                                                              # But we handle terminal with 'if dones_np[t]'
            advantages[t] = gae
        
        returns_np = advantages + values_np # Q-values estimates using GAE
        return advantages, returns_np

    def learn(self, last_value): # last_value is V(s_N)
        # Convert buffer to tensors
        states_np = np.stack(self.buffer['states'])
        states_t = torch.from_numpy(states_np).float().to(self.device)
        actions_t = torch.tensor(self.buffer['actions'], device=self.device, dtype=torch.long)
        old_log_probs_t = torch.tensor(self.buffer['log_probs'], device=self.device, dtype=torch.float)
        
        rewards_np = np.array(self.buffer['rewards'], dtype=np.float32)
        dones_np = np.array(self.buffer['dones'], dtype=np.bool_)
        values_np = np.array(self.buffer['values'], dtype=np.float32) # These are V(s_t) collected at each step

        advantages_np, returns_np = self._compute_gae(rewards_np, values_np, dones_np, last_value)
        
        if np.isnan(advantages_np).any() or np.isinf(advantages_np).any(): print(f"PPO LEARN DEBUG: NaN/Inf in advantages_np after GAE: {advantages_np}")
        if np.isnan(returns_np).any() or np.isinf(returns_np).any(): print(f"PPO LEARN DEBUG: NaN/Inf in returns_np after GAE: {returns_np}")
        
        advantages_t = torch.from_numpy(advantages_np).float().to(self.device)
        returns_t = torch.from_numpy(returns_np).float().to(self.device)

        # Normalize advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std()
        if torch.isnan(adv_std) or torch.isinf(adv_std) or adv_std < 1e-8:
            print(f"PPO LEARN DEBUG: Invalid adv_std: {adv_std}. Advantages before norm: {advantages_t}. Mean: {adv_mean}")
            # If std is very small or NaN, avoid division by it; effectively no normalization or use raw advantages.
            # This can happen if trajectory_n_steps is 1 and advantages are constant.
            normalized_advantages_t = advantages_t - adv_mean # Just center if std is bad
        else:
            normalized_advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)
        
        if torch.isnan(normalized_advantages_t).any() or torch.isinf(normalized_advantages_t).any():
             print(f"PPO LEARN DEBUG: NaN/Inf in normalized_advantages_t: {normalized_advantages_t}")

        total_loss_accumulator = 0.0
        num_minibatches = 0
        
        for epoch_num in range(self.ppo_epochs):
            # Create minibatches from the full trajectory data
            num_samples = self.current_buffer_size # Should be len(states_t)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, num_samples) # Handle last smaller batch
                if start_idx == end_idx: continue # Skip if somehow start_idx >= num_samples

                mb_indices = indices[start_idx:end_idx]

                mb_states = states_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = normalized_advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                # Get new log_probs, values, and entropy from current policy
                new_action_logits, new_values_pred = self.network(mb_states) # V_phi(s_t)
                
                if torch.isnan(new_action_logits).any() or torch.isinf(new_action_logits).any():
                    print(f"PPO LEARN DEBUG Epoch {epoch_num} MB {start_idx//self.mini_batch_size}: NaN/Inf in new_action_logits: {new_action_logits}")
                    continue # Skip this minibatch if logits are bad
                if torch.isnan(new_values_pred).any() or torch.isinf(new_values_pred).any():
                    print(f"PPO LEARN DEBUG Epoch {epoch_num} MB {start_idx//self.mini_batch_size}: NaN/Inf in new_values_pred: {new_values_pred}")
                    continue

                try:
                    dist = Categorical(logits=new_action_logits)
                except ValueError as e:
                    print(f"PPO LEARN CRITICAL: ValueError creating Categorical. Logits: {new_action_logits}. Error: {e}")
                    continue
                
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                new_values_pred = new_values_pred.squeeze(-1) # Shape: (mini_batch_size,)

                # Ratio for PPO loss
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print(f"PPO LEARN DEBUG: NaN/Inf in ratio. new_log_probs: {new_log_probs}, mb_old_log_probs: {mb_old_log_probs}")
                    continue

                # Clipped Surrogate Objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value Function Loss
                critic_loss = F.mse_loss(new_values_pred, mb_returns) # mb_returns are GAE-based targets
                
                # Total Loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"PPO LEARN DEBUG: NaN/Inf in total loss. Actor: {actor_loss.item()}, Critic: {critic_loss.item()}, Entropy: {entropy.item()}")
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (helps with exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5) # Value from original stable-baselines
                self.optimizer.step()
                
                total_loss_accumulator += loss.item()
                num_minibatches += 1
        
        return total_loss_accumulator / num_minibatches if num_minibatches > 0 else 0.0

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"PPO model saved to {path}")

    def load(self, path):
        try:
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.train() # Set to train mode after loading
            print(f"PPO model loaded from {path}")
        except Exception as e:
            print(f"Error loading PPO model from {path}: {e}")