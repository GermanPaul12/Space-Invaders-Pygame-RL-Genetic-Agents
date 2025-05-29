# agents/a2c_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random 
from .agent import Agent
from .dqn_agent import preprocess_observation 

# Consider defining REWARD_SCALING_FACTOR in a config or a shared constants file
# For now, if you use it, ensure it's defined or passed appropriately.
# REWARD_SCALING_FACTOR = 100.0 

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
                 lr=7e-4, gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5,
                 mini_batch_size=64, a2c_epochs=4): # Added mini_batch_size and a2c_epochs
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"A2C Agent using device: {self.device}")

        self.input_channels = observation_shape[0]
        self.processed_h = observation_shape[1]
        self.processed_w = observation_shape[2]

        self.network = ActorCriticNetwork(
            self.input_channels, action_size, h=self.processed_h, w=self.processed_w
        ).to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.mini_batch_size = mini_batch_size 
        self.a2c_epochs = a2c_epochs         

        self.buffer = { # Main buffer for trajectory data
            'states': [], 
            'actions': [], 
            'rewards': [], 
            'dones': []
        }
        self.current_buffer_size = 0 # Tracks items in self.buffer for current trajectory
        self._temp_action_data = {}  # To temporarily hold data between choose_action and store_outcome
        # self.is_evaluating is inherited from Agent base class

    def choose_action(self, raw_observation):
        state_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        
        action_item = 0 
        with torch.no_grad(): 
            original_mode = self.network.training # Store current mode
            self.network.eval() # Set to eval for deterministic action selection if needed
            action_logits, _ = self.network(state_tensor) # Value not strictly needed for action choice
            self.network.train(original_mode) # Restore previous mode

        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print(f"A2C CHOOSE_ACTION DEBUG: NaN/Inf in action_logits: {action_logits}")
            action_item = random.randrange(self.action_size) 
        else:
            try:
                m = Categorical(logits=action_logits)
                if self.is_evaluating: 
                    action = torch.argmax(action_logits, dim=1)
                else: 
                    action = m.sample()
                action_item = action.item()
            except ValueError as e:
                print(f"A2C CHOOSE_ACTION CRITICAL: ValueError Categorical. Logits: {action_logits}. E: {e}")
                action_item = random.randrange(self.action_size) 

        if not self.is_evaluating: # Only store temp data if training
            self._temp_action_data = { 
                'state_np': state_np, 
                'action_item': action_item,
            }
        return action_item

    def store_outcome(self, reward, done):
        if self.is_evaluating or not self._temp_action_data: 
            self._temp_action_data = {} 
            return

        self.buffer['states'].append(self._temp_action_data['state_np'])
        self.buffer['actions'].append(self._temp_action_data['action_item'])
        # Reward scaling could be applied here if desired, e.g.:
        # scaled_reward = reward / REWARD_SCALING_FACTOR 
        # self.buffer['rewards'].append(scaled_reward)
        self.buffer['rewards'].append(reward) 
        self.buffer['dones'].append(done)
        
        self.current_buffer_size +=1 
        self._temp_action_data = {} 

    def _clear_buffers(self):
        if hasattr(self, 'buffer'): 
            for k in self.buffer: 
                if isinstance(self.buffer[k], list): 
                    self.buffer[k].clear()
            self.current_buffer_size = 0
        else: 
            self.buffer = {'states': [], 'actions': [], 'rewards': [], 'dones': []}
            self.current_buffer_size = 0

    def learn(self, raw_next_observation=None): 
        if self.is_evaluating or not self.buffer['rewards']: 
            self._clear_buffers(); return None

        R = 0.0 
        if not self.buffer['dones'][-1] and raw_next_observation is not None: 
            next_state_p_np = preprocess_observation(raw_next_observation, new_size=(self.processed_h, self.processed_w))
            next_state_tensor = torch.from_numpy(next_state_p_np).float().unsqueeze(0).to(self.device)
            with torch.no_grad(): 
                original_mode = self.network.training
                self.network.eval() 
                _, R_tensor = self.network(next_state_tensor)
                self.network.train(original_mode)
            if torch.isnan(R_tensor).any() or torch.isinf(R_tensor).any():
                print(f"A2C LEARN DEBUG: NaN/Inf in R_tensor (bootstrap value): {R_tensor}"); self._clear_buffers(); return None
            R = R_tensor.item()
        
        policy_discounted_returns = []
        for r_val, d_val in zip(reversed(self.buffer['rewards']), reversed(self.buffer['dones'])):
            if d_val: R = 0.0 
            R = r_val + self.gamma * R 
            policy_discounted_returns.insert(0, R)
        
        if not policy_discounted_returns:
            print("A2C LEARN DEBUG: policy_discounted_returns is empty!"); self._clear_buffers(); return None

        returns_t = torch.tensor(policy_discounted_returns, device=self.device, dtype=torch.float32)
        
        states_np_batch_full = np.stack(self.buffer['states'])
        states_t_full = torch.from_numpy(states_np_batch_full).float().to(self.device)
        actions_t_full = torch.tensor(self.buffer['actions'], device=self.device, dtype=torch.long)

        num_samples = len(self.buffer['states'])
        if num_samples == 0: # Should be caught by `not self.buffer['rewards']`
            self._clear_buffers(); return None

        total_loss_accumulator = 0.0
        num_update_steps = 0

        # Set network to train mode for the learning epochs
        self.network.train()

        for epoch in range(self.a2c_epochs): 
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, num_samples)
                if start_idx == end_idx: continue
                
                mb_indices = indices[start_idx:end_idx]
                mb_states = states_t_full[mb_indices]
                mb_actions = actions_t_full[mb_indices]
                mb_returns = returns_t[mb_indices] 

                current_action_logits, current_state_values = self.network(mb_states)
                current_state_values = current_state_values.squeeze(-1)

                if torch.isnan(current_action_logits).any() or torch.isinf(current_action_logits).any(): print(f"A2C LEARN DEBUG MB (Ep{epoch}): NaN/Inf current_action_logits"); continue
                if torch.isnan(current_state_values).any() or torch.isinf(current_state_values).any(): print(f"A2C LEARN DEBUG MB (Ep{epoch}): NaN/Inf current_state_values"); continue
                
                try:
                    dist = Categorical(logits=current_action_logits)
                    current_log_probs = dist.log_prob(mb_actions)
                    entropy_term = dist.entropy().mean()
                except ValueError as e_cat:
                    print(f"A2C LEARN CRITICAL MB (Ep{epoch}): Cat. ValueError. Logits: {current_action_logits}. E: {e_cat}"); continue
                
                if torch.isnan(current_log_probs).any() or torch.isinf(current_log_probs).any(): print(f"A2C LEARN DEBUG MB (Ep{epoch}): NaN/Inf current_log_probs"); continue

                advantages = mb_returns - current_state_values 
                
                actor_loss = -(current_log_probs * advantages.detach()).mean() 
                critic_loss = F.mse_loss(current_state_values, mb_returns) 
                
                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_term

                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    actor_item = actor_loss.item() if not (torch.isnan(actor_loss).any() or torch.isinf(actor_loss).any()) else 'NaN/Inf'
                    critic_item = critic_loss.item() if not (torch.isnan(critic_loss).any() or torch.isinf(critic_loss).any()) else 'NaN/Inf'
                    entropy_item = entropy_term.item() if not (torch.isnan(entropy_term).any() or torch.isinf(entropy_term).any()) else 'NaN/Inf'
                    print(f"A2C LEARN DEBUG MB (Ep{epoch}): NaN/Inf total_loss. A:{actor_item}, C:{critic_item}, E:{entropy_item}"); continue

                self.optimizer.zero_grad()
                try:
                    total_loss.backward()
                except RuntimeError as e_backward: 
                    print(f"A2C LEARN CRITICAL MB (Ep{epoch}): RuntimeError backward(): {e_backward}"); 
                    # Optionally print param/grad checks here if error persists
                    continue 
                    
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) 
                self.optimizer.step()
                total_loss_accumulator += total_loss.item()
                num_update_steps +=1
        
        self._clear_buffers() 
        return total_loss_accumulator / num_update_steps if num_update_steps > 0 else 0.0

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"A2C model saved to {path}")

    def load(self, path):
        try:
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.train() 
            print(f"A2C model loaded from {path}")
        except Exception as e:
            print(f"Error loading A2C model from {path}: {e}")

    # set_eval_mode is inherited from Agent base class