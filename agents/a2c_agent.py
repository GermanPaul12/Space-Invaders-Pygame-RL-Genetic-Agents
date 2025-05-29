# agents/a2c_agent.py
import random
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

    
    def _clear_buffers(self): # Ensure this method is defined like this
        self.log_probs.clear()
        self.state_values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.action_logits_buffer.clear()
    
    def choose_action(self, raw_observation):
        state_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        
        log_prob = torch.tensor(0.0, device=self.device) # Default log_prob
        action_item = 0 # Default action

        with torch.no_grad(): # Always no_grad for action selection inference part
            self.network.eval() # Set to eval mode for choosing action
            action_logits, state_value = self.network(state_tensor) 
            if not self.is_evaluating: # If training, set back to train for subsequent .learn() calls
                self.network.train()


        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print(f"A2C CHOOSE_ACTION DEBUG: NaN/Inf in action_logits from network: {action_logits}")
            action_item = random.randrange(self.action_size) # Fallback
            # log_prob remains default (or could be a very small number)
        else:
            try:
                m = Categorical(logits=action_logits)
                if self.is_evaluating: # Greedy action for evaluation
                    action = torch.argmax(action_logits, dim=1)
                else: # Sample during training
                    action = m.sample()
                
                log_prob = m.log_prob(action) # Calculate log_prob for the chosen action
                action_item = action.item()
            except ValueError as e:
                print(f"A2C CHOOSE_ACTION CRITICAL: ValueError creating Categorical. Logits: {action_logits}. Error: {e}")
                action_item = random.randrange(self.action_size) # Fallback

        # Store necessary items for learning if in training mode
        if not self.is_evaluating:
            self.log_probs.append(log_prob) # Store tensor log_prob
            self.state_values.append(state_value) # Store tensor state_value
            self.action_logits_buffer.append(action_logits) # Store tensor action_logits
        
        return action_item

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self, raw_next_observation=None): 
        if self.is_evaluating or not self.log_probs: 
            self._clear_buffers() 
            return None

        # --- Check for NaNs in collected data BEFORE processing ---
        for i, lp in enumerate(self.log_probs):
            if torch.isnan(lp).any() or torch.isinf(lp).any():
                print(f"A2C LEARN DEBUG: NaN/Inf in self.log_probs[{i}]: {lp}")
                self._clear_buffers(); return None
        for i, sv in enumerate(self.state_values):
            if torch.isnan(sv).any() or torch.isinf(sv).any():
                print(f"A2C LEARN DEBUG: NaN/Inf in self.state_values[{i}]: {sv}")
                self._clear_buffers(); return None
        for i, r in enumerate(self.rewards):
            if np.isnan(r) or np.isinf(r): # rewards are Python floats/ints
                print(f"A2C LEARN DEBUG: NaN/Inf in self.rewards[{i}]: {r}")
                self._clear_buffers(); return None
        for i, alogit in enumerate(self.action_logits_buffer):
            if torch.isnan(alogit).any() or torch.isinf(alogit).any():
                print(f"A2C LEARN DEBUG: NaN/Inf in self.action_logits_buffer[{i}]: {alogit}")
                self._clear_buffers(); return None
        # --- End NaN check for collected data ---

        R = 0.0 
        if not self.dones[-1] and raw_next_observation is not None: 
            next_state_p_np = preprocess_observation(raw_next_observation, new_size=(self.processed_h, self.processed_w))
            next_state_tensor = torch.from_numpy(next_state_p_np).float().unsqueeze(0).to(self.device)
            with torch.no_grad(): 
                original_mode = self.network.training # Store original mode
                self.network.eval() 
                _, R_tensor = self.network(next_state_tensor)
                self.network.train(original_mode) # Restore original mode
            
            if torch.isnan(R_tensor).any() or torch.isinf(R_tensor).any():
                print(f"A2C LEARN DEBUG: NaN/Inf in R_tensor (bootstrap value): {R_tensor}")
                self._clear_buffers(); return None
            R = R_tensor.item()
        
        policy_discounted_returns = []
        for r_val, d_val in zip(reversed(self.rewards), reversed(self.dones)): # Use different var names
            if d_val: R = 0.0 
            R = r_val + self.gamma * R # r_val is already a float from the buffer
            policy_discounted_returns.insert(0, R)
        
        if not policy_discounted_returns: # Should not happen if self.log_probs was not empty
            print("A2C LEARN DEBUG: policy_discounted_returns is empty!")
            self._clear_buffers(); return None

        returns_t = torch.tensor(policy_discounted_returns, device=self.device, dtype=torch.float32)
        
        # Ensure buffers are not empty before cat
        if not self.log_probs: self._clear_buffers(); return None # Should have been caught earlier
        log_probs_t = torch.cat(self.log_probs) 
        if not self.state_values: self._clear_buffers(); return None
        state_values_t = torch.cat(self.state_values).squeeze(-1) 
        if not self.action_logits_buffer: self._clear_buffers(); return None
        action_logits_t = torch.cat(self.action_logits_buffer)

        if state_values_t.ndim == 0: state_values_t = state_values_t.unsqueeze(0)
        if returns_t.ndim == 0: returns_t = returns_t.unsqueeze(0)
        if log_probs_t.ndim == 0: log_probs_t = log_probs_t.unsqueeze(0)

        # --- Check for NaNs before loss calculation ---
        if torch.isnan(returns_t).any() or torch.isinf(returns_t).any(): print(f"A2C LEARN DEBUG: NaN/Inf in returns_t: {returns_t}"); self._clear_buffers(); return None
        if torch.isnan(log_probs_t).any() or torch.isinf(log_probs_t).any(): print(f"A2C LEARN DEBUG: NaN/Inf in log_probs_t: {log_probs_t}"); self._clear_buffers(); return None
        if torch.isnan(state_values_t).any() or torch.isinf(state_values_t).any(): print(f"A2C LEARN DEBUG: NaN/Inf in state_values_t: {state_values_t}"); self._clear_buffers(); return None
        if torch.isnan(action_logits_t).any() or torch.isinf(action_logits_t).any(): print(f"A2C LEARN DEBUG: NaN/Inf in action_logits_t: {action_logits_t}"); self._clear_buffers(); return None
        # --- End NaN check ---

        advantages = returns_t - state_values_t         
        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values_t, returns_t)
        entropy_term = torch.tensor(0.0, device=self.device) # Default
        try:
            if not (torch.isnan(action_logits_t).any() or torch.isinf(action_logits_t).any()):
                m_entropy = Categorical(logits=action_logits_t)
                entropy_term = m_entropy.entropy().mean()
            else:
                 print(f"A2C LEARN DEBUG: Skipped entropy due to NaN/Inf in action_logits_t")
        except ValueError as e_cat_entropy:
            print(f"A2C LEARN CRITICAL: ValueError for entropy Categorical. Logits: {action_logits_t}. E: {e_cat_entropy}")


        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_term

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"A2C LEARN DEBUG: NaN/Inf in total_loss. Actor: {actor_loss.item() if not (torch.isnan(actor_loss).any() or torch.isinf(actor_loss).any()) else 'NaN/Inf'}, Critic: {critic_loss.item() if not (torch.isnan(critic_loss).any() or torch.isinf(critic_loss).any()) else 'NaN/Inf'}, Entropy: {entropy_term.item() if not (torch.isnan(entropy_term).any() or torch.isinf(entropy_term).any()) else 'NaN/Inf'}")
            self._clear_buffers() 
            return None 

        self.optimizer.zero_grad()
        try:
            total_loss.backward()
        except RuntimeError as e_backward: # Catch specific RuntimeError
            print(f"A2C LEARN CRITICAL: RuntimeError during total_loss.backward(): {e_backward}")
            print(f"  Loss components: Actor={actor_loss}, Critic={critic_loss}, Entropy={entropy_term}")
            print(f"  Network parameters may have NaNs. Checking first few layer weights:")
            for name, param in self.network.named_parameters():
                if param.grad is None and param.requires_grad: # Grads not computed yet, check weights
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(f"    WARNING: NaN/Inf in weights of layer: {name}")
                elif param.grad is not None: # Check computed gradients
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"    WARNING: NaN/Inf in gradients of layer: {name}")
            self._clear_buffers()
            return None # Stop this learning update
            
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) 
        self.optimizer.step()
        self._clear_buffers()
        return total_loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)
        print(f"A2C model saved to {path}")

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.train()
        print(f"A2C model loaded from {path}")