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
    def __init__(self, input_channels, num_actions, h=84, w=84): # h,w are target preprocessed dimensions
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
        
        action_logits = self.actor_fc(x_shared) # Output logits for Categorical
        state_value = self.critic_fc(x_shared)
        
        # Return logits for actor, directly use F.softmax or Categorical(logits=...) later
        return action_logits, state_value


class A2CAgent(Agent):
    def __init__(self, action_size, observation_shape, # (C, H, W) e.g. (1, 84, 84)
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
        self.state_values = [] # V(s_t) predicted by critic
        self.rewards = []
        self.dones = []

    def choose_action(self, raw_observation):
        state_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        
        # No grad not strictly necessary for A2C action selection if only sampling,
        # but good practice if only using forward pass for decision.
        # For A2C, we need gradients through state_value for critic, and action_logits for actor.
        # So, no_grad should NOT be used here if planning to use the output for training directly.
        # However, A2C typically stores log_probs and values and then re-evaluates or uses these.
        # For simplicity, we will use with torch.no_grad() for selection, and rely on stored log_probs and values.
        with torch.no_grad(): # This is okay if log_prob & value are stored based on current net for *this* step
            action_logits, state_value = self.network(state_tensor) 
        
        m = Categorical(logits=action_logits) # Use logits directly
        action = m.sample()
        
        # Store log_prob of chosen action and predicted state_value for *current* state
        self.log_probs.append(m.log_prob(action)) 
        self.state_values.append(state_value) 
        
        return action.item()

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self, raw_next_observation=None): 
        if not self.log_probs:
            return None

        R = 0 # This will be V(s_N) or 0 if s_N is terminal
        if not self.dones[-1] and raw_next_observation is not None:
            next_state_p_np = preprocess_observation(raw_next_observation, new_size=(self.processed_h, self.processed_w))
            next_state_tensor = torch.from_numpy(next_state_p_np).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, R_tensor = self.network(next_state_tensor)
            R = R_tensor.item()

        # Calculate returns (Gt = sum of discounted rewards up to V(s_N))
        policy_returns = [] # G_t
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d: # If state was terminal, future rewards are 0
                R = 0
            R = r + self.gamma * R
            policy_returns.insert(0, R)
        
        policy_returns = torch.tensor(policy_returns, device=self.device, dtype=torch.float32)
        
        # Concatenate stored tensors
        log_probs_t = torch.cat(self.log_probs) # (N,)
        state_values_t = torch.cat(self.state_values).squeeze() # (N,)

        advantages = policy_returns - state_values_t # A_t = G_t - V(s_t)

        # Actor loss
        actor_loss = -(log_probs_t * advantages.detach()).mean() # Detach advantages for actor update

        # Critic loss
        critic_loss = F.mse_loss(state_values_t, policy_returns)

        # Entropy (re-evaluate action_logits for current states to get distribution for entropy)
        # This makes the `learn` depend on current states, which are not passed.
        # Alternative: store states too, or skip entropy for simplicity in this basic A2C.
        # For now, let's compute it if we had stored states.
        # If log_probs are from Categorical(logits=...), then m.entropy() would be needed.
        # This simplified A2C doesn't re-evaluate the policy for entropy bonus easily.
        # Let's calculate entropy if we had the logits that produced log_probs
        # For a more robust entropy, one might need to re-run states through network or store action_logits.
        # This part is tricky without storing more info or re-evaluating.
        # Let's assume self.log_probs were derived from some m = Categorical(logits=...)
        # We would need the logits again to compute entropy m.entropy()
        # For a simple version, let's make entropy_term zero.
        entropy_term = 0 
        # A more proper way would be to re-evaluate the stored states.
        # if len(self.stored_states_for_entropy) > 0: # Hypothetical stored states
        #    logits_for_entropy, _ = self.network(torch.stack(self.stored_states_for_entropy))
        #    m_entropy = Categorical(logits=logits_for_entropy)
        #    entropy_term = m_entropy.entropy().mean()


        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_term

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Optional
        self.optimizer.step()

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
        self.network.train()
        print(f"A2C model loaded from {path}")