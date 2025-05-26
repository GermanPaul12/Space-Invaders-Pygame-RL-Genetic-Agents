# agents/dqn_agent.py (and by extension for ppo_agent.py, genetic_agent.py, a2c_agent.py)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from .agent import Agent
from PIL import Image # Import Pillow Image

# Preprocessing (example, adjust as needed)
def preprocess_observation(obs, new_size=(84, 84)):
    # Convert to grayscale, resize, normalize, and change to CxHxW
    if obs is None: # Handle initial reset where obs might be None
        # Return a black image of the correct preprocessed shape
        return np.zeros((1, new_size[0], new_size[1]), dtype=np.float32)

    # Ensure obs is a NumPy array (it should be from game.py)
    # game._get_observation_for_ai() returns pg.surfarray.array3d() which is HxWxC
    
    # Convert Pygame surface array (HxWxC) to PIL Image
    pil_image = Image.fromarray(obs.astype(np.uint8))
    
    # Convert to grayscale
    pil_image = pil_image.convert('L') # 'L' mode is (8-bit pixels, black and white)
    
    # Resize
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS) # Use a good resampling filter
    
    # Convert back to NumPy array
    img_array = np.array(pil_image, dtype=np.float32) # Shape: (new_size_H, new_size_W)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add channel dimension: (H, W) -> (1, H, W) for PyTorch Conv2D
    img_array = np.expand_dims(img_array, axis=0) 
    
    return img_array # Shape: (1, new_size_H, new_size_W)


class QNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, h=84, w=84): # h,w are target preprocessed dimensions
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size based on h, w (which are now the actual input H, W to conv layers)
        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_h = conv_output_size(conv_output_size(conv_output_size(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv_output_size(conv_output_size(conv_output_size(w, 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x): # x should be (N, C, H, W) e.g. (N, 1, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Args are already preprocessed numpy arrays for state and next_state
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent):
    def __init__(self, action_size, observation_shape, # observation_shape expected as (C, H, W) e.g. (1, 84, 84)
                 buffer_size=10000, batch_size=32, gamma=0.99,
                 lr=1e-4, target_update_freq=1000, eps_start=1.0,
                 eps_end=0.01, eps_decay=50000):
        super().__init__(action_size, observation_shape) # observation_shape is (1, 84, 84)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")

        self.input_channels = observation_shape[0] # C from (C, H, W)
        self.processed_h = observation_shape[1] # H from (C, H, W)
        self.processed_w = observation_shape[2] # W from (C, H, W)

        self.policy_net = QNetwork(self.input_channels, action_size, h=self.processed_h, w=self.processed_w).to(self.device)
        self.target_net = QNetwork(self.input_channels, action_size, h=self.processed_h, w=self.processed_w).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def choose_action(self, raw_observation): # observation is raw from game
        # Preprocess observation using the new_size from self.processed_h, self.processed_w
        state_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device) # Add batch dim

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = random.randrange(self.action_size)
        return action

    def store_transition(self, raw_state, action, raw_next_state, reward, done):
        # Preprocess states before storing
        processed_state_np = preprocess_observation(raw_state, new_size=(self.processed_h, self.processed_w))
        processed_next_state_np = preprocess_observation(raw_next_state, new_size=(self.processed_h, self.processed_w))
        
        # ReplayBuffer stores numpy arrays (state, action, next_state, reward, done)
        self.memory.push(processed_state_np, action, processed_next_state_np, reward, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert preprocessed numpy arrays from buffer to tensors
        # batch.state is a tuple of numpy arrays, so stack them
        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.bool)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(self.batch_size, device=self.device)
        # Compute V(s_{t+1}) for all next states.
        # For non-final next states, V(s_{t+1}) = max_a Q_target(s_{t+1}, a)
        # For final states, V(s_{t+1}) = 0
        non_final_mask = ~done_batch
        non_final_next_states = next_state_batch[non_final_mask]

        if non_final_next_states.size(0) > 0: # Check if there are any non-final states
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_q_values = reward_batch + (self.gamma * next_q_values)

        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)
        print(f"DQN model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.policy_net.train()
        self.target_net.eval()
        print(f"DQN model loaded from {path}")