# agents/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from .agent import Agent

# Preprocessing (example, adjust as needed)
def preprocess_observation(obs, new_size=(84, 84)):
    # Convert to grayscale, resize, normalize, and change to CxHxW
    if obs is None: # Handle initial reset where obs might be None
        return np.zeros((1, new_size[0], new_size[1]), dtype=np.float32)

    img = np.array(obs) # HxWxC
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # Grayscale: HxW
    
    # Resize (Using PIL/Pillow is better for quality, but simple slicing for now or use cv2)
    # This is a very crude resize, proper resizing (e.g. cv2.resize) is recommended
    # For simplicity here, let's assume the input image is roughly proportional and we can take a section
    # Or, use torchvision.transforms if integrating more deeply
    # temp_h, temp_w = img.shape
    # if temp_h > new_size[0] and temp_w > new_size[1]:
    #     img = img[:new_size[0], :new_size[1]] # Crude crop/slice
    # else:
    #     # If smaller, pad or handle differently. For now, this example might fail.
    #     # A robust solution would use cv2.resize or similar.
    #     pass # Keep original if too small to crop meaningfully this way.
    # For now, let's assume game.py _get_observation_for_ai returns a fixed size image,
    # and we'll just do the grayscale and channel permute.
    # Actual resizing would be:
    # import cv2
    # img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


    img = img.astype(np.float32) / 255.0 # Normalize
    img = np.expand_dims(img, axis=0) # CxHxW (C=1 for grayscale)
    return img


class QNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, h=84, w=84):
        super(QNetwork, self).__init__()
        # Simplified CNN for demonstration
        # Input: (batch_size, input_channels, H, W) e.g. (N, 1, 84, 84) for grayscale
        # Or (N, 4, 84, 84) if stacking frames
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size (dummy forward pass)
        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_h = conv_output_size(conv_output_size(conv_output_size(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv_output_size(conv_output_size(conv_output_size(w, 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
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
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent):
    def __init__(self, action_size, observation_shape, # observation_shape expected as (C, H, W)
                 buffer_size=10000, batch_size=32, gamma=0.99,
                 lr=1e-4, target_update_freq=1000, eps_start=1.0,
                 eps_end=0.01, eps_decay=50000): # eps_decay in steps
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")

        # Assuming observation_shape is (C, H, W) for preprocessed images
        # If raw pixels (H, W, C) are passed, preprocess_observation will handle it
        # For QNetwork, input_channels should be C from observation_shape
        self.input_channels = observation_shape[0] if observation_shape else 1 # Default to 1 for grayscale

        self.policy_net = QNetwork(self.input_channels, action_size).to(self.device)
        self.target_net = QNetwork(self.input_channels, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def choose_action(self, observation): # observation is raw from game
        # Preprocess observation
        state = preprocess_observation(observation)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # Add batch dim

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # Get Q values from policy_net
                q_values = self.policy_net(state_tensor)
                # Choose action with max Q value
                action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = random.randrange(self.action_size)
        return action

    def store_transition(self, state, action, next_state, reward, done):
        # Preprocess states before storing
        processed_state = preprocess_observation(state)
        processed_next_state = preprocess_observation(next_state)
        self.memory.push(processed_state, action, processed_next_state, reward, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None # Not enough samples to learn

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions)) # Converts batch-array of Transitions to Transition of batch-arrays

        # Convert to tensors
        state_batch = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1) # [[0],[1]...]
        reward_batch = torch.tensor(batch.reward, device=self.device).float()
        next_state_batch = torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.bool)


        # Get Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Get max Q(s_{t+1}, a') from target network
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # If s_{t+1} is terminal, then Q value is just reward
        expected_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))

        # Compute Huber loss (or MSE)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # Optional grad clipping
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.eps_end + (self.eps_start - self.eps_end) * \
                       np.exp(-1. * self.steps_done / self.eps_decay)
        }, path)
        print(f"DQN model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0) # Handle older checkpoints
        # Epsilon will be recalculated based on steps_done
        self.policy_net.train() # Ensure policy_net is in train mode
        self.target_net.eval()  # Ensure target_net is in eval mode
        print(f"DQN model loaded from {path}")