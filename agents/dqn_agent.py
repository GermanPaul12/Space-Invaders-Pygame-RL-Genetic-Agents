# agents/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from .agent import Agent
from PIL import Image 

def preprocess_observation(obs, new_size=(84, 84)):
    if obs is None: 
        return np.zeros((1, new_size[0], new_size[1]), dtype=np.float32)
    pil_image = Image.fromarray(obs.astype(np.uint8))
    pil_image = pil_image.convert('L') 
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS) 
    img_array = np.array(pil_image, dtype=np.float32) 
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

class QNetwork(nn.Module):
    # ... (QNetwork definition as before, no changes here) ...
    def __init__(self, input_channels, num_actions, h=84, w=84): 
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
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
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    # ... (ReplayBuffer definition as before, no changes here) ...
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQNAgent(Agent):
    def __init__(self, action_size, observation_shape, 
                 buffer_size=10000, batch_size=32, gamma=0.99,
                 lr=1e-4, target_update_freq=1000, eps_start=1.0,
                 eps_end=0.01, eps_decay=50000):
        super().__init__(action_size, observation_shape) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")

        self.input_channels = observation_shape[0] 
        self.processed_h = observation_shape[1] 
        self.processed_w = observation_shape[2] 

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
        # self.is_evaluating is inherited from Agent base class

    def choose_action(self, raw_observation):
        state_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)

        eps_threshold = self.eps_end # Default to greedy for safety or if not training
        if not self.is_evaluating: # If in training mode, use epsilon-greedy
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            np.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1 # Only increment steps_done during training exploration
        else: # In evaluation mode
            eps_threshold = 0.0 # Force greedy action

        if random.random() > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval() # Ensure policy_net is in eval mode for inference
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].view(1, 1).item()
                if not self.is_evaluating: # Switch back to train mode if it was training
                    self.policy_net.train() 
        else:
            action = random.randrange(self.action_size)
        return action

    def store_transition(self, raw_state, action, raw_next_state, reward, done):
        processed_state_np = preprocess_observation(raw_state, new_size=(self.processed_h, self.processed_w))
        processed_next_state_np = preprocess_observation(raw_next_state, new_size=(self.processed_h, self.processed_w))
        self.memory.push(processed_state_np, action, processed_next_state_np, reward, done)

    def learn(self):
        # ... (learn method as before, no changes needed here regarding eval_mode) ...
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.bool)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = ~done_batch
        non_final_next_states = next_state_batch[non_final_mask]
        if non_final_next_states.size(0) > 0: 
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
        # ... (save method as before) ...
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'eps_start': self.eps_start, # Save exploration params too
            'eps_end': self.eps_end,
            'eps_decay': self.eps_decay
        }, path)
        print(f"DQN model saved to {path}")


    def load(self, path):
        # ... (load method as before, ensure to load exploration params if saved) ...
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        # Load exploration parameters if they were saved
        self.eps_start = checkpoint.get('eps_start', self.eps_start)
        self.eps_end = checkpoint.get('eps_end', self.eps_end)
        self.eps_decay = checkpoint.get('eps_decay', self.eps_decay)
        
        self.policy_net.train() # Ensure policy_net is in train mode after loading for further training
        self.target_net.eval()  # Target net is always in eval mode
        print(f"DQN model loaded from {path}")

    # set_eval_mode is inherited from Agent base class