import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import DDPGActor, DDPGCritic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4        # learning rate of the actor 
LR_CRITIC = 1e-3       # learning rate of the critic
NOISE_EPSILON = 0.99        # L2 weight decay
NOISE_DECAY = 1
LEARN_EVERY_STEPS = 1         # learning very number of timesteps
UPDATE_RATE = 1         # number of updates per step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of agents
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = DDPGActor(state_size, action_size, random_seed).to(device)
        self.actor_target = DDPGActor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = DDPGCritic(state_size, action_size, random_seed).to(device)
        self.critic_target = DDPGCritic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)#, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_epsilon = NOISE_EPSILON

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, buffer, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        # Learn, if enough samples are available in memory
        if len(buffer) > BATCH_SIZE and timestep % (LEARN_EVERY_STEPS) == 0:
            for _ in range(UPDATE_RATE):
                experiences = buffer.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_epsilon * self.noise.sample()
            self.noise_epsilon *= NOISE_DECAY
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_list, actions_list, rewards, next_states_list, dones = experiences

        # Convert next states_list to tensor             
        next_states = torch.cat(next_states_list, dim=1).to(device)

        # Convert states_list to tensor 
        states = torch.cat(states_list, dim=1).to(device)

        # Convert actions_list to tensor 
        actions = torch.cat(actions_list, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = [self.actor_target(i_states) for i_states in states_list] 
        # Convert next_actions to tensor       
        next_actions = torch.cat(next_actions, dim=1).to(device)   
        # Get Q values from target models
        Q_targets_next = self.critic_target(next_states, next_actions)       
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # Calculate predicted actions from actor
        actions_pred = [self.actor_local(i_states) for i_states in states_list]  
        # Convert predicted actions to tensor
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        # Compute actor loss from critic
        actor_loss = -self.critic_local(states, actions_pred).mean()        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()        
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agents = 2
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = [torch.from_numpy(np.vstack([e.state[i_agent] for e in experiences if e is not None])).float().to(device) for i_agent in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.action[i_agent] for e in experiences if e is not None])).float().to(device) for i_agent in range(self.num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_state[i_agent] for e in experiences if e is not None])).float().to(device) for i_agent in range(self.num_agents)]            
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)