import os
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, tensor
from torch.nn import functional as F
from torch import Tensor
from torch.optim.adam import Adam

# %matplotlib inline


class LinearModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(LinearModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.apply(self.init_weights)
        self.hidden_dim = hidden_dim

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)


class DQN(nn.Module):
    def __init__(self, observation_space: int, action_space: int):
        super(DQN, self).__init__()
        self.input_dim = observation_space
        self.output_dim = action_space

        self.model = LinearModel(self.input_dim, self.output_dim)
        self.target_model = LinearModel(self.input_dim, self.output_dim)
        # try loading saved params:
        self._load()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99

        # don't want to update grad here

        for target_param in self.target_model.parameters():
            target_param.requires_grad = False

        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.batches = []
        self.to(self.device)

    def act(self, epsilon: float, state: Tensor):
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.output_dim, (1,)).cpu().numpy()
        else:
            state=tensor(state, device=self.device, dtype=torch.float32)
            return self.model.forward(state).argmax(keepdim=True).cpu().numpy()

    def train(self, batch: Tensor):
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.85)
        self.optimizer.step()
        return loss.item()

    def _compute_loss(self, batch: Tensor):
        
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        batch_size = states.shape[0]
        with torch.no_grad():
            q_next_target: Tensor = self.target_model(next_states)
            q_next_value: Tensor = self.model(next_states)

        action_prime = q_next_value.argmax(-1)
        max_q_next = q_next_target.gather(-1, action_prime.unsqueeze(-1))

        current_Q: Tensor = self.model(states)
        current_Q = current_Q.gather(-1, actions)

        target_q: Tensor = rewards.reshape(batch_size,1) + self.gamma * max_q_next * (1 - dones.reshape(batch_size,1))

        return F.mse_loss(current_Q, target_q.detach())

    def transfer_params(self):
        model_params = self.model.parameters()
        target_params = self.target_model.parameters()
        for model_param, target_param in zip(model_params, target_params):
            target_param.data.copy_(model_param.data)

    def save(self):
        torch.save(self.model.state_dict(), "model.pth")
        torch.save(self.target_model.state_dict(), "target_model.pth")

    def _load(self):
        if os.path.exists("model.pth"):
            print("Loading saved models")
            self.model.load_state_dict(torch.load("model.pth"))
            self.target_model.load_state_dict(torch.load("target_model.pth"))

    def add_batch(
        self, state: list, action: int, next_state: list, reward: int, done: int
    ):
       
        batch = {
            "states": tensor(state, device=self.device, dtype=torch.float32),
            "actions": tensor(action, device=self.device, dtype=torch.long),
            "next_states": tensor(next_state, device=self.device, dtype=torch.float32),
            "rewards": tensor(reward, device=self.device, dtype=torch.float32),
            "dones": tensor(done, device=self.device, dtype=torch.int8),
        }
        self.update_batch(batch)

    def update_batch(self, batch: Tensor):
        if len(self.batches) > 10000:
            print("Clearing batches")
            self.batches = []
        self.batches.append(batch)

    def sample_batch(self, batch_size: int):
        max_batch_size = len(self.batches)
        if max_batch_size < batch_size:
            batch_size = max_batch_size
        indices = np.random.choice(range(len(self.batches)), size=batch_size)
        batch = {}
        batch["states"] = torch.stack([self.batches[i]["states"] for i in indices])
        batch["actions"] = torch.stack([self.batches[i]["actions"] for i in indices])
        batch["next_states"] = torch.stack(
            [self.batches[i]["next_states"] for i in indices]
        )
        batch["rewards"] = torch.stack([self.batches[i]["rewards"] for i in indices])
        batch["dones"] = torch.stack([self.batches[i]["dones"] for i in indices])
        return batch


config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": False,
        "order": "sorted",
    }
}


def main():
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.configure(config)

    max_steps = 1000
    dqn = DQN(len(config["observation"]["features"]), 5)
    epsilon = 1
    epsilon_decay = 0.999
    epsilon_min = 0.15
    batch_size = 32
    checkpoint = 100
    for generation in range(20):
        print(f"Generation: {generation}")
        obs, info = env.reset()
        state = obs[0]
        cumulative_reward = 0
        #epsilon = 1
        epsilon_decay = 0.0999
        epsilon_min = 0.15
        batch_size = 32
        checkpoint = 100
        for i in range(max_steps):
            if epsilon % 0.1 == 0:
                print(f"Step: {i}, Epsilon: {epsilon}")

            action = dqn.act(epsilon, state)
            #print(f'Action: {action}')
            obs, reward, done, truncated, info = env.step(action[0])
            next_state = obs[0]

            dqn.add_batch(state, action, next_state, reward, done)
            state = next_state
            if i % batch_size == 0:
                batch = dqn.sample_batch(batch_size)
                loss = dqn.train(batch)
                print(f"Step: {i}, Loss: {loss}")
            if i % checkpoint == 0:
                dqn.transfer_params()
                dqn.save()
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            cumulative_reward += reward
            if done:
                break
            # env.render()
        print(f"Generation: {generation}, Cumulative Rewards: {cumulative_reward}")
    env.close()


def render_run():
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.configure(config)
    obs, info = env.reset()
    state = obs[0]
    dqn = DQN(len(config["observation"]["features"]), 5)
    dqn.load()
    max_steps = 1000
    for i in range(max_steps):
        action = dqn.act(0, state)
        obs, reward, done, truncated, info = env.step(action)
        state = obs[0]
        if done:
            break
        env.render()
    env.close()


# plt.imshow(env.render())
# plt.show()
if __name__ == "__main__":
    main()
    # render_run()
