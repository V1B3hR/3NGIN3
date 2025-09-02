# metamorph_3ngin3_pz_full.py
"""
Fully PettingZoo-compatible prototype:
- 3NGIN3 DuetMindAgent integrated
- Actor-Critic learning heads
- Centralized Critic (CTDE)
- Adaptive meta-controller
- Runs multi-agent experiments in PettingZoo
"""

import numpy as np
from collections import deque, namedtuple
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v2

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition", ("obs","msg","action","reward","next_obs","next_msg","done"))

# -----------------------------
# DuetMindAgent Wrapper
# -----------------------------
try:
    from 3NGIN3.DuetMindAgent import DuetMindAgent as BaseDuetAgent
except ImportError:
    class BaseDuetAgent:
        def __init__(self, **kwargs): pass
        def compute_action(self, obs, msg): return np.random.randint(0, 5)
        def generate_message(self, obs, action): return np.random.randn(len(obs)//2)

class DuetMindAgentWrapper:
    def __init__(self, agent_id, obs_dim, msg_dim, action_dim, lr=3e-4, gamma=0.99):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.action_dim = action_dim
        self.temperature = 1.0
        self.gamma = gamma
        self.device = Device

        # 3NGIN3 reasoning
        self.agent = BaseDuetAgent()

        # Lightweight actor-critic heads
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        ).to(self.device)

        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

    def act(self, obs, msg, deterministic=False):
        # Base 3NGIN3 reasoning for action
        base_action = self.agent.compute_action(obs, msg)
        out_msg = self.agent.generate_message(obs, base_action)
        out_msg = np.resize(np.array(out_msg), (self.msg_dim,))

        # Actor-Critic logits for updates
        obs_t = torch.tensor(np.concatenate([obs, msg]), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(obs_t)
        probs = torch.softmax(logits / max(1e-8,self.temperature), dim=-1)
        action = int(torch.multinomial(probs,1).item()) if not deterministic else int(torch.argmax(probs).item())

        return action, out_msg

    def update_from_batch(self, batch, global_critic=None):
        obs = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=self.device)
        msg = torch.tensor(np.stack(batch.msg), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=self.device)
        next_msg = torch.tensor(np.stack(batch.next_msg), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        x = torch.cat([obs, msg], dim=-1)
        logits = self.policy_net(x)
        values = self.value_net(x).squeeze(-1)
        logp = torch.log_softmax(logits, dim=-1)
        chosen_logp = logp[range(len(actions)), actions]

        # centralized critic returns
        if global_critic:
            with torch.no_grad():
                global_obs_msg = torch.cat([x], dim=-1)
                next_values = global_critic(global_obs_msg).squeeze(-1)
            returns = compute_returns(rewards.cpu().numpy(), dones.cpu().numpy(),
                                      next_values.detach().cpu().numpy(), self.gamma)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        else:
            returns = rewards + self.gamma * values * (1 - dones)

        advantages = returns - values
        policy_loss = -(chosen_logp * advantages.detach()).mean()
        value_loss = 0.5 * (advantages**2).mean()
        entropy = -(torch.exp(logp) * logp).sum(dim=-1).mean()
        loss = policy_loss + 0.5*value_loss - 0.01*entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + list(self.value_net.parameters()), 0.5)
        self.optimizer.step()
        return {'policy_loss':policy_loss.item(),'value_loss':value_loss.item(),'entropy':entropy.item()}

# -----------------------------
# Coordinator / Centralized Critic
# -----------------------------
class Coordinator:
    def __init__(self, agents: List[DuetMindAgentWrapper], msg_dim):
        self.agents = agents
        self.msg_dim = msg_dim
        self.device = Device
        total_obs_dim = sum(a.obs_dim for a in agents)
        total_msg_dim = msg_dim * len(agents)
        self.critic = nn.Sequential(
            nn.Linear(total_obs_dim + total_msg_dim,128),
            nn.ReLU(),
            nn.Linear(128,1)
        ).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.adapt_history = deque(maxlen=500)

    def aggregate_messages(self, messages: Dict[str,np.ndarray]):
        stack = np.stack([messages[a.agent_id] for a in self.agents], axis=0)
        mean_msg = np.mean(stack, axis=0)
        return {a.agent_id: mean_msg + 0.01*(messages[a.agent_id]-mean_msg) for a in self.agents}

    def adapt(self, avg_reward: float):
        self.adapt_history.append(avg_reward)
        if len(self.adapt_history) < 10: return
        trend = self.adapt_history[-1]-self.adapt_history[0]
        for agent in self.agents:
            if trend <0: agent.temperature = min(agent.temperature*1.05+0.01,5.0)
            else: agent.temperature = max(agent.temperature*0.98-0.001,0.1)

    def train_critic(self, global_obs, global_msgs, returns):
        x = torch.tensor(np.concatenate([global_obs, global_msgs], axis=-1), dtype=torch.float32, device=self.device)
        y = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        preds = self.critic(x)
        loss = nn.functional.mse_loss(preds,y)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.item()

# -----------------------------
# Helper
# -----------------------------
def compute_returns(rewards, dones, next_values, gamma):
    returns = []
    R = next_values[-1] if len(next_values)>0 else 0.0
    for r,d,nv in zip(reversed(rewards), reversed(dones), reversed(next_values)):
        R = r + gamma*R*(1-d)
        returns.insert(0,R)
    return np.array(returns)

# -----------------------------
# Training Loop
# -----------------------------
def run_training_pz(episodes=50, steps_per_episode=25):
    env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5)
    env.reset()
    agent_ids = list(env.agents)
    obs_dim = env.observation_spaces[agent_ids[0]].shape[0]
    action_dim = env.action_spaces[agent_ids[0]].n
    msg_dim = 4

    agents = [DuetMindAgentWrapper(i, obs_dim, msg_dim, action_dim) for i in agent_ids]
    coord = Coordinator(agents, msg_dim)
    buffers = {agent.agent_id: [] for agent in agents}

    for ep in range(episodes):
        obs_dict = env.reset()
        messages = {agent.agent_id: np.zeros(msg_dim) for agent in agents}
        ep_rewards = {agent.agent_id:0 for agent in agents}
        done = {agent.agent_id:False for agent in agents}

        for t in range(steps_per_episode):
            aggregated = coord.aggregate_messages(messages)
            actions = {}
            out_messages = {}
            for agent in agents:
                if not done[agent.agent_id]:
                    a, out_msg = agent.act(obs_dict[agent.agent_id], aggregated[agent.agent_id])
                    actions[agent.agent_id] = a
                    out_messages[agent.agent_id] = out_msg

            obs_dict, rewards, dones, _ = env.step(actions)
            for agent in agents:
                ep_rewards[agent.agent_id] += rewards[agent.agent_id]
                buffers[agent.agent_id].append(
                    Transition(obs_dict[agent.agent_id], aggregated[agent.agent_id], actions.get(agent.agent_id,0),
                               rewards[agent.agent_id], obs_dict[agent.agent_id], out_messages.get(agent.agent_id,np.zeros(msg_dim)), dones[agent.agent_id])
                )
            messages.update(out_messages)
            done.update(dones)

        # Batch update (simple for demo)
        for agent in agents:
            if len(buffers[agent.agent_id])>5:
                batch = buffers[agent.agent_id]
                agent.update_from_batch(batch, global_critic=coord.critic)
                buffers[agent.agent_id] = []

        avg_reward = np.mean(list(ep_rewards.values()))
        coord.adapt(avg_reward)
        print(f"[EP {ep}] Avg Reward: {avg_reward:.3f} | Temps: {[round(a.temperature,3) for a in agents]}")

    env.close()
    return agents, coord

# -----------------------------
if __name__=="__main__":
    run_training_pz()
