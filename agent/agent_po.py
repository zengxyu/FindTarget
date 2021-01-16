import torch
import torch.nn.functional as F
import random

from torch import optim

from env.grid_env import ACTION
from network.replay_buffer import ReplayBuffer

UPDATE_EVERY = 10


class PPOPolicy(torch.nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        self.con1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=6)
        self.con2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # self.con3 = torch.nn.Conv2d(32, 32, 3)
        # self.sm = torch.nn.Softmax2d()
        self.fc1 = torch.nn.Linear(128, 32)
        # self.fc2 = torch.nn.Linear(64, 64)
        # self.fc3 = torch.nn.Linear(64, 64)

        self.fc_pose = torch.nn.Linear(2, 32)

        self.fc_val = torch.nn.Linear(64, 1)
        self.fc_pol = torch.nn.Linear(64, action_space_size)

        # Initialize neural network weights
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear or type(m) is torch.nn.Conv2d:
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, frame, robot_pose):
        out = self.con1(frame)
        out = F.relu(out)
        out = self.con2(out)
        out = F.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)

        out_pose = self.fc_pose(robot_pose)
        out_pose = F.relu(out_pose)
        out = torch.cat((out, out_pose), dim=1)

        val = self.fc_val(out)
        pol = self.fc_pol(out)
        pol = F.softmax(pol, dim=1)
        return val, pol


class Agent:
    def __init__(self, field=None):
        self.name = "grid world"
        self.seed = random.seed(42)
        self.batch_size = 32
        self.gamma = 0.8
        self.eps = 0.2

        self.action_size = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 目标target
        self.policy_net = PPOPolicy(len(ACTION)).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayBuffer(buffer_size=100000, batch_size=self.batch_size, seed=self.seed)

        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.time_step += 1
        if self.time_step % UPDATE_EVERY == 0:
            if len(self.memory) > self.batch_size:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def load_model(self, file_path):
        state_dict = torch.load(file_path, map_location=None)
        self.policy_net.load_state_dict(state_dict)

    def store_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def reset(self):
        pass

    def act(self, frame, robot_pose):
        """

        :param frame: [w,h]
        :param robot_pose: [1,2]
        :return:
        """
        frame_in = torch.Tensor([frame]).to(self.device)
        robot_pose_in = torch.Tensor([robot_pose]).to(self.device)

        self.policy_net.eval()

        with torch.no_grad():
            value, pol = self.policy_net(frame_in, robot_pose_in)
            action = torch.distributions.Categorical(pol).sample().cpu().data.numpy()[0]
        self.policy_net.train()

        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        frames_in, robot_poses_in = states
        next_frames_in, next_robot_poses_in = next_states

        new_vals, new_probs = self.policy_net(torch.cat([frames_in], dim=0).to(self.device),
                                              torch.cat([robot_poses_in], dim=0).to(self.device))

        new_pol = torch.distributions.Categorical(new_probs)

        # action_tensor = torch.cat(actions, dim=0)

        returns_tensor = torch.cat([rewards], dim=0)

        loss_val = F.mse_loss(new_vals, returns_tensor)

        # loss_ent = -F.nll_loss(new_probs, action_tensor) #F.cross_entropy(new_probs, action_tensor)
        loss_ent = -new_pol.entropy().mean()

        c1, c2 = 1, 0.01

        loss = c1 * loss_val + c2 * loss_ent

        # loss.backward(retain_graph=True)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
