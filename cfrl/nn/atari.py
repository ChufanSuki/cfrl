from turtle import forward
import torch
import torch.nn as nn

class AtariNet(torch.nn.Module):
    def __init__(self, num_inpts, action_space):
        super(AtariNet, self).__init__()
        assert num_inpts == 4 
        self.network = nn.Sequential(
            nn.Conv2d(num_inpts, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=0,end_dim=-1),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, action_space.n)

        
    def forward(self, inputs):
        x = self.network(inputs)
        return self.critic_linear(x), self.actor_linear(x)


class AtariLSTMNet(torch.nn.Module):
    def __init__(self, num_inpts, action_space):
        super(AtariLSTMNet, self).__init__()
        assert num_inpts == 4 
        self.network = nn.Sequential(
            nn.Conv2d(num_inpts, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=0,end_dim=-1),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(512, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space.n)

        
    def forward(self, inputs):
        x = self.network(inputs)
        x = self.lstm.forward(x)
        return self.critic_linear(x), self.actor_linear(x)