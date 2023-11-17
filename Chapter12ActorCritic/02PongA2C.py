import gym
import ptan
import numpy as np
import argparse

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()
        
        self.conv = nn.Sequential(
            # Conv ist CNN, kernel_size ist Größe des Filters, stride die Schrittweite
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512), # Linear sind wie Dense Layers aus Tensorflow
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        def _get_conv_out(self, shape):
            o = self.conv(torch.zeros(1, *shape))
            return int(np.prod(o.size()))
        
        def forward(self, x):
            fx = x.float() / 256
            conv_out = self.conv(fx).view(fx.size()[0], -1)
            return self.policy(conv_out), self.value(conv_out)
        
def unpack_batch(batch, net, device="cpu") :
    """ 
    Convert batch into training tensors
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    
    for idx, exp  in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
        
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    
    # handle new rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False))
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:,0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np
    
    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    
    return states_v, actions_t, ref_vals_v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable Cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameSkip-v4"))
    """Die Umgebung wird mit der Funktion ptan....wrap_dyn umschlossen. 
    Diese Funktion wickelt die Umgebung in einer Weise ein, dass sie mit der DQN Architektur 
    kompatibel ist. Sie kann das Beobachtungsformat ändern, Aktionen vor- und nachverarbeiten, ...
    1. Das Umschalten der Beobachtungen auf Graustufen und die Skalierung auf einen festgelegten Bereich. 
    Dies hilft dabei, die Dimensionalität der Eingabe zu reduzieren und die Verarbeitung zu vereinfachen.
    2. Stapeln mehrerer aufeinanderfolgender Frames in einer einzigen Eingabe, um Informationen über die 
    Geschwindigkeit und Bewegung zu erfassen. 
    3. Anwenden von geeigneten Nachverarbeitungsschritten auf die Aktionen, wie z.B. das Überspringen 
    von Frames oder das Wiederholen von Aktionen, um die Geschwindigkeit des Spiels zu verringern oder 
    die Erkundung zu verbessern.
    """
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)