import gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter
""""""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=False, action="store_true", help="Enable mean baseline")
    args = parser.parse_args()
    
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg" + "-baseline=%s" % args.baseline)
    
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)
    
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    
    batch_states, batch_actions, batch_scales = [], [], []
    
    # Beginn der Trainingsschleife
    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        """ Bevor das Training beginnt, erstellt man eine Instanz des SummaryWriter und gibt ihm den Pfad zum 
        Verzeichnis an, in dem die Summaries gespeichert werden sollen. Während des Trainings oder der 
        Evaluation können dann verschiedene Arten von Summaries, wie z.B. Skalare (z.B. Verlust oder 
        Genauigkeit), Histogramme oder Bilder, mit Hilfe des SummaryWriters erstellt und regelmäßig in das 
        angegebene Verzeichnis geschrieben werden."""
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        if args.baseline:
            batch_scales.append(exp.reward - baseline)
        else:
            batch_scales.append(exp.reward)
        
        
        # handle new rewards
        """ Neue Belohnungen: Wenn neue Belohnungen verfügbar sind, werden sie verarbeitet, indem sie 
        zur Gesamtbelohnungsliste hinzugefügt werden und der Durchschnitt der letzten 100 Belohnungen 
        berechnet wird. Diese Werte werden zur Tensorboard-Zusammenfassung hinzugefügt. Wenn der 
        Durchschnitt der letzten 100 Belohnungen über 195 liegt, wird das Training als erfolgreich 
        abgeschlossen angesehen und die Schleife wird beendet."""
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue
        
        """Batch-Größe: Wenn genügend Erfahrungen gesammelt wurden, wird der nächste Schritt ausgeführt. 
        Der aktuelle Batch von Zuständen, Aktionen und Skalierungen wird in Tensoren umgerechnet."""
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)
        
        """Vorwärtsdurchgang: Der Vorwärtsdurchgang des neuronalen Netzwerks wird durchgeführt und die 
        log_softmax-Werte werden berechnet. Es wird auch der log_likelihood-Wert für die ausgewählten 
        Aktionen berechnet."""
        optimizer.zero_grad() # Gradienten zu 0 setzen
        logits_v = net(states_v) # Rohausgaben des Models
        log_prob_v = F.log_softmax(logits_v, dim=1) # logarithmierte Wahrscheinlichkeiten
        log_p_a_v = log_prob_v[range(BATCH_SIZE), batch_actions_t]
        log_prob_actions_v = batch_scale_v * log_p_a_v
        loss_policy_v = -log_prob_actions_v.mean()

        """Richtungsrückgabe: Der Rückwärtsdurchgang wird durchgeführt und die Gradienten der 
        Netzwerkparameter werden berechnet. Außerdem werden verschiedene Loss-Werte, wie der 
        Policy-Loss und der Entropy-Loss, berechnet und zur Tensorboard-Zusammenfassung hinzugefügt."""
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate([p.grad.data.numpy().flatten()
                                for p in net.parameters()
                                if p.grad is not None])

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        entropy_loss_v.backward()
        optimizer.step()

        loss_v = loss_policy_v + entropy_loss_v

        # calc KL-div
        """KL-Divergenz: Die KL-Divergenz zwischen den alten und den neuen 
        Wahrscheinlichkeitsverteilungen wird berechnet und zur Tensorboard-Zusammenfassung hinzugefügt."""
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)

        g_l2 = np.sqrt(np.mean(np.square(grads)))
        g_max = np.max(np.abs(grads))
        writer.add_scalar("grad_l2", g_l2, step_idx)
        writer.add_scalar("grad_max", g_max, step_idx)
        writer.add_scalar("grad_var", np.var(grads), step_idx)

        """Weiter zum nächsten Schritt: Schließlich werden alle Listen von Zuständen, Aktionen 
        und Skalierungen geleert und die Schleife wird fortgesetzt."""
        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()