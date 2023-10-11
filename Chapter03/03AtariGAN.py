import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dim of input image
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1_000

class InputWrapper(gym.ObservationWrapper):
    """Preprocessing of input numpy array:
    1. Resize image into predefined size
    2. Move color channel axis to a first place

    Args:
        gym (_type_): _description_
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32
        )
        
    def observation(self, observation):
        # resize
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE)
        )
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)
    
    """Der obige Code definiert eine Klasse "Discriminator", die ein neuronales Netzwerkmodell für die Klassifizierung von Bildern 
    implementiert. Der Discriminator nimmt ein Bild der Größe input_shape als Eingabe und gibt eine Wahrscheinlichkeit aus, dass 
    das Bild real oder generiert ist. 
    Das Netzwerk besteht aus einer Sequenz von Convolutional Layern, die schrittweise das Eingangsbild in eine immer kleinere 
    Featuremap konvergieren. Jeder Convolutional Layer besteht aus einer Convolutional Operation, gefolgt von einer nichtlinearen 
    Aktivierungsfunktion (ReLU) und optional Batch Normalization. 
    Der Discriminator verwendet auch eine spezielle Aktivierungsfunktion, die Sigmoid-Funktion, um die Ausgabe des Discriminators 
    in eine Wahrscheinlichkeit umzuwandeln. Der letzte Conv2d-Layer hat einen Ausgabekanal und wendet die Sigmoid-Funktion an, um 
    die Ausgabe zwischen 0 und 1 zu begrenzen. 
    Die forward-Methode des Discriminators führt die Vorwärtsberechnung durch, indem sie das Eingangsbild durch die conv_pipe 
    leitet und die Ausgabe des letzten Layers in eine flache Form bringt. 
    Insgesamt dient dieser Discriminator dazu, die Echtheit von Bildern zu bewerten und sie als real oder generiert zu klassifizieren."""
    
class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.pipe(x)
    
    """Der obige Code definiert eine Klasse "Generator", die ein neuronales Netzwerkmodell für die Erzeugung von Bildern implementiert. 
    Der Generator verwendet eine sogenannte Transposed Convolution (auch bekannt als Deconvolution), um aus einem Eingabevektor ein 
    Bild der Größe (3, 64, 64) zu erzeugen. 
    Das Netzwerk besteht aus einer Sequenz von Transposed Convolutional Layern, die schrittweise den Eingabevektor in eine immer 
    größere Featuremap umwandeln. Jeder Transposed Convolutional Layer besteht aus einer Transposed Convolutional Operation, 
    gefolgt von Batch Normalization und einer nichtlinearen Aktivierungsfunktion (ReLU). 
    Der Generator verwendet auch eine spezielle Aktivierungsfunktion, die Sigmoid-Funktion, um sicherzustellen, dass die Ausgabe des 
    Generators im Bereich von -1 bis 1 liegt. Dies wird durch den letzten ConvTranspose2d-Layer erreicht, der die Anzahl der Kanäle 
    auf die Anzahl der Kanäle des gewünschten Bildformats (output_shape[0]) reduziert und dann die Sigmoid-Funktion anwendet. 
    Insgesamt erzeugt dieser Generator also aus einem Eingabevektor ein Bild der Größe (3, 64, 64)."""
   
def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen =  iter(lambda: random.choice(envs), None)
       
    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
           batch. append(obs)
        if len(batch) == batch_size:
            # Normierung auf Werte zwischen -1 und 1
            batch_np = np.array(batch, dtype = np.float32)
            batch_np *= 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()
    
"""Die obenstehende Funktion "iterate_batches" implementiert eine Batch-Generierung für das Training eines neuronalen Netzwerks. 
Sie nimmt eine Liste von Umgebungen (envs) als Eingabe und generiert ständig Batches von Beobachtungen aus diesen Umgebungen. 
Zu Beginn wird ein Batch initialisiert, indem reset() für jede Umgebung in envs aufgerufen wird. Die Funktion iter(lambda: random.choice(envs), None) 
erstellt einen Iterator, der zufällig eine Umgebung aus envs auswählt. 
In der while-Schleife wird das nächste Umgebungselement ausgewählt und eine Aktion mit e.action_space.sample() ausgeführt. 
Die Beobachtung (obs), Belohnung (reward), is_done (ein boolscher Wert, der angibt, ob die Episode beendet ist) und weitere Informationen werden zurückgegeben. 
Wenn der Durchschnitt der Beobachtung größer als 0.01 ist, wird die Beobachtung dem Batch hinzugefügt. 
Sobald die Batch-Größe erreicht ist, wird der Batch normiert, indem er auf Werte zwischen -1 und 1 skaliert wird (batch_np *= 2.0 / 255.0 - 1.0). 
Der normierte Batch wird als torch.Tensor zurückgegeben und der Batch wird geleert. Wenn is_done True ist, wird die Umgebung zurückgesetzt, um eine 
neue Episode zu beginnen. Die Funktion wird als Generator implementiert, daher wird sie in einer Schleife verwendet, um kontinuierlich Batches von 
Beobachtungen zu generieren."""
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Cuda-Berechnung aktivieren")
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
    ]
    input_shape = envs[0].observation_space.shape

    """Der obenstehende Code liest Einstellungen vom Befehlszeilenaufruf ein und initialisiert Variablen für das Gerät und die Umgebungen."""

    net_discr = Discriminator(input_shape = input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)
    
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    """Der obenstehende Code initialisiert ein Diskriminator-Modell (net_discr) und ein Generator-Modell (net_gener) für das Training 
    eines GANs (Generative Adversarial Network) mit den gegebenen Eingabe- und Ausgabeformen.
    Die BCELoss-Funktion (Binary Cross Entropy Loss) wird als Objektiv (objective) für das Training verwendet.
    Der Generator-Optimierer (gen_optimizer) und der Diskriminator-Optimierer (dis_optimizer) werden beide mit dem Adam-Optimierer 
    initialisiert und die entsprechenden Parameter und Lernrate (LEARNING_RATE) werden festgelegt. Die betas-Werte (0.5, 0.999) werden 
    als Default-Werte für den Adam-Optimierer verwendet."""
    
    writer=SummaryWriter()
    
    gen_losses = []
    dis_losses = []
    iter_no = 0
    
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    
    for batch_v in iterate_batches(envs):
        # Zusätzliche gefälschte Beispiele erzeugen
        # Eingabe ist 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)
        
        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())
        
        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())
        
        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)  
        
        