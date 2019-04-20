import torch
import torch.nn as nn


activation = nn.LeakyReLU

# authors use this initializer, but it doesn't seem essential
def Initializer(layers, slope=0.2):
    for layer in layers:
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            std = 1/np.sqrt((1 + slope**2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)
        if hasattr(layer, 'bias'):
            layer.bias.data.zero_()

def Encoder(scales, depth, latent, colors):
    layers = []
    layers.append(nn.Conv2d(colors, depth, 1, padding=1, bias=True))
    kp = depth
    for scale in range(scales):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.AvgPool2d(2))
        kp = k
    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)

def Decoder(scales, depth, latent, colors):
    layers = []
    kp = latent
    for scale in range(scales - 1, -1, -1):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.Upsample(scale_factor=2))
        kp = k
    layers.extend([nn.Conv2d(kp, depth, 3, padding=1, bias=True), activation()])
    layers.append(nn.Conv2d(depth, colors, 3, padding=1, bias=True))
    Initializer(layers)
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        """
        The only modification :
                avg down scale 2d --> conv2d + stride 2
        """
        super(Discriminator, self).__init__()

        self.encoder = Encoder(scales, depth, latent, colors)
        self.fc = torch.nn.Linear(depth << (scales-1), 1)
        self.fc = Initializer([self.fc])
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(torch.mean(x, dim = 2), dim = 2)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


