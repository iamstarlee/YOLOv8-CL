import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass



class BateVAE(BaseVAE):

    def __init__(self,
                 size: Tuple,
                 view: Tuple,
                 in_channels: int,
                 outchannels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BateVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.size = size
        self.view = view
        self.outchannels = outchannels

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=outchannels, kernel_size=3, padding=1), # 512可用
            # nn.Conv2d(hidden_dims[-1], out_channels=256, kernel_size=3, padding=1), # 256可用
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(self.size)  # 自适应调整到 [512, 20, 20]
            # nn.AdaptiveAvgPool2d((40, 40))
            # nn.AdaptiveAvgPool2d((80, 80))
        )

    def encode(self, input:Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(self.view) # 20,40可用
        # result = result.view(-1, 576, 2, 2) # 80可用
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # 使用方法
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        decoded_data = self.decode(z)
        return  decoded_data
    

def main():
    hidden_dims3 = [64, 128, 256, 512] #20可用
    hidden_dims2 = [32, 64, 128, 256, 512] #40可用
    hidden_dims1 = [16, 32, 64, 128, 256, 576] #80可用

    size3 = (20,20)
    size2 = (40,40)
    size1 = (80,80)

    view23 = (-1, 512, 2, 2)
    view1 = (-1, 576, 2, 2)

    outchannels23 = 512
    outchannels1 = 576

    in_channels1 = 256
    in_channels23 = 512

    latent_dim = 102400

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # decode_model1 = BateVAE(size=size1,view=view1,in_channels=in_channels1, outchannels=outchannels1, latent_dim=latent_dim, hidden_dims=hidden_dims1, beta=8, gamma=1000, max_capacity=500)
    decode_model2 = BateVAE(size=size2,view=view23,in_channels=in_channels23, outchannels=outchannels23, latent_dim=latent_dim, hidden_dims=hidden_dims2, beta=8, gamma=1000, max_capacity=500)
    # decode_model3 = BateVAE(size=size3,view=view23,in_channels=in_channels23, outchannels=outchannels23, latent_dim=latent_dim, hidden_dims=hidden_dims3, beta=8, gamma=1000, max_capacity=500)

    return decode_model2