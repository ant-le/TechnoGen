import torch.nn as nn
import torch
import yaml

from encoder import Encoder
from decoder import Decoder
from vectorQuantiser import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.channels = config["model"]["channels"]
        self.dim_embedding = config["model"]["dim_embedding"]
        self.n_embedding = config["model"]["n_embedding"]
        self.kernel_size = config["model"]["kernel_size"]
        self.encoder = Encoder(self.channels, self.kernel_size)
        self.to_embd_conv = nn.Conv2d(
            self.channels[-1],
            self.dim_embedding,
            kernel_size=1,
        )
        self.quantizer = VectorQuantizer(self.dim_embedding, self.n_embedding)
        self.from_embd_conv = nn.Conv2d(
            self.dim_embedding,
            self.channels[-1],
            kernel_size=1,
        )
        self.decoder = Decoder(self.channels[::-1], self.kernel_size)

    def forward(self, x):
        out_encoded = self.encoder(x)
        out_embd = self.to_embd_conv(out_encoded)
        embd, loss, indeces = self.quantizer(out_embd)
        in_embd = self.from_embd_conv(embd)
        out_decoded = self.decoder(in_embd)
        return {
            "spectogram": out_decoded,
            "embedding": embd,
            "losses": loss,
            # "indices": indeces,
        }

    def decode_from_codebook_indices(self, indices):
        quantized_output = self.quantizer.quantize_indices(indices)
        dec_input = self.post_quant_conv(quantized_output)
        return self.decoder(dec_input)


if __name__ == "__main__":
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    net = VQVAE(config=config)
    net.eval()
    input = torch.rand(10, 1, 100, 100)
    out = net.forward(input)
    module_list = net.__dict__["_modules"].values()

    print("Found modules:")
    for idx, module in enumerate(module_list):
        print("{}:\t{}".format(idx + 1, module))
    print("-----------")

    print(input.shape, out["spectogram"].shape)
    print(out["spectogram"])
