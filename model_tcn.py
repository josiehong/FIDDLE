import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            weight_norm(nn.Conv1d(n_inputs, n_outputs, 1))
            if n_inputs != n_outputs
            else None
        )

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Kaiming (He) initialization for Conv1D layers
        init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        # Initialize biases to zero if they exist
        if self.conv1.bias is not None:
            init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            init.zeros_(self.conv2.bias)

        # Downsample layer initialization (if it exists)
        if self.downsample is not None:
            init.kaiming_normal_(
                self.downsample.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.downsample.bias is not None:
                init.zeros_(self.downsample.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MS2FNet_tcn(nn.Module):
    """TCN-based encoder for MS/MS spectra with multi-task formula prediction heads.

    Encodes a binned MS/MS spectrum together with precursor metadata (m/z, NCE,
    adduct type) into a fixed-size embedding, then decodes it into four targets:
    atom-count formula vector, monoisotopic mass, total atom count, and H/C ratio.

    Args:
            config: Dict with keys: input_channels, tcn_channels, tcn_dilations,
                    tcn_kernel_sizes, tcn_dropout, mass_embedding_dim, ce_embedding_dim,
                    add_embedding_dim, num_add, embedding_dim, output_dim, and
                    {formula,mass,atomnum,hcnum}_decoder_layers.
    """

    def __init__(self, config):
        super(MS2FNet_tcn, self).__init__()
        self.env_embedding_dim = (
            config["add_embedding_dim"]
            + config["ce_embedding_dim"]
            + config["mass_embedding_dim"]
        )

        layers = []
        tcn_channels = config["tcn_channels"]
        num_levels = len(tcn_channels)
        for i in range(num_levels):
            dilation_size = config["tcn_dilations"][i]
            in_channels = config["input_channels"] if i == 0 else tcn_channels[i - 1]
            out_channels = tcn_channels[i]
            kernel_size = config["tcn_kernel_sizes"][i]
            padding = (kernel_size - 1) * dilation_size
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=config["tcn_dropout"],
                )
            ]
            if i < num_levels - 1:  # don't add pooling for the last layer
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        # self.encoder_ms = nn.Sequential(*layers)
        self.encoder_ms = nn.ModuleList(layers)

        self.embedding_m = weight_norm(
            nn.Linear(in_features=1, out_features=config["mass_embedding_dim"])
        )
        self.embedding_ce = weight_norm(
            nn.Linear(in_features=1, out_features=config["ce_embedding_dim"])
        )
        self.embedding_add = weight_norm(
            nn.Embedding(
                num_embeddings=config["num_add"] + 1,
                embedding_dim=config["add_embedding_dim"],
            )
        )

        self.fc = nn.Sequential(
            weight_norm(
                nn.Linear(
                    in_features=int(tcn_channels[-1] * 2 + self.env_embedding_dim),
                    out_features=config["embedding_dim"],
                )
            ),
            nn.ReLU(),
            weight_norm(
                nn.Linear(
                    in_features=config["embedding_dim"],
                    out_features=config["embedding_dim"],
                )
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.decoder_formula = self._build_decoder(config, "formula")
        self.decoder_mass = self._build_decoder(config, "mass")
        self.decoder_atomnum = self._build_decoder(config, "atomnum")
        self.decoder_hcnum = self._build_decoder(config, "hcnum")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                # Kaiming initialization for convolutional and linear layers with ReLU activation
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)
            ):
                # Batch normalization layers: Initialize weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                # Initialize embeddings with a normal distribution
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def _build_decoder(self, config, decoder_type):
        """Build an MLP decoder head for a given prediction target.

        Args:
                config: Model config dict. Uses key f'{decoder_type}_decoder_layers' for
                        hidden layer widths and 'output_dim' for formula output size.
                decoder_type: One of 'formula', 'mass', 'atomnum', or 'hcnum'.
                              Non-formula heads produce a scalar output (dim=1).

        Returns:
                nn.Sequential: Decoder MLP ending with LeakyReLU.
        """
        layers = []
        decoder_layers = config[f"{decoder_type}_decoder_layers"]
        input_dim = config["embedding_dim"]
        for layer_dim in decoder_layers:
            layers.append(weight_norm(nn.Linear(input_dim, layer_dim)))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            input_dim = layer_dim

        output_dim = config["output_dim"] if decoder_type == "formula" else 1
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        return nn.Sequential(*layers)

    def forward(self, x, env):
        """Forward pass.

        Args:
                x: Spectrum tensor of shape (B, L) or (B, L, C) where L is the number
                   of m/z bins and C is input_channels.
                env: Metadata tensor of shape (B, 3) — [precursor_mz, nce, adduct_index].

        Returns:
                tuple: (encoded_x, f, mass, atomnum, hcnum)
                    - encoded_x: Embedding tensor (B, embedding_dim).
                    - f:         Formula prediction (B, output_dim).
                    - mass:      Mass prediction (B,).
                    - atomnum:   Total atom count prediction (B,).
                    - hcnum:     H/C ratio prediction (B,).
        """
        # Adjust input tensors
        x = x.unsqueeze(2) if len(x.size()) == 2 else x
        x = torch.permute(x, (0, 2, 1))

        # Spectra embedding
        xs = []
        for layer in self.encoder_ms:
            x = layer(x)
            if isinstance(layer, TemporalBlock):
                xp = self.global_pool(x).squeeze(-1)
                xs.append(xp)

        # Metadata embedding
        m = self.embedding_m(env[:, 0].unsqueeze(1))
        ce = self.embedding_ce(env[:, 1].unsqueeze(1))
        add = self.embedding_add(env[:, 2].int())
        env = torch.cat([m, ce, add], dim=1)

        # Feature fusion and projection
        x = torch.cat(xs + [env], dim=1)
        # print(x.min(), x.max())
        encoded_x = self.fc(x)

        # Decoders
        f = self.decoder_formula(encoded_x)
        mass = self.decoder_mass(encoded_x).squeeze(dim=1)
        atomnum = self.decoder_atomnum(encoded_x).squeeze(dim=1)
        hcnum = self.decoder_hcnum(encoded_x).squeeze(dim=1)

        return encoded_x, f, mass, atomnum, hcnum


class FDRNet(MS2FNet_tcn):
    """MS2FNet_tcn extended with a false discovery rate (FDR) scoring head.

    Inherits the TCN encoder and formula decoders from MS2FNet_tcn and adds a
    separate MLP that takes the spectrum embedding concatenated with a candidate
    formula vector to produce a confidence score used for re-ranking.

    Args:
            config: Same as MS2FNet_tcn, plus 'fdr_decoder_layers' for hidden widths
                    of the FDR head.
    """

    def __init__(self, config):
        super(FDRNet, self).__init__(config)
        self.decoder_fdr = self._build_fdr_decoder(config)
        self.init_weights()

    def _build_fdr_decoder(self, config):
        """Build the FDR scoring MLP.

        Input dimension is embedding_dim + output_dim (embedding concatenated with
        the candidate formula vector). The final layer has no activation; sigmoid
        is applied externally during re-ranking.

        Args:
                config: Model config dict with 'embedding_dim', 'output_dim', and
                        'fdr_decoder_layers'.

        Returns:
                nn.Sequential: FDR decoder MLP with scalar output.
        """
        layers = []
        input_dim = config["embedding_dim"] + config["output_dim"]
        for layer_dim in config["fdr_decoder_layers"]:
            layers.append(weight_norm(nn.Linear(input_dim, layer_dim)))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            input_dim = layer_dim
        # No activation at the final layer — sigmoid is applied externally in rerank_by_fdr
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x, env, f):
        """Forward pass for FDR scoring.

        Args:
                x:   Spectrum tensor (B, L) or (B, L, C).
                env: Metadata tensor (B, 3) — [precursor_mz, nce, adduct_index].
                f:   Candidate formula vector (B, output_dim).

        Returns:
                fdr: FDR score tensor (B,). Apply sigmoid externally to get probability.
        """
        # Adjust input tensors
        x = x.unsqueeze(2) if len(x.size()) == 2 else x
        x = torch.permute(x, (0, 2, 1))

        # Spectra embedding
        xs = []
        for layer in self.encoder_ms:
            x = layer(x)
            if isinstance(layer, TemporalBlock):
                xp = self.global_pool(x).squeeze(-1)
                xs.append(xp)

        # Metadata embedding
        m = self.embedding_m(env[:, 0].unsqueeze(1))
        ce = self.embedding_ce(env[:, 1].unsqueeze(1))
        add = self.embedding_add(env[:, 2].int())
        env = torch.cat([m, ce, add], dim=1)

        # Feature fusion and projection
        x = torch.cat(xs + [env], dim=1)
        x = self.fc(x)

        # Decoder for fdr prediction
        x = torch.cat((x, f), dim=1)
        fdr = self.decoder_fdr(x).squeeze(dim=1)

        return fdr
