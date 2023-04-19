"""Hyena operator as introduced by https://arxiv.org/abs/2302.10866."""

import math

from einops import rearrange
import torch

from megatron import get_args
from megatron.enums import AttnMaskType, AttnType
from .module import MegatronModule


# def _args_to_kwargs():
#     args = get_args()

#     common_kwargs = {
#         "params_dtype": args.params_dtype,
#         "use_cpu_initialization": args.use_cpu_initialization,
#         "perform_initialization": args.perform_initialization,
#         # "gradient_accumulation_fusion": args.gradient_accumulation_fusion,
#         # "sequence_parallel_enabled": args.sequence_parallel,
#     }
#     return common_kwargs


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class OptimModule(torch.nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, torch.nn.Parameter(tensor))

            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(torch.nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            torch.nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(
            self, emb_dim: int, seq_len: int,
            lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool = True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
            self,
            d_model,
            # dim of input to MLP, augments with positional encoding
            emb_dim=3,
            order=16,  # width of the implicit MLP
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1,  # frequency of periodic activations
            wd=0,  # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding
                (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = torch.nn.Parameter(torch.randn(self.d_model))
        self.dropout = torch.nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, (
            "emb_dim must be odd and greater or equal to 3 "
            "(time, sine and cosine)")
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(torch.nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(
            torch.nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class ParallelHyena(MegatronModule):
    """Parallel Hyena operator layer.

    Drop-in replacement for attention layers.

    The Hyena operator layer takes input with size [s, b, h] and returns
    output of the same size.
    """

    def __init__(
            self,
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.padding,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            **filter_args,
    ):
        super(ParallelHyena, self).__init__()
        args = get_args()
        assert attn_mask_type is AttnMaskType.causal, \
            'Hyena currently only supports causal attention'

        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type

        d_model = args.hidden_size
        l_max = args.seq_length
        order = args.hyena_order
        filter_order = args.hyena_filter_order
        dropout = args.hyena_dropout
        filter_dropout = args.hyena_filter_dropout

        modulation_args = dict(
            # Modulation kwargs.
            fast_decay_pct=args.hyena_fast_decay_pct,
            slow_decay_pct=args.hyena_slow_decay_pct,
            target=args.hyena_modulation_target,
            modulation_lr=args.hyena_modulation_lr,
            modulate=args.hyena_modulate,
            shift=args.hyena_shift,
        )
        filter_args = dict(
            emb_dim=args.hyena_emb_dim,
            fused_fft_conv=args.hyena_fused_fft_conv,
            lr=args.hyena_lr,
            lr_pos_emb=args.hyena_pos_emb_lr,
            w=args.hyena_activation_frequency,
            wd=args.hyena_weight_decay,
            bias=args.hyena_bias,
            num_inner_mlps=args.hyena_num_inner_mlps,
            normalized=args.hyena_normalized,
            **modulation_args,
        )

        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)

        self.dropout = torch.nn.Dropout(dropout)
        self.in_proj = torch.nn.Linear(d_model, inner_width)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj_bias = self.out_proj.bias
        self.out_proj.bias = None

        self.short_filter = torch.nn.Conv1d(
            inner_width,
            inner_width,
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args
        )

        self.to(args.params_dtype)

    def forward(
            self, u, attention_mask, layer_past=None, get_key_value=False,
            encoder_output=None, alibi=None,
    ):
        assert (
            layer_past is None
            and get_key_value is False
            and encoder_output is None
            and alibi is None
        ), 'arguments are not supported by Hyena'
        length = u.size(-2)
        l_filter = min(length, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y, self.out_proj_bias
