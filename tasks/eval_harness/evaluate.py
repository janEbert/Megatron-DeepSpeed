import argparse
from functools import reduce
import importlib
from logging import logMultiprocessing
import os
import sys
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
import numpy as np

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.t5_dataset import (
    make_attention_mask_3d,
    make_history_mask_3d,
)
from megatron.training import setup_model_and_optimizer, get_model
from megatron.mpu.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import (
    get_ltor_masks_and_position_ids,
    get_prefix_indices,
    unwrap_model,
)
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.gpt_model import GPTModelPipe
from megatron.model.module import Float16Module
from deepspeed.runtime.pipe import schedule
from deepspeed.runtime.pipe.engine import PipelineEngine

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.eod

        self._max_length = args.seq_length
        self._prefix_tokens = args.prefix_tokens
        self._prefix_token_ids = [
            self.tokenizer.tokenizer.convert_tokens_to_ids(token)
            for token in self._prefix_tokens
        ]

        # TODO More general check for pipelined models would be desirable.
        self._is_encoder_decoder = not (
                isinstance(self.model, GPTModelPipe)
                or isinstance(self.model, PipelineEngine)
                or hasattr(self.model, 'language_model')
                and hasattr(self.model.language_model, 'add_decoder')
                and not self.model.language_model.add_decoder
        )

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size * args.micro_bs_multiplier

        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self._device = torch.cuda.current_device()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        if self.is_data_parallel:
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def _prepend_prefix_token_ids(self, tokens):
        if not self._prefix_token_ids:
            pass
        elif tokens and tokens[0] == self.EOT_TOKEN_ID:
            tokens = tokens[:1] + self._prefix_token_ids + tokens[1:]
        else:
            tokens = self._prefix_token_ids + tokens
        return tokens

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, ctxlens, contlens, inplens, padding_length = [], [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    context_len = len(context_enc) + len(self._prefix_tokens)
                    total_len = context_len + len(continuation_enc)

                    context_num_truncated = max(
                        total_len - self.max_length + 1, 0)
                    # Need actual truncated length of context here
                    # (without prefix tokens).
                    continuation_num_truncated = max(
                        context_num_truncated - len(context_enc), 0)

                    context_enc = context_enc[context_num_truncated:]
                    continuation_enc = \
                        continuation_enc[continuation_num_truncated:]

                    # Add prefix token after truncation.
                    context_enc = self._prepend_prefix_token_ids(context_enc)

                    inp = torch.tensor(
                        context_enc + continuation_enc,
                        dtype=torch.long,
                    ).to(self.device)
                    inplen, = inp.shape

                    if len(continuation_enc) == 0:
                        ctxlen = 1
                    else:
                        ctxlen = max(context_len - context_num_truncated, 1)

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)
                    ctxlens.append(ctxlen)

                logits = self._model_call((torch.cat(inps, dim=0), ctxlens))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        args = get_args()

        if isinstance(tokens, tuple) and len(tokens) == 2:
            tokens, ctxlens = tokens
        else:
            ctxlens = None

        # TODO Handle encoder-only
        if not self._is_encoder_decoder:
            if args.prefix_lm and ctxlens is not None:
                prefix_indices = get_prefix_indices(
                    tokens,
                    self.EOT_TOKEN_ID,
                    partial_prefix_indices=ctxlens,
                    reset_attention_mask=args.reset_attention_mask
                )
            else:
                if args.prefix_lm:
                    print(
                        'Warning: requested PrefixLM inputs, but cannot determine '
                        'prefix length – prefix is empty.'
                    )
                prefix_indices = None

            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                self.EOT_TOKEN_ID,
                args.reset_position_ids,
                args.reset_attention_mask,
                args.eod_mask_loss,
                prefix_indices=prefix_indices,
                loss_on_targets_only=False)
            return (tokens, position_ids, attention_mask), (tokens, loss_mask)
        else:
            assert ctxlens is not None

            # Split tokens to separate encoder and decoder input.
            # No BOS token used with eval harness, so we do not need to
            # worry about the decoder receiving it in mind of the split.
            enc_tokens = torch.stack([
                F.pad(tok[:ctxlen], (0, len(tok) - ctxlen), value=0)
                for (tok, ctxlen) in zip(tokens, ctxlens)
            ])
            dec_tokens = torch.stack([
                F.pad(tok[ctxlen:], (0, ctxlen), value=0)
                for (tok, ctxlen) in zip(tokens, ctxlens)
            ])

            enc_attn_mask = make_attention_mask_3d(enc_tokens, enc_tokens)
            dec_attn_mask = make_attention_mask_3d(dec_tokens, dec_tokens)
            dec_attn_mask *= make_history_mask_3d(dec_tokens)
            enc_dec_attn_mask = make_attention_mask_3d(dec_tokens, enc_tokens)

            loss_mask = torch.ones(
                dec_tokens.shape[:2],
                device=dec_tokens.device,
                dtype=dec_tokens.dtype,
            )
            for (i, ctxlen) in enumerate(ctxlens):
                if ctxlen != 0:
                    loss_mask[i, -ctxlen:] = 0

            return (
                (enc_tokens, dec_tokens, enc_attn_mask,
                 dec_attn_mask, enc_dec_attn_mask),
                (dec_tokens, loss_mask)
            )

    def _model_call(self, inps):
        args = get_args()

        if isinstance(inps, tuple) and len(inps) == 2:
            inps, ctxlens = inps
        else:
            ctxlens = None

        if args.deepspeed:
            self.model.set_batch_fn(self.create_model_inputs)
            # round up to multiple of micro_batch_size
            new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
            padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
            ctxlens = ctxlens + [1] * (new_size - len(ctxlens))
            # dummy data iterator for pipelining.
            data_iterator = list((
                (torch.stack(inp), ctxlen)
                for (inp, ctxlen) in zip(
                        utils.chunks(padded, args.micro_batch_size),
                        utils.chunks(ctxlens, args.micro_batch_size),
                )
            ))
            self.model.micro_batches = len(data_iterator)
            
            if self.adaptive_seq_len:
                # Allow different shapes than the default seq_len to be communicated across pipes
                # Without this Deepspeed will hang when trying to receive activations
                self.model.reset_activation_shape()
            
            output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)


            if output is not None:
                output = torch.cat(output, 0)[:len(inps)]
            else:
                output = None

            # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
            if args.adaptive_seq_len:
                self.model.total_loss = None
        else:
            # Since the shape of the micro-batch will change
            # We need set the correct shapes here
            # So that latter pipeline stages knows which shapes to expect.
            # Otherwise we will deadlock.

            args.micro_batch_size = len(inps)
            args.seq_length = len(inps[0])
            args.max_position_embeddings = args.seq_length

            input_tensor = recv_forward()

            # Forward pass through the model.
            unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)
            output = self.model(*self.create_model_inputs((inps, ctxlens))[0])
            if isinstance(output, tuple):
                output = output[0]
            send_forward(output)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.tokenizer.tokenizer, GPT2Tokenizer):
            return self.tokenizer.tokenizer.encode(text)
        else:
            return self.tokenizer.tokenizer.encode(text, add_special_tokens=False)


from megatron.initialize import initialize_megatron
import megatron

from deepspeed.checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(args):
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None

    if not os.path.exists(args.load):
        raise ValueError(f"checkpoint path {args.load} doesn't exit")

    try:
        is_ds_cp = True
        ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                            tp_degree=args.tensor_model_parallel_size,
                                            pp_degree=args.pipeline_model_parallel_size)

        cp_args = ds_checkpoint.get_args()
    except (AssertionError, ZeroDivisionError):
        is_ds_cp = False
        cp_path = os.path.join(args.load, 'mp_rank_00', 'model_optim_rng.pt')
        state_dict = torch.load(cp_path, map_location='cpu')
        cp_args = state_dict['args']
    # Merge the current args with the checkpoint args.
    skip_keys = [
        'abort_on_unmet_fused_kernel_constraints',
        'batch_size',
        'data_parallel_size',
        'deepspeed',
        'deepspeed_config',
        'device_count',
        'global_batch_size',
        'inference',
        'iteration',
        'load',
        'local_rank',
        'micro_batch_size',
        'pipeline_model_parallel_size',
        'rampup_batch_size',
        'rank',
        'tensor_model_parallel_size',
        'tensorboard_dir',
        'world_size',
    ]

    skip_if_specified = ['merge_file', 'vocab_file']

    if args.eval_fp32:
        cp_args.fp16 = False
        cp_args.bf16 = False
        cp_args.params_dtype = torch.float32

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    megatron.global_vars._parse_args = lambda *_args, **kwarg: args
    megatron.global_vars._GLOBAL_ARGS = args

    initialize_megatron()
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    model_provider = importlib.import_module(
        f'pretrain_{args.model_name}',
    ).model_provider

    # print final arguments.
    _print_args(args)
    if not is_ds_cp:
        model = get_model(model_provider)[0]
        model.load_state_dict(state_dict['model'], strict=True)
    elif args.deepspeed:

        # Hack #3:
        # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
        # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
        # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
        # which does not contain these attention-specific keys.
        #
        # Deepspeed does however manage to load the model if we just turn off this sanity check.
        import deepspeed
        deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None


        cp_path = args.load
        args.load = None
        model, _, _ = setup_model_and_optimizer(model_provider)
        model = model[0]
        zero_enabled = model._config.zero_enabled
        model._config.zero_enabled = False
        _, _ = model.load_checkpoint(cp_path, tag = '.', load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        model._config.zero_enabled = zero_enabled
    else:
        model = get_model(model_provider)[0]
        # Initialize megatron model using the parsed state dict.
        sd = _create_rank_checkpoint(ds_checkpoint, None, mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(), True)

        model.load_state_dict(sd['model'], strict=True)

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model

def tasks_args(parser):

    results_path_default = f"results-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--model_name', type=str, default="gpt",
                       help=(
                           'Which model architecture to use (must exist as '
                           '`pretrain_{model_name}.py` script).'
                       ))
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = results_path_default, help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    group.add_argument('--intermed_results',  default = False, action='store_true', help='Whether to print & write intermediate results for each task')
    group.add_argument('--num_fewshot', type=int, default=0,
                       help='How many examples to show.')
    group.add_argument('--bootstrap_iters', type=int, default=100000, help='How many iterations to use for stderr estimation')
    group.add_argument('--micro_bs_multiplier', type=int, default=1, help='Increase the global batch size to remove bubble when pipeline parallel')
    group.add_argument('--prefix_lm', action='store_true',
                       help='Whether to adjust attention masks for a PrefixLM '
                       'decoder-only model.')
    group.add_argument('--prefix_tokens', type=str, nargs='*', default=[],
                       help='Tokens to add at the front of the input sequence.')
    # Automatically add UL2 tokens.
    group.add_argument('--_is_ul2', default=True, help=argparse.SUPPRESS)
    return parser

from megatron.global_vars import _parse_args

def main():
    # parse the megatron args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    args = _parse_args(tasks_args)
    load_path = args.load
    model = load_ds_checkpoint_and_setup_megatron(args)

    args = get_args()
    if args.deepspeed and args.adaptive_seq_len:
        # adaptive_seq_len hack #1:
        # CL automatically enables reset_activation_shape() which allows us to change input shapes
        # and it also reshapes the attenion scores in attention_mask_func
        args.curriculum_learning = 1

    task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
    task_dict = tasks.get_task_dict(task_list)

    if hasattr(model, 'module'):
        model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer)
    num_fewshot = args.num_fewshot

    if args.intermed_results:
        global_results = {"results": {}, "versions": {}}
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        iteration_id = load_path.split("/")[-1].replace("/", "")
        results_path = args.results_path.replace(".json", f"_lm-eval_{iteration_id}_{timestamp}.json")
        # Backup file in case of interruption during writing
        results_path_backup = args.results_path.replace(".json", f"_lm-eval_{iteration_id}_{timestamp}_backup.json")
        for task_name, task in task_dict.items():
            results = evaluator.evaluate(adaptor, {task_name: task}, False, num_fewshot, None, bootstrap_iters=args.bootstrap_iters)
            global_results["results"] = {**global_results["results"], **results["results"]}
            global_results["versions"] = {**global_results["versions"], **results["versions"]}
            if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                print(json.dumps(results, indent=2))
                with open(results_path, 'w') as outfile:
                    json.dump(global_results, outfile, indent=4)
                with open(results_path_backup, 'w') as outfile:
                    json.dump(global_results, outfile, indent=4)
    else:
        global_results = evaluator.evaluate(adaptor, task_dict, False, num_fewshot, None, bootstrap_iters=args.bootstrap_iters)
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            print(json.dumps(global_results, indent=2))
            with open(args.results_path, 'w') as outfile:
                json.dump(global_results, outfile, indent=4)

if __name__ == '__main__':
    main()
