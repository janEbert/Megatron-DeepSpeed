# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UL2-style dataset."""

import numpy as np

from megatron import get_tokenizer
from megatron.data.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
    SamplingStyle
)
from megatron.data.t5_dataset import pad_and_convert_to_numpy, T5Dataset


class UL2Dataset(T5Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, denoiser_ratios,
                 denoisers, mean_span_lengths, mask_ratios,
                 denoiser_tokens, max_seq_length, max_seq_length_dec,
                 short_seq_prob, seed):

        if denoiser_ratios is None:
            # Uniform distribution by default.
            denoiser_ratios = [1 / len(denoisers)] * len(denoisers)

        assert (
            len(denoiser_ratios) == len(denoisers)
            == len(mean_span_lengths) == len(mask_ratios)
        ), (
            'some UL2 configurations do not correspond to the amount of '
            'denoising objectives'
        )

        super().__init__(name, indexed_dataset, data_prefix,
                         num_epochs, max_num_samples, None,
                         max_seq_length, max_seq_length_dec,
                         short_seq_prob, seed)

        # Params to store.
        self.denoiser_ratios = [
            denoiser_ratio / sum(denoiser_ratios)
            for denoiser_ratio in denoiser_ratios
        ]
        self.denoisers = [denoiser.upper() for denoiser in denoisers]
        self.mean_span_lengths = mean_span_lengths
        self.mask_ratios = mask_ratios

        # Vocab stuff.
        tokenizer = get_tokenizer()
        # Remove CLS token because we don't need it.
        del self.cls_id
        self.cls_ids = {
            denoiser: tokenizer.vocab[token]
            for (denoiser, token) in denoiser_tokens.items()
        }
        # cls_token = self.vocab_id_to_token_dict[tokenizer.cls]
        # if cls_token not in self.cls_ids:
        #     self.cls_ids[cls_token] = tokenizer.cls

        # Filter out denoiser tokens.
        self.sentinel_tokens = [
            token
            for token in tokenizer.additional_special_tokens_ids
            if token not in self.cls_ids.values()
        ]
        assert len(self.sentinel_tokens) > 0, \
            "Provide the argument --vocab-extra-ids 100 to the script"

    def __getitem__(self, idx):

        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_training_sample(sample, seq_length,
                                     self.max_seq_length,  # needed for padding
                                     self.max_seq_length_dec,
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_ids, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.denoiser_ratios, self.denoisers,
                                     self.mean_span_lengths, self.mask_ratios,
                                     np_rng,
                                     self.bos_id, self.eos_id,
                                     self.sentinel_tokens)


def build_training_sample(sample, target_seq_length,
                          max_seq_length, max_seq_length_dec,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_ids, sep_id, mask_id, pad_id,
                          denoiser_ratios, denoisers,
                          mean_span_lengths, mask_ratios,
                          np_rng, bos_id=None,
                          eos_id=None, sentinel_tokens=None):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_ids: Start of example ids.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        denoiser_ratios: Probability of each denoising objective to be selected.
        denoisers: What type of UL2 denoising objective the other UL2
              configurations refer to.
        mean_span_lengths: Mean length for sampling span lengths. Numbers < 1
              indicate a mean length of the sequence length times that number.
        mask_ratios: Ratio of masked token in the full sequence.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        sentinel_tokens: unique value to be substituted for every replaced span
    """

    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Denoiser selection
    denoiser_index = np_rng.choice(np.arange(len(denoisers)), p=denoiser_ratios)
    denoiser = denoisers[denoiser_index]
    masked_lm_prob = mask_ratios[denoiser_index]
    mean_ngrams = mean_span_lengths[denoiser_index]
    if mean_ngrams < 1:
        mean_ngrams = round(len(tokens) * mean_ngrams)
    max_ngrams = mean_ngrams * 2 - 1

    # Prepend objective token.
    cls_id = cls_ids.get(denoiser)
    if cls_id is None:
        raise ValueError('unknown denoiser')
    tokens = [cls_id] + tokens

    # Masking.
    max_predictions_per_seq = masked_lm_prob * len(tokens)
    if denoiser == 'R' or denoiser == 'X':
        sampling_style = SamplingStyle.NORMAL
        prefix_lm = False
    elif denoiser == 'S':
        sampling_style = SamplingStyle.UNIFORM
        prefix_lm = True
    else:
        raise ValueError('unknown denoiser')
    (
        tokens, masked_positions, masked_labels, _, masked_spans,
    ) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
        max_ngrams=max_ngrams, masking_style="t5",
        sampling_style=sampling_style, prefix_lm=prefix_lm,
    )

    # Padding.
    tokens_enc, tokens_dec_in, labels, enc_mask, \
    dec_mask, enc_dec_mask, loss_mask \
        = pad_and_convert_to_numpy(tokens, masked_positions,
                                   masked_labels, pad_id, max_seq_length,
                                   max_seq_length_dec, masked_spans,
                                   bos_id, eos_id, sentinel_tokens)

    train_sample = {
        'text_enc': tokens_enc,
        'text_dec': tokens_dec_in,
        'labels': labels,
        'loss_mask': loss_mask,
        'truncated': int(truncated),
        'enc_mask': enc_mask,
        'dec_mask': dec_mask,
        'enc_dec_mask': enc_dec_mask,
    }
    return train_sample
