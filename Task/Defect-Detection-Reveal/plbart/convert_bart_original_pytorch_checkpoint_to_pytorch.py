# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert BART checkpoint."""
import sys
sys.path.append("../../PLBART")

import argparse
import logging
import os
from pathlib import Path

import fairseq
import torch
from packaging import version
from fairseq import models
from fairseq.models.bart import BARTModel, mbart_base_architecture, BARTHubInterface

from transformers import (
    MBartConfig,
    MBartModel,
)
from source.sentence_prediction import BARTSentencePredictionTask



FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BARTModel}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = torch.nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight"
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    # hub_interface = torch.hub.load("pytorch/fairseq", "bart.base").eval()
    args = argparse.Namespace(activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, add_prev_output_tokens=True, all_gather_list_size=16384, arch='mbart_base', attention_dropout=0.1, batch_size=4, batch_size_valid=4, best_checkpoint_metric='accuracy', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', classification_head_name='sentence_classification_head', clip_norm=1.0, cpu=False, criterion='sentence_prediction', cross_self_attention=False, curriculum=0, data='/home/zzr/CodeStudy/Defect-detection/plbart/processed/data-bin', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=12, decoder_embed_dim=768, decoder_embed_path=None, decoder_ffn_embed_dim=3072, decoder_input_dim=768, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=768, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=0, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=12, encoder_embed_dim=768, encoder_embed_path=None, encoder_ffn_embed_dim=3072, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, end_learning_rate=0.0, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', init_token=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, langs='java,python,en_XX', layernorm_embedding=True, local_rank=0, localsgd_frequency=3, log_format='json', log_interval=10, lr=[5e-05], lr_scheduler='polynomial_decay', max_epoch=5, max_positions=512, max_source_positions=1024, max_target_positions=1024, max_tokens=2048, max_tokens_valid=2048, max_update=15000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_classes=2, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pooler_activation_fn='tanh', pooler_dropout=0.0, power=1.0, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, regression_target=False, relu_dropout=0.0, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=False, reset_meters=True, reset_optimizer=True, restore_file='/data2/cg/CodeStudy/PLBART/pretrain/checkpoint_11_100000.pt', save_dir='/home/zzr/CodeStudy/Defect-detection/plbart/devign', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1234, sentence_avg=False, separator_token=None, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=True, shorten_data_split_list='', shorten_method='truncate', skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='plbart_sentence_prediction', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_num_update=1000000, tpu=False, train_subset='train', update_freq=[4], use_bmuf=False, use_old_adam=False, user_dir='/home/zzr/CodeStudy/PLBART/source', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_updates=500, weight_decay=0.0, zero_sharding='none')
    # args = BARTModel.add_args()
    mbart_base_architecture(args)
    args.encoder_layers_to_keep = None
    args.decoder_layers_to_keep = None

    task = BARTSentencePredictionTask.setup_task(args)
    model = BARTModel.build_model(args, task).eval()

    hub_interface = BARTHubInterface(args, task, model)

    # vocab_size = sd["model"]["decoder.embed_tokens.weight"].shape[0]
    # d_modle = sd["model"]["decoder.embed_tokens.weight"].shape[1]
    # hub_interface.model.encoder.embed_tokens = torch.nn.Embedding(vocab_size, d_modle, padding_idx=1)
    # hub_interface.model.decoder.embed_tokens = torch.nn.Embedding(vocab_size, d_modle, padding_idx=1)
    # hub_interface.model.decoder.output_projection = torch.nn.Linear(in_features=d_modle, out_features=vocab_size, bias=False)
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path)

    bart.model.upgrade_state_dict(bart.model.state_dict())

    vocab_size=bart.model.encoder.embed_tokens.num_embeddings
    max_position_embeddings=bart.model.args.max_source_positions
    encoder_layers=bart.model.args.encoder_layers
    encoder_ffn_dim=bart.model.args.encoder_ffn_embed_dim
    encoder_attention_heads=bart.model.args.encoder_attention_heads
    decoder_layers=bart.model.args.decoder_layers
    decoder_ffn_dim=bart.model.args.decoder_ffn_embed_dim
    decoder_attention_heads=bart.model.args.decoder_attention_heads
    encoder_layerdrop=bart.model.args.encoder_layerdrop
    decoder_layerdrop=bart.model.args.decoder_layerdrop
    activation_function=bart.model.args.activation_fn
    d_model=bart.model.args.encoder_embed_dim
    dropout=bart.model.args.dropout
    attention_dropout=bart.model.args.attention_dropout
    activation_dropout=bart.model.args.activation_dropout

    config = MBartConfig(vocab_size=vocab_size,
                         max_position_embeddings=max_position_embeddings,
                         encoder_layers=encoder_layers,
                         encoder_ffn_dim=encoder_ffn_dim,
                         encoder_attention_heads=encoder_attention_heads,
                         decoder_layers=decoder_layers,
                         decoder_ffn_dim=decoder_ffn_dim,
                         decoder_attention_heads=decoder_attention_heads,
                         encoder_layerdrop=encoder_layerdrop,
                         decoder_layerdrop=decoder_layerdrop,
                         activation_function=activation_function,
                         d_model=d_model,
                         dropout=dropout,
                         attention_dropout=attention_dropout,
                         activation_dropout=activation_dropout)

    # tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    # tokens2 = MBartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    # assert torch.eq(tokens, tokens2).all()

    tokens = torch.LongTensor([[    0, 20920,   232,   328,   740,  1140, 12695,    69, 46078,  1588,2]])

    state_dict = bart.model.state_dict()
    remove_ignore_keys_(state_dict)
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    fairseq_output = bart.extract_features(tokens)

    model = MBartModel(config).eval()
    model.load_state_dict(state_dict)
    new_model_outputs = model(tokens)["last_hidden_state"]


    # Check results
    print(new_model_outputs)
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    args = parser.parse_args()
    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
