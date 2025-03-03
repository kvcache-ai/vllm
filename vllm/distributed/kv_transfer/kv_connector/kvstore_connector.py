# SPDX-License-Identifier: Apache-2.0
"""
KVStore Based Connector for Distributed Machine Learning Inference
The KVStoreConnector transfers KV caches between prefill vLLM workers (KV cache
producer) and decode vLLM workers (KV cache consumer) using a database-style
KVStore such as MooncakeStore, Redis, and Valkey.
"""
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class KVStoreConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size

        self.local_tp_rank = local_rank
        self.kv_store = None

        # Init kv_store
        if self.config.kv_connector == "MooncakeStoreConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_store = os.getenv('MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_store:
                raise ValueError(
                    "To use MooncakeStoreConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_lookup_buffer.mooncake_store import (  # noqa: E501
                    MooncakeStore)
                logger.info(
                    "Initializing KVStoreConnector under kv_transfer_config %s",
                    self.config)
                self.kv_store = MooncakeStore(config)
        else:
            logger.error("Can not find %s", self.config.kv_connector)

    def close(self) -> None:
        """Close the buffer and release resources.
        This method is responsible for cleaning up resources related to the 
        connector when it is no longer needed.
        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        self.kv_store.close()

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # get info from kv_transfer_params
        store_keys_list = model_input.kv_transfer_params.kvcache_store_keys
        prefix_ids_list = model_input.kv_transfer_params.prefix_prompt_ids

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)

            # get store keys prefix for current seq
            store_keys_prefix = store_keys_list[idx][0]

            # get roi for current seq
            prefix_ids = torch.tensor(prefix_ids_list[idx], dtype=int)
            roi = torch.ones_like(prefix_ids, dtype=bool)

            self.kv_store.insert(
                current_tokens, roi, keys, values,
                hidden_or_intermediate_states[start_pos:end_pos],
                store_keys_prefix, self.local_tp_rank)

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        bypass_model_exec = True
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_or_intermediate_states_for_one_req = []

        # get info from kv_transfer_params
        load_keys_list = model_input.kv_transfer_params.kvcache_load_keys
        prefix_ids_list = model_input.kv_transfer_params.prefix_prompt_ids

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # get load keys prefix for current seq
            load_keys_prefix = load_keys_list[idx][0]

            # get roi for current seq
            prefix_ids = torch.tensor(prefix_ids_list[idx], dtype=int)
            roi = torch.ones_like(prefix_ids, dtype=bool)

            ret = self.kv_store.drop_select(current_tokens, roi,
                                            load_keys_prefix,
                                            self.local_tp_rank)

            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                continue

            remote_kv, hidden = ret[0], ret[1]
            num_computed_tokens = roi.shape[0]

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # call self.kv_store to get kv layer by layer
            for layer_id in range(start_layer, end_layer):
                layer = model_executable.model.layers[layer_id]
                # get kvcache object
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                # get remote kvcache

                remote_k, remote_v = remote_kv[0][layer_id], remote_kv[1][
                    layer_id]
                # use ops.reshape_and_cache_flash to put kv into kvcache
                ops.reshape_and_cache_flash(
                    remote_k.to(key_cache.device),
                    remote_v.to(value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input
