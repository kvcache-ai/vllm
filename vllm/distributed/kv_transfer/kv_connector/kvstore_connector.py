from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import MooncakeStore

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
        self.message_queue
        
        # Init kv_store
        if self.config.kv_connector == "MooncakeStoreConnector":
            self.kv_store = MooncakeStore(self.config.store_url, local_rank, config)
            logger.info("Initializing KVStoreConnector")
        else:
            logger.error("Can not find %s", self.config.kv_connector)

  
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
        
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            # get prefix name
            store_keys_prefix = store_keys_list[idx]
            
            # call self.kv_store to put kv layer by layer
            for layer_id in range(start_layer, end_layer):
                # get kvcache
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]
                kcache = key_cache[current_slot_mapping].unsqueeze(0)
                vcache = value_cache[current_slot_mapping].unsqueeze(0)
                
                kvcache_to_sent = torch.stack((kcache, vcache), dim=0)
                store_keys = store_keys_prefix + "_" + str(layer_id) + "_" + str(self.local_tp_rank)
                self.kv_store.put(store_keys, kvcache_to_sent)
            
            # call self.kv_store to put hidden_or_intermediate_states
            tmp_keys = store_keys_prefix + "_" + "hidden" + "_" + str(self.local_tp_rank)
            self.kv_store.put(tmp_keys, hidden_or_intermediate_states[start_pos:end_pos])

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())



    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        # maybe implement like this
        bypass_model_exec = True
        
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_or_intermediate_states_for_one_req = []

        # get info from kv_transfer_params
        load_keys_list = model_input.kv_transfer_params.kvcache_load_keys

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            # get prefix name
            load_keys_prefix = load_keys_list[idx]
            
            # call self.kv_store to get kv layer by layer
            for layer_id in range(start_layer, end_layer):
                layer = model_executable.model.layers[i]
                # get kvcache object
                kv_cache = kv_caches[layer_id - start_layer]
                kcache, vcache = kv_cache[0], kv_cache[1]
                # get remote kvcache
                load_keys = load_keys_prefix + "_" + str(layer_id) + "_" + str(self.local_tp_rank)
                remote_kv = self.kv_store.get(load_keys)
                remote_k, remote_v = remote_kv[0], remote_kv[1]
                # use ops.reshape_and_cache_flash to put kv into kvcache
                ops.reshape_and_cache_flash(
                    remote_k.to(
                        key_cache.device),
                    remote_v.to(
                        value_cache.device),
                    kcache,
                    vcache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )
              
                
            hidden_or_intermediate_states_for_one_req
            # call self.kv_store to put hidden_or_intermediate_states
            tmp_keys = load_keys_prefix + "_" + "hidden" + "_" + str(self.local_tp_rank)
            hidden_or_intermediate_states_for_one_req.append(self.kv_store.get(tmp_keys))
        
        if not bypass_model_exec:
            logger.debug(
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


