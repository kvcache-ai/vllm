# SPDX-License-Identifier: Apache-2.0
import threading
from collections import deque
from typing import Deque, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import json
import torch
import pickle
import os

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)

class MooncakeStore(KVLookupBufferBase):
    def __init__(
        self,
        url: str,
        local_tp_rank: int,
        config: VllmConfig,
    ):
        """
        from distributed_object_store import DistributedObjectStore
        """
        try:
            import mooncake_vllm_adaptor as mva
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run vLLM with MooncakeConnector.") from e
        self.url = url
        self.local_tp_rank = local_tp_rank
        self.store = mva.MooncakeDistributedStore()  # 

        self.put_submit_thread: Optional[ThreadPoolExecutor] = None
        self.get_submit_thread: Optional[ThreadPoolExecutor] = None

        try:
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise
        
        self.store.setup(self.config.local_hostname, 
                           self.config.metadata_server, 
                           self.config.global_segment_size,
                           self.config.local_buffer_size, 
                           self.config.protocol, 
                           self.config.device_name,
                           self.config.master_server_address) 

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        # V1 (pack and put all tensors): insert the tensors into MooncakeStore's buffer
        raise NotImplementedError("Insert method is not implemented")

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:  
        # V1 (get and unpack all tensors): consume tensors from MooncakeStore's buffer
        raise NotImplementedError("Insert method is not implemented")
    
    def close(self):
        pass
  
    def put(
        self,
        key: str,
        value: Optional[torch.Tensor],
    ) -> None:
        # submit asynchronous put thread
        if self.put_submit_thread is None:
            self.put_submit_thread = ThreadPoolExecutor(max_workers=1)
        if value is not None:
            self.put_submit_thread.submit(self._put_impl, key, value)
      
    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        # submit asynchronous get thread
        if self.get_submit_thread is None:
            self.get_submit_thread = ThreadPoolExecutor(max_workers=1)
        value = self.get_submit_thread.submit(self._get_impl, key).result()
        return value

    def _put_impl(
        self,
        key: str,
        value: torch.Tensor,
    ) -> None:
        """Put KVCache to Mooncake Store"""
        value_bytes = pickle.dumps(value)
        try:
            self.store.put(key, value_bytes)
        except TypeError as e:
            raise TypeError("Mooncake Store Put Type Error.")
        
    def _get_impl(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Get KVCache from Mooncake Store"""
        try:
            data = self.store.get(key)
        except TypeError as e:
            raise TypeError("Mooncake Store Get Type Error.")
        if len(data):
            return pickle.loads(data)
        return None