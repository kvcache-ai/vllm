# SPDX-License-Identifier: Apache-2.0

import os
import random

import torch
from tqdm import tqdm

from vllm.config import KVTransferConfig
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import MooncakeStore


if __name__ == "__main__":

    # my_rank = int(os.environ['RANK'])

    # print(f"initialized! My rank is {my_rank}")

    # config = KVTransferConfig(
    #     kv_connector='PyNcclConnector',
    #     kv_buffer_device='cuda',
    #     kv_buffer_size=1e9,
    #     kv_rank=my_rank,
    #     kv_role="kv_both",  # this arg doesn't matter in this test
    #     kv_parallel_size=2,
    #     kv_ip="127.0.0.1",
    #     kv_port=12345,
    # )
    vllm_config = VllmConfig()
    mooncake = MooncakeStore("localhost", 0, vllm_config)

    print('Done')
