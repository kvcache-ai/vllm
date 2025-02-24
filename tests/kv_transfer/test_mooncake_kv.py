# SPDX-License-Identifier: Apache-2.0

import os
import random

import torch
from tqdm import tqdm

from vllm.config import KVTransferConfig
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.kvstore_buffer import MooncakeStore


if __name__ == "__main__":

    vllm_config = VllmConfig()
    mooncake = MooncakeStore("localhost", 0, vllm_config)
    # mooncake.put("zzz", "yyy")
    print(mooncake.get("ppp"))

    print('Done')
