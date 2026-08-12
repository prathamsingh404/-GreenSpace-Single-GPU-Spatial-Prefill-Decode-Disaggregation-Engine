"""
Zero-Copy PagedAttention Block Manager.
Tracks physical HBM block allocations and enables pointer flips between Prefill and Decode.
"""

import threading
from typing import List, Optional, Dict, Tuple
from greenspace.core.config import CacheConfig
from greenspace.core.request import Request


class BlockManager:
    """
    Manages the lifecycle and allocation of physical KV cache blocks in GPU HBM.
    Thread-safe to support concurrent Prefill and Decode engine threads.
    """
    def __init__(self, cache_config: CacheConfig):
        self.block_size = cache_config.block_size
        self.num_total_blocks = cache_config.num_gpu_blocks
        self._lock = threading.Lock()
        
        # Free block pool
        self._free_blocks: List[int] = list(range(self.num_total_blocks))
        # Request -> list of allocated physical block indices
        self._block_tables: Dict[str, List[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """Checks if enough free blocks exist to store num_tokens."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        with self._lock:
            return len(self._free_blocks) >= blocks_needed

    def allocate_for_prompt(self, request: Request) -> List[int]:
        """
        Allocates physical HBM blocks for a new prompt request during the Prefill phase.
        Assigns allocated block IDs directly to request.allocated_blocks.
        """
        blocks_needed = (request.num_prompt_tokens + self.block_size - 1) // self.block_size
        with self._lock:
            if len(self._free_blocks) < blocks_needed:
                raise RuntimeError(
                    f"Out of GPU HBM KV blocks: requested {blocks_needed}, free {len(self._free_blocks)}"
                )
            
            allocated = [self._free_blocks.pop() for _ in range(blocks_needed)]
            self._block_tables[request.request_id] = allocated
            request.allocated_blocks = allocated
            return allocated

    def append_slot_for_decode(self, request: Request) -> Tuple[int, int]:
        """
        Called by Decode Thread when generating a token.
        Returns (block_id, slot_idx) where the new token's KV must be written.
        Allocates a new block if the current block is full.
        """
        with self._lock:
            total_tokens = request.total_tokens  # prompt + already generated
            slot_idx = total_tokens % self.block_size
            
            # If starting a new block
            if slot_idx == 0:
                if not self._free_blocks:
                    raise RuntimeError(f"Out of KV cache blocks during decode for request {request.request_id}")
                new_block = self._free_blocks.pop()
                request.allocated_blocks.append(new_block)
                self._block_tables[request.request_id] = request.allocated_blocks

            current_block = request.allocated_blocks[-1]
            return current_block, slot_idx

    def free(self, request: Request):
        """Releases all physical blocks allocated to the request back to the free pool."""
        with self._lock:
            allocated = self._block_tables.pop(request.request_id, None)
            if allocated:
                self._free_blocks.extend(allocated)
                request.allocated_blocks.clear()

    def get_block_table(self, request_id: str) -> List[int]:
        with self._lock:
            return list(self._block_tables.get(request_id, []))
