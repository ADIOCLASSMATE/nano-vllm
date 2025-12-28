from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # Stack/Queue of free blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def num_free_blocks(self):
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self):
        return len(self.used_block_ids)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        if block_id in self.free_block_ids:
            self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        # Safety check handled by caller usually, but good to have
        if block.ref_count == 0:
            if block_id in self.used_block_ids:
                self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)

    # ------------------------------------------------------------------
    # Common Methods (Prefill / Cleanup)
    # ------------------------------------------------------------------

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we have enough blocks for the initial prompt/prefill."""
        # Note: seq.num_blocks calculation depends on seq implementation logic
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate blocks for the initial prompt (Prefix Caching logic)."""
        assert not seq.block_table
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # Only compute hash if the block is full
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            block_id = self.hash_to_block_id.get(h, -1)
            
            # Check for collision or if block content matches
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # Assign a new free block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache hit
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Rare case: block in hash map but technically free (shouldn't happen with proper GC)
                    block = self._allocate_block(block_id)
            
            # Update block metadata
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Release all blocks held by a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # ------------------------------------------------------------------
    # Diffusion Specific Methods (Batch Allocation)
    # ------------------------------------------------------------------

    def can_append_blocks(self, num_blocks: int) -> bool:
        """Check if we can allocate N blocks at once (for Diffusion)."""
        return len(self.free_block_ids) >= num_blocks

    def append_blocks(self, seq: Sequence, num_blocks: int):
        """Allocate N fresh blocks for Diffusion generation."""
        for _ in range(num_blocks):
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            assert block.ref_count == 0, f"Block {block_id} has ref_count {block.ref_count}"
            block.reset()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

    # ------------------------------------------------------------------
    # AR Specific Methods (Incremental Allocation)
    # ------------------------------------------------------------------

    def can_append(self, seq: Sequence) -> bool:
        """Check if the sequence needs a NEW block for the next token."""
        # If len % block_size == 1, it means we just filled the previous block 
        # (assuming 0-indexed check logic from original code) or started a new one.
        # Original AR logic: we need a free block if the current length spills over.
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Handle block boundary crossing for AR generation."""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        # Case 1: Just started a new block territory
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1 # Previous block must be finalized/hashed
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        # Case 2: Just finished filling a block -> Compute its hash
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1 # Currently filling this block
            token_ids = seq.block(seq.num_blocks - 1)
            
            # Compute hash with prefix (previous block's hash)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
            
        # Case 3: In the middle of a block -> Do nothing
        else:
            assert last_block.hash == -1