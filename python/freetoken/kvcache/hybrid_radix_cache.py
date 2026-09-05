"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``."""

    node: RadixTreeNode
    kv_indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free


class HybridRadixCache:
    def __init__(self, device: torch.device, page_size: int) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0
        # Uids of RUNNING requests with in-flight committed spans. A node whose
        # ``donor`` is in here is mid-flight: the donor's page_table row still
        # references its pages, so eviction may free the GDN slot (tombstone)
        # but NEVER the KV. Maintained by CacheManager (add at chunk commit,
        # discard at finish/abort via _free_req_slots).
        self.live_donors: set[int] = set()

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a LIVE snapshot (a continuation can only resume the GDN
        recurrence from a checkpointed boundary)."""
        import os as _os

        _dbg = _os.environ.get("FREETOKEN_RADIX_DEBUG", "0") == "1"
        node, _ = self._walk(input_ids)
        if _dbg:
            _pl = self._path_len(node)
            chain = []
            _cur = node
            while not _cur.is_root():
                chain.append(f"len={_cur.length} mv={'Y' if _cur.mamba_value is not None else 'N'} refs={_cur.ref_count}/{_cur.mamba_ref_count}")
                _cur = _cur.parent
            print(
                f"[walk] query={len(input_ids)} walked_to={_pl} "
                f"children_of_root={list(self.root.children.keys())[:4]} "
                f"chain(root->node): {list(reversed(chain))}",
                flush=True,
            )
        # walk up to the deepest node whose END boundary has a live snapshot
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.mamba_value is not None:
                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int, donor: int | None = None) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (matched_prefix_len, mamba_exist). If the boundary node already
        owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup). ``donor`` (running request uid) marks the committed span
        mid-flight: its KV is unfreeable until the donor leaves live_donors."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        import os as _os

        if _os.environ.get("FREETOKEN_RADIX_DEBUG", "1") == "1":
            print(
                f"[insert] insert_len={insert_len} walked_prefix={prefix_len} "
                f"node_children_after={list(node.children.keys())[:3]}",
                flush=True,
            )
        new_node_len = 0
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], kv_indices[prefix_len:].clone())
            new_node.set_parent(node)
            new_node.donor = donor
            self.full_evictable += new_node.length
            new_node_len = new_node.length
            node = new_node
        if node.is_root():
            return prefix_len, True   # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            return prefix_len, True                 # dedup: caller frees its donated slot
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        node.donor = donor                          # newest donor governs the span
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        import os as _os

        if _os.environ.get("FREETOKEN_RADIX_DEBUG", "0") == "2":
            real = sum(n.length for n in self._leaves() if n.ref_count == 0)
            if real != self.full_evictable:
                print(
                    f"[ins-drift] new_span={new_node_len} counter={self.full_evictable} "
                    f"actual={real} prefix_len={prefix_len} insert_len={insert_len}",
                    flush=True,
                )
                self.full_evictable = real
        return prefix_len, False

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba, freed = [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            if node.donor in self.live_donors:
                continue  # mid-flight donor's committed KV: its row still points here
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            self._free_node_mamba(node, mamba)
            parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    def evict_mamba(self, num: int) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang)."""
        cands = [n for n in self._snapshot_nodes() if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        kv, mamba, freed = [], [], 0
        while freed < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if (node.is_leaf() and node.ref_count == 0
                    and node.donor not in self.live_donors):
                kv.append(node.value)
                self.full_evictable -= node.length
                self._free_node_mamba(node, mamba)
                freed += 1
                self._cascade_tombstone_leaves(self._unlink(node), kv)
            else:
                # Internal / locked-KV / donor-live leaf: free the SLOT only.
                # A donor-live leaf's KV pages are still referenced by the
                # running request's page_table row (no commit locks by design,
                # 4da3ac3) -- freeing them returned live ids to the allocator
                # (row aliasing; the [audit] dead-run double-frees).
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
        import os as _os

        if _os.environ.get("FREETOKEN_RADIX_DEBUG", "0") == "2":
            import torch as _t

            real_fe = sum(n.length for n in self._leaves() if n.ref_count == 0)
            real_mv = len(self._snapshot_nodes())
            if real_fe != self.full_evictable or real_mv != self.mamba_evictable:
                print(
                    f"[evict-drift] full_evictable counter={self.full_evictable} "
                    f"actual={real_fe} | mamba_evictable counter={self.mamba_evictable} "
                    f"actual={real_mv} | freed={freed} kv={len(kv)} mamba={len(mamba)}",
                    flush=True,
                )
                self.full_evictable = real_fe
                self.mamba_evictable = real_mv
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    @property
    def full_evictable_size(self) -> int:
        self._recount()
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        self._recount()
        return self.mamba_evictable

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo

        self._recount()
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0

    # ---------------------------------------------------------------- helpers
    def _recount(self) -> None:
        """Exact recompute of the four accounting scalars from the tree.

        The incremental maintenance drifted through the dedup/split/evict
        interplay (COW-shared spans double-counted; mixed span/node-length
        decompositions) and tripped check_integrity at idle, killing workers.
        The tree is small (per-token nodes, <= a few tens of thousands) and
        recount runs only on status/integrity reads -- O(nodes) is negligible.
        """
        fe = fp = me = mp = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.is_root():
                stack.extend(n.children.values())
                continue
            if n.ref_count > 0:
                fp += n.length
            else:
                fe += n.length
            if n.mamba_value is not None:
                if n.mamba_ref_count > 0:
                    mp += 1
                else:
                    me += 1
            stack.extend(n.children.values())
        self.full_evictable, self.full_protected = fe, fp
        self.mamba_evictable, self.mamba_protected = me, mp

    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()
               and parent.donor not in self.live_donors):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            parent = self._unlink(parent)
        return parent, freed

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

    def _leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns()
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle"]
