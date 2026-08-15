from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pprint import pprint, pformat


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    hidden_size: int = 128
    num_heads: int = 4
    head_dim: int = 32  # hidden_size = num_heads * head_dim
    num_experts: int = 8
    top_k: int = 2
    moe_inter: int = 256
    max_seq_len: int = 64  # KV cache length per request


@dataclass
class ForwardBatch:
    batch_size: int
    input_ids: torch.Tensor  # [bs]   (decode: 1 token per request)
    positions: torch.Tensor  # [bs]
    seq_lens: torch.Tensor  # [bs]   current length of each request
    req_pool_indices: torch.Tensor  # [bs]   slot in the KV cache pool
    out_cache_loc: torch.Tensor  # [bs]   where to write new K/V (= seq_len-1)


class MiniDecoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_bs: int, device):
        super().__init__()
        self.cfg = cfg
        H, nh, hd = cfg.hidden_size, cfg.num_heads, cfg.head_dim

        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.qkv = nn.Linear(H, 3 * H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)

        # MoE
        self.router = nn.Linear(H, cfg.num_experts, bias=False)
        self.w_in = nn.Parameter(torch.randn(cfg.num_experts, H, cfg.moe_inter) * 0.02)
        self.w_out = nn.Parameter(torch.randn(cfg.num_experts, cfg.moe_inter, H) * 0.02)

        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)

        # Static KV cache pool: [max_bs (req slots), max_seq, num_heads, head_dim]
        self.register_buffer(
            "k_cache", torch.zeros(max_bs, cfg.max_seq_len, nh, hd, device=device)
        )
        self.register_buffer(
            "v_cache", torch.zeros(max_bs, cfg.max_seq_len, nh, hd, device=device)
        )
        self.to(device)

    def _attention(self, h, fb: ForwardBatch):
        cfg = self.cfg
        nh, hd = cfg.num_heads, cfg.head_dim
        bs = h.shape[0]

        qkv = self.qkv(h).view(bs, 3, nh, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [bs, nh, hd]

        # Write the new K/V into the cache at out_cache_loc for each request.
        # Static-shape scatter: indices come from a buffer, safe inside a graph.
        slot = fb.req_pool_indices  # [bs]
        pos = fb.out_cache_loc  # [bs]  (= seq_len - 1)
        self.k_cache[slot, pos] = k
        self.v_cache[slot, pos] = v

        # Gather this batch's cache rows: [bs, max_seq, nh, hd]
        kc = self.k_cache[slot]
        vc = self.v_cache[slot]

        # scores: [bs, nh, max_seq]
        scores = torch.einsum("bhd,bshd->bhs", q, kc) / (hd**0.5)

        # Mask out positions >= seq_len for each request (static shape, dynamic
        # mask computed from the seq_lens buffer -> graph friendly).
        ar = torch.arange(cfg.max_seq_len, device=h.device).view(1, 1, -1)
        valid = ar < fb.seq_lens.view(bs, 1, 1)
        scores = scores.masked_fill(~valid, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhs,bshd->bhd", attn, vc).reshape(bs, nh * hd)
        return self.o_proj(out)

    def _moe(self, h):
        cfg = self.cfg
        logits = self.router(h)  # [bs, E]
        weights, idx = torch.topk(logits, cfg.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)  # [bs, k]

        # Dense compute (all experts) then select -- the naive, graph-safe way.
        # [E, bs, moe_inter] -> act -> [E, bs, H]
        x = torch.einsum("bh,ehi->ebi", h, self.w_in)
        x = F.gelu(x)
        x = torch.einsum("ebi,eih->ebh", x, self.w_out)  # [E, bs, H]

        out = torch.zeros_like(h)
        for j in range(cfg.top_k):
            sel = idx[:, j]  # [bs]
            chosen = x[sel, torch.arange(h.shape[0], device=h.device)]  # [bs, H]
            out = out + weights[:, j : j + 1] * chosen
        return out

    def forward(self, input_ids, positions, fb: ForwardBatch):
        h = self.embed(input_ids)
        h = h + self._attention(self.norm1(h), fb)
        h = h + self._moe(self.norm2(h))
        return self.lm_head(h)  # [bs, vocab]

@dataclass
class DecodeInputBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, device, seq_len_fill: int = 1):
        return cls(
            input_ids=torch.zeros(max_bs, dtype=torch.int64, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int64, device=device),
            seq_lens=torch.full((max_bs,), seq_len_fill, dtype=torch.int64, device=device),
            req_pool_indices=torch.zeros(max_bs, dtype=torch.int64, device=device),
            out_cache_loc=torch.zeros(max_bs, dtype=torch.int64, device=device),
        )

    def populate_from_forward_batch(self, fb: ForwardBatch, raw_bs: int, bs: int,
                                    seq_len_fill: int):
        """Copy the real (raw_bs) inputs into the static buffers.

        When the captured batch size `bs` is larger than the real `raw_bs`,
        the padded tail must hold *safe* values: seq_lens=fill (so attention
        masks reduce to a valid 1-length window), out_cache_loc=0 (write into
        a sentinel slot), so padded requests can't corrupt real KV state.
        """
        if bs != raw_bs:
            self.seq_lens.fill_(seq_len_fill)
            self.out_cache_loc.zero_()
            self.req_pool_indices.zero_()

        self.input_ids[:raw_bs].copy_(fb.input_ids)
        self.positions[:raw_bs].copy_(fb.positions)
        self.seq_lens[:raw_bs].copy_(fb.seq_lens)
        self.req_pool_indices[:raw_bs].copy_(fb.req_pool_indices)
        self.out_cache_loc[:raw_bs].copy_(fb.out_cache_loc)


def get_batch_sizes_to_capture(max_num_requests: int) -> List[int]:
    candidate = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    capture_bs = sorted({bs for bs in candidate if bs <= max_num_requests})
    if max_num_requests not in capture_bs:
        capture_bs.append(max_num_requests)
    capture_bs = sorted(set(capture_bs))
    assert capture_bs and capture_bs[0] > 0
    return capture_bs


class CudaGraphRunner:
    def __init__(self, model: MiniDecoderModel, max_num_requests: int, device):
        self.model = model
        self.device = device
        self.use_cuda_graph = device.type == "cuda"

        self.capture_bs = get_batch_sizes_to_capture(max_num_requests)
        self.max_bs = max(self.capture_bs)
        self.seq_len_fill = 1

        # ONE set of static buffers shared by every graph.
        self.buffers = DecodeInputBuffers.create(self.max_bs, device, self.seq_len_fill)

        # bs -> captured graph, bs -> static output tensor
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.output_buffers: Dict[int, torch.Tensor] = {}
        self.graph_memory_pool = None  # shared across all graphs

        self.capture()

    # --- capture ---------------------------------------------------------- #

    def _make_forward_batch(self, bs: int) -> ForwardBatch:
        b = self.buffers
        return ForwardBatch(
            batch_size=bs,
            input_ids=b.input_ids[:bs],
            positions=b.positions[:bs],
            seq_lens=b.seq_lens[:bs],
            req_pool_indices=b.req_pool_indices[:bs],
            out_cache_loc=b.out_cache_loc[:bs],
        )

    def capture(self):
        if not self.use_cuda_graph:
            logger.debug("[runner] CUDA unavailable -> eager fallback (no real graphs).")
            return

        # Largest first so smaller graphs reuse the big memory pool.
        for bs in tqdm(reversed(self.capture_bs), desc="Capturing"):
            graph, out = self.capture_one_batch_size(bs)
            self.graphs[bs] = graph
            self.output_buffers[bs] = out
        logger.debug(f"[runner] captured {len(self.graphs)} graphs for bs={self.capture_bs}")

    def capture_one_batch_size(self, bs: int):
        fb = self._make_forward_batch(bs)
        ids, pos = self.buffers.input_ids[:bs], self.buffers.positions[:bs]

        def run_once():
            return self.model(ids, pos, fb)

        # Warm up a couple of times on a side stream before capture, so cuBLAS
        # workspaces / lazy inits happen outside the graph (sglang does this).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_once()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        # All graphs share one memory pool -> smaller graphs reuse the memory
        # already reserved by the largest one.
        with torch.cuda.graph(graph, pool=self.graph_memory_pool):
            static_out = run_once()
        if self.graph_memory_pool is None:
            self.graph_memory_pool = graph.pool()
        return graph, static_out

    # --- replay ----------------------------------------------------------- #

    def can_run(self, fb: ForwardBatch) -> bool:
        return fb.batch_size <= self.max_bs

    def replay_prepare(self, fb: ForwardBatch) -> int:
        raw_bs = fb.batch_size
        # Pad up to the nearest captured batch size.
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        self.buffers.populate_from_forward_batch(fb, raw_bs, bs, self.seq_len_fill)
        return bs

    @torch.no_grad()
    def replay(self, fb: ForwardBatch) -> torch.Tensor:
        assert self.can_run(fb)
        raw_bs = fb.batch_size
        bs = self.replay_prepare(fb)

        if self.use_cuda_graph:
            self.graphs[bs].replay()
            out = self.output_buffers[bs]
        else:
            # Eager fallback: run the same static buffers through the model.
            out = self.model(
                self.buffers.input_ids[:bs],
                self.buffers.positions[:bs],
                self._make_forward_batch(bs),
            )
        # Slice padded rows back off -> return only the real requests.
        return out[:raw_bs]


# --------------------------------------------------------------------------- #
# Demo / smoke test
# --------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    max_requests = 32

    model = MiniDecoderModel(cfg, max_bs=max_requests, device=device).eval()
    runner = CudaGraphRunner(model, max_num_requests=max_requests, device=device)

    # Simulate several decode steps with different (unpadded) batch sizes.
    for step, raw_bs in enumerate([5, 1, 13, 32, 7]):
        seq_lens = torch.randint(1, cfg.max_seq_len, (raw_bs,), device=device)
        fb = ForwardBatch(
            batch_size=raw_bs,
            input_ids=torch.randint(0, cfg.vocab_size, (raw_bs,), device=device),
            positions=seq_lens - 1,
            seq_lens=seq_lens,
            req_pool_indices=torch.arange(raw_bs, device=device),
            out_cache_loc=seq_lens - 1,
        )
        logits = runner.replay(fb)
        next_tok = logits.argmax(-1)
        padded_to = runner.capture_bs[bisect.bisect_left(runner.capture_bs, raw_bs)]
        logger.debug(
            f"step {step}: raw_bs={raw_bs:3d} -> padded_bs={padded_to:3d} "
            f"logits={tuple(logits.shape)} next_tokens={next_tok.tolist()[:8]}"
        )

    logger.critical(f"runner.graphs\n:{pformat(runner.graphs)}")


if __name__ == "__main__":
    main()