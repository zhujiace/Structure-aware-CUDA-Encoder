from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


GENERATED_PREFIX_MARKER = "\n<<<SCEM_GENERATED_PREFIX>>>\n"


def stable_hash_id(value: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    digest = hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (vocab_size - 1) + 1


@dataclass
class CudaASTNode:
    """One node from a native tree-sitter-cuda AST."""

    id: int
    type: str
    named: bool
    start_byte: int
    end_byte: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    has_error: bool
    is_missing: bool
    text: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CudaASTEdge:
    """A typed relation between two AST nodes."""

    source: int
    target: int
    type: str
    field_name: Optional[str] = None
    child_index: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class CudaASTSnapshot:
    """Full parser-native AST graph for a generated CUDA prefix."""

    provider: str
    incremental_reused: bool
    source_bytes: int
    root_id: int
    root_type: str
    root_has_error: bool
    nodes: List[CudaASTNode]
    edges: List[CudaASTEdge]
    node_type_counts: Dict[str, int]
    edge_type_counts: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "incremental_reused": self.incremental_reused,
            "source_bytes": self.source_bytes,
            "root_id": self.root_id,
            "root_type": self.root_type,
            "root_has_error": self.root_has_error,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_type_counts": self.node_type_counts,
            "edge_type_counts": self.edge_type_counts,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "incremental_reused": self.incremental_reused,
            "source_bytes": self.source_bytes,
            "root_id": self.root_id,
            "root_type": self.root_type,
            "root_has_error": self.root_has_error,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_type_counts": self.node_type_counts,
            "edge_type_counts": self.edge_type_counts,
        }


@dataclass
class CudaASTGraphBatch:
    """Padded AST graph tensors consumed directly by SCEM."""

    node_type_ids: torch.LongTensor
    node_text_ids: torch.LongTensor
    node_depths: torch.LongTensor
    node_child_indices: torch.LongTensor
    node_flags: torch.Tensor
    node_positions: torch.Tensor
    node_mask: torch.BoolTensor
    edge_sources: torch.LongTensor
    edge_targets: torch.LongTensor
    edge_type_ids: torch.LongTensor
    edge_mask: torch.BoolTensor

    def to(self, device: torch.device | str) -> "CudaASTGraphBatch":
        return CudaASTGraphBatch(
            node_type_ids=self.node_type_ids.to(device),
            node_text_ids=self.node_text_ids.to(device),
            node_depths=self.node_depths.to(device),
            node_child_indices=self.node_child_indices.to(device),
            node_flags=self.node_flags.to(device),
            node_positions=self.node_positions.to(device),
            node_mask=self.node_mask.to(device),
            edge_sources=self.edge_sources.to(device),
            edge_targets=self.edge_targets.to(device),
            edge_type_ids=self.edge_type_ids.to(device),
            edge_mask=self.edge_mask.to(device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.node_type_ids.shape[0])

    def zero_like(self) -> "CudaASTGraphBatch":
        return CudaASTGraphBatch(
            node_type_ids=torch.zeros_like(self.node_type_ids),
            node_text_ids=torch.zeros_like(self.node_text_ids),
            node_depths=torch.zeros_like(self.node_depths),
            node_child_indices=torch.zeros_like(self.node_child_indices),
            node_flags=torch.zeros_like(self.node_flags),
            node_positions=torch.zeros_like(self.node_positions),
            node_mask=self.node_mask.clone(),
            edge_sources=torch.zeros_like(self.edge_sources),
            edge_targets=torch.zeros_like(self.edge_targets),
            edge_type_ids=torch.zeros_like(self.edge_type_ids),
            edge_mask=torch.zeros_like(self.edge_mask),
        )

    def roll(self, shifts: int, dims: int = 0) -> "CudaASTGraphBatch":
        return CudaASTGraphBatch(
            node_type_ids=self.node_type_ids.roll(shifts=shifts, dims=dims),
            node_text_ids=self.node_text_ids.roll(shifts=shifts, dims=dims),
            node_depths=self.node_depths.roll(shifts=shifts, dims=dims),
            node_child_indices=self.node_child_indices.roll(shifts=shifts, dims=dims),
            node_flags=self.node_flags.roll(shifts=shifts, dims=dims),
            node_positions=self.node_positions.roll(shifts=shifts, dims=dims),
            node_mask=self.node_mask.roll(shifts=shifts, dims=dims),
            edge_sources=self.edge_sources.roll(shifts=shifts, dims=dims),
            edge_targets=self.edge_targets.roll(shifts=shifts, dims=dims),
            edge_type_ids=self.edge_type_ids.roll(shifts=shifts, dims=dims),
            edge_mask=self.edge_mask.roll(shifts=shifts, dims=dims),
        )

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Sequence[CudaASTSnapshot],
        *,
        max_nodes: int,
        max_edges: int,
        node_type_vocab_size: int,
        edge_type_vocab_size: int,
        text_vocab_size: int,
        max_depth: int,
        max_child_index: int,
        node_flag_dim: int = 6,
        node_position_dim: int = 5,
        device: torch.device | str | None = None,
    ) -> "CudaASTGraphBatch":
        batch_size = len(snapshots)
        node_type_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
        node_text_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
        node_depths = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
        node_child_indices = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
        node_flags = torch.zeros((batch_size, max_nodes, node_flag_dim), dtype=torch.float32, device=device)
        node_positions = torch.zeros((batch_size, max_nodes, node_position_dim), dtype=torch.float32, device=device)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
        edge_sources = torch.zeros((batch_size, max_edges), dtype=torch.long, device=device)
        edge_targets = torch.zeros((batch_size, max_edges), dtype=torch.long, device=device)
        edge_type_ids = torch.zeros((batch_size, max_edges), dtype=torch.long, device=device)
        edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=device)

        for batch_index, snapshot in enumerate(snapshots):
            if snapshot.source_bytes <= 0:
                continue
            nodes = snapshot.nodes[:max_nodes]
            old_to_new = {node.id: index for index, node in enumerate(nodes)}
            depths, child_indices = _derive_depths_and_child_indices(snapshot, max_nodes, max_depth, max_child_index)
            child_counts = _count_children(snapshot)
            cursor_node = _find_cursor_node(snapshot, max_nodes)
            cursor_ancestors = _ancestor_ids(snapshot, cursor_node) if cursor_node is not None else set()
            source_bytes = max(snapshot.source_bytes, 1)
            source_length = min(_log_norm(snapshot.source_bytes, 65536), 1.0)

            for node_index, node in enumerate(nodes):
                node_type_ids[batch_index, node_index] = stable_hash_id(f"node:{node.type}", node_type_vocab_size)
                node_text_ids[batch_index, node_index] = _node_text_id(node, child_counts.get(node.id, 0), text_vocab_size)
                node_depths[batch_index, node_index] = depths.get(node.id, 0)
                node_child_indices[batch_index, node_index] = child_indices.get(node.id, 0)
                touches_cursor = cursor_node == node.id or abs(node.end_byte - snapshot.source_bytes) <= 1
                is_cursor_ancestor = node.id in cursor_ancestors
                flag_values = [
                    float(node.named),
                    float(node.has_error),
                    float(node.is_missing),
                    float(touches_cursor),
                    float(is_cursor_ancestor),
                    float(snapshot.source_bytes > 0),
                ][:node_flag_dim]
                node_flags[batch_index, node_index, : len(flag_values)] = torch.tensor(
                    flag_values,
                    dtype=torch.float32,
                    device=device,
                )
                start = min(max(node.start_byte / source_bytes, 0.0), 1.0)
                end = min(max(node.end_byte / source_bytes, 0.0), 1.0)
                span = min(max((node.end_byte - node.start_byte) / source_bytes, 0.0), 1.0)
                cursor_distance = min(max((snapshot.source_bytes - node.end_byte) / source_bytes, 0.0), 1.0)
                position_values = [start, end, span, cursor_distance, source_length][:node_position_dim]
                node_positions[batch_index, node_index, : len(position_values)] = torch.tensor(
                    position_values,
                    dtype=torch.float32,
                    device=device,
                )
                node_mask[batch_index, node_index] = True

            kept_edges = [
                edge
                for edge in snapshot.edges
                if edge.source in old_to_new and edge.target in old_to_new
            ][:max_edges]
            for edge_index, edge in enumerate(kept_edges):
                edge_sources[batch_index, edge_index] = old_to_new[edge.source]
                edge_targets[batch_index, edge_index] = old_to_new[edge.target]
                edge_type_ids[batch_index, edge_index] = stable_hash_id(f"edge:{edge.type}", edge_type_vocab_size)
                edge_mask[batch_index, edge_index] = True

        return cls(
            node_type_ids=node_type_ids,
            node_text_ids=node_text_ids,
            node_depths=node_depths,
            node_child_indices=node_child_indices,
            node_flags=node_flags,
            node_positions=node_positions,
            node_mask=node_mask,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_type_ids=edge_type_ids,
            edge_mask=edge_mask,
        )


class IncrementalTreeSitterCudaAST:
    """Incrementally parse generated CUDA code and expose the full AST graph."""

    def __init__(
        self,
        include_text: bool = True,
        text_limit: int = 160,
        allow_raw_code_fallback: bool = True,
    ):
        self.parser, self.provider = self._make_parser()
        self.include_text = include_text
        self.text_limit = text_limit
        self.allow_raw_code_fallback = allow_raw_code_fallback
        self._source = ""
        self._tree = None

    def reset(self) -> None:
        self._source = ""
        self._tree = None

    def parse(self, source: str, incremental: bool = True) -> CudaASTSnapshot:
        source = self._focus_code(source)
        previous_source = self._source if incremental else ""
        previous_tree = self._tree if incremental else None
        tree, reused = self._parse_incremental(source, previous_source, previous_tree)
        if incremental:
            self._source = source
            self._tree = tree
        return self._snapshot(source, tree, reused)

    @staticmethod
    def _make_parser() -> Tuple[Any, str]:
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_cuda
        except ImportError as exc:
            raise RuntimeError(
                "AST-state SCEM requires tree_sitter and tree_sitter_cuda. "
                "Install with `/home/zhujiace/anaconda3/envs/llama/bin/pip install tree-sitter tree-sitter-cuda`."
            ) from exc

        try:
            language = Language(tree_sitter_cuda.language())
        except TypeError:
            language = tree_sitter_cuda.language()
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
        return parser, "tree_sitter_cuda"

    def _parse_incremental(self, source: str, previous_source: str, previous_tree: Any) -> Tuple[Any, bool]:
        source_bytes = source.encode("utf-8", errors="replace")
        if previous_tree is not None and source.startswith(previous_source):
            previous_bytes = previous_source.encode("utf-8", errors="replace")
            start_byte = len(previous_bytes)
            start_point = self._point_at_byte(previous_bytes, start_byte)
            new_end_point = self._point_at_byte(source_bytes, len(source_bytes))
            previous_tree.edit(
                start_byte=start_byte,
                old_end_byte=start_byte,
                new_end_byte=len(source_bytes),
                start_point=start_point,
                old_end_point=start_point,
                new_end_point=new_end_point,
            )
            return self.parser.parse(source_bytes, previous_tree), True
        return self.parser.parse(source_bytes), False

    def _snapshot(self, source: str, tree: Any, incremental_reused: bool) -> CudaASTSnapshot:
        source_bytes = source.encode("utf-8", errors="replace")
        raw_nodes = list(self._walk_nodes(tree.root_node))
        node_keys = [_node_key(node) for node in raw_nodes]
        node_ids = {key: index for index, key in enumerate(node_keys)}
        nodes = [
            CudaASTNode(
                id=node_ids[key],
                type=node.type,
                named=self._is_named(node),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_point=self._point_tuple(node.start_point),
                end_point=self._point_tuple(node.end_point),
                has_error=self._node_has_error(node),
                is_missing=self._is_missing(node),
                text=self._node_text_limited(source_bytes, node) if self.include_text else None,
            )
            for key, node in zip(node_keys, raw_nodes)
        ]
        edges = self._build_edges(raw_nodes, node_ids)
        node_type_counts: Dict[str, int] = {}
        for node in raw_nodes:
            node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
        edge_type_counts: Dict[str, int] = {}
        for edge in edges:
            edge_type_counts[edge.type] = edge_type_counts.get(edge.type, 0) + 1
        return CudaASTSnapshot(
            provider=self.provider,
            incremental_reused=incremental_reused,
            root_has_error=self._node_has_error(tree.root_node),
            source_bytes=len(source_bytes),
            root_id=node_ids[_node_key(tree.root_node)],
            root_type=tree.root_node.type,
            nodes=nodes,
            edges=edges,
            node_type_counts={key: node_type_counts[key] for key in sorted(node_type_counts)},
            edge_type_counts={key: edge_type_counts[key] for key in sorted(edge_type_counts)},
        )

    def _focus_code(self, text: str) -> str:
        return self._focus_code_text(text, allow_raw_code_fallback=self.allow_raw_code_fallback)

    @staticmethod
    def _focus_code_text(text: str, *, allow_raw_code_fallback: bool = True) -> str:
        if GENERATED_PREFIX_MARKER in text:
            text = text.rsplit(GENERATED_PREFIX_MARKER, 1)[1]
        fences = list(re.finditer(r"```[^\n`]*\n?", text))
        if fences:
            if len(fences) % 2 == 1:
                return text[fences[-1].end() :]
            return ""
        if not allow_raw_code_fallback:
            return ""
        code_start = IncrementalTreeSitterCudaAST._raw_code_start(text)
        return text[code_start:] if code_start is not None else ""

    @staticmethod
    def _raw_code_start(text: str) -> Optional[int]:
        patterns = (
            r"(?m)^\s*#\s*include\b",
            r"(?m)^\s*(?:extern\s+\"C\"\s+)?__(?:global__|device__|host__)\b",
            r"(?m)^\s*template\s*<[^>\n]+>\s*(?:\n\s*)?(?:extern\s+\"C\"\s+)?__(?:global__|device__|host__)\b",
        )
        starts = [match.start() for pattern in patterns for match in re.finditer(pattern, text)]
        return min(starts) if starts else None

    @staticmethod
    def _walk_nodes(root: Any) -> Iterable[Any]:
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    @classmethod
    def _build_edges(cls, nodes: Sequence[Any], node_ids: Dict[Tuple[Any, ...], int]) -> List[CudaASTEdge]:
        edges: List[CudaASTEdge] = []
        for parent in nodes:
            parent_id = node_ids[_node_key(parent)]
            children = list(parent.children)
            for index, child in enumerate(children):
                child_id = node_ids[_node_key(child)]
                field_name = cls._field_name_for_child(parent, index)
                edges.append(CudaASTEdge(parent_id, child_id, "child", field_name, index))
                edges.append(CudaASTEdge(child_id, parent_id, "parent", field_name, index))
                if field_name:
                    edges.append(CudaASTEdge(parent_id, child_id, f"field:{field_name}", field_name, index))
                if index > 0:
                    previous_id = node_ids[_node_key(children[index - 1])]
                    edges.append(CudaASTEdge(previous_id, child_id, "next_sibling", child_index=index))
                    edges.append(CudaASTEdge(child_id, previous_id, "prev_sibling", child_index=index - 1))
        return edges

    @staticmethod
    def _field_name_for_child(parent: Any, child_index: int) -> Optional[str]:
        method = getattr(parent, "field_name_for_child", None)
        if not method:
            return None
        try:
            return method(child_index)
        except (TypeError, IndexError):
            return None

    @staticmethod
    def _point_at_byte(source_bytes: bytes, byte_offset: int) -> Any:
        from tree_sitter import Point

        prefix = source_bytes[:byte_offset]
        row = prefix.count(b"\n")
        last_newline = prefix.rfind(b"\n")
        column = len(prefix) if last_newline < 0 else len(prefix) - last_newline - 1
        return Point(row, column)

    @staticmethod
    def _point_tuple(point: Any) -> Tuple[int, int]:
        if hasattr(point, "row") and hasattr(point, "column"):
            row = point.row
            column = point.column
        elif point:
            row = point[0]
            column = point[1]
        else:
            row = 0
            column = 0
        return int(row), int(column)

    @staticmethod
    def _is_named(node: Any) -> bool:
        value = getattr(node, "is_named", False)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _node_has_error(node: Any) -> bool:
        value = getattr(node, "has_error", False)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _is_missing(node: Any) -> bool:
        value = getattr(node, "is_missing", False)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _node_text(source_bytes: bytes, node: Any) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _node_text_limited(self, source_bytes: bytes, node: Any) -> str:
        text = self._node_text(source_bytes, node)
        if len(text) <= self.text_limit:
            return text
        return text[: self.text_limit] + "..."


class CudaASTGraphExtractor:
    """Extract parser-native CUDA AST graphs for SCEM."""

    def __init__(
        self,
        *,
        max_nodes: int = 768,
        max_edges: int = 3072,
        node_type_vocab_size: int = 4096,
        edge_type_vocab_size: int = 1024,
        text_vocab_size: int = 8192,
        max_depth: int = 64,
        max_child_index: int = 64,
        node_flag_dim: int = 6,
        node_position_dim: int = 5,
        text_limit: int = 160,
        cache_dir: str | os.PathLike[str] | None = None,
        allow_raw_code_fallback: bool = True,
    ):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_type_vocab_size = node_type_vocab_size
        self.edge_type_vocab_size = edge_type_vocab_size
        self.text_vocab_size = text_vocab_size
        self.max_depth = max_depth
        self.max_child_index = max_child_index
        self.node_flag_dim = node_flag_dim
        self.node_position_dim = node_position_dim
        self.text_limit = text_limit
        self.allow_raw_code_fallback = allow_raw_code_fallback
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._parser = IncrementalTreeSitterCudaAST(
            include_text=True,
            text_limit=text_limit,
            allow_raw_code_fallback=allow_raw_code_fallback,
        )
        self._incremental_parsers: List[IncrementalTreeSitterCudaAST] = []

    def reset(self) -> None:
        self._parser.reset()
        self._incremental_parsers = []

    def extract_batch(
        self,
        prefixes: Sequence[str],
        *,
        device: torch.device | str | None = None,
        incremental: bool = False,
    ) -> CudaASTGraphBatch:
        if incremental:
            while len(self._incremental_parsers) < len(prefixes):
                self._incremental_parsers.append(
                    IncrementalTreeSitterCudaAST(
                        include_text=True,
                        text_limit=self.text_limit,
                        allow_raw_code_fallback=self.allow_raw_code_fallback,
                    )
                )
            snapshots = [
                self._incremental_parsers[index].parse(prefix, incremental=True)
                for index, prefix in enumerate(prefixes)
            ]
        elif self.cache_dir is not None:
            items = [self._cached_or_parse(prefix) for prefix in prefixes]
            return _batch_from_cache_items(items, device=device)
        else:
            snapshots = [self._parser.parse(prefix, incremental=False) for prefix in prefixes]
        return CudaASTGraphBatch.from_snapshots(
            snapshots,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            node_type_vocab_size=self.node_type_vocab_size,
            edge_type_vocab_size=self.edge_type_vocab_size,
            text_vocab_size=self.text_vocab_size,
            max_depth=self.max_depth,
            max_child_index=self.max_child_index,
            node_flag_dim=self.node_flag_dim,
            node_position_dim=self.node_position_dim,
            device=device,
        )

    def _cached_or_parse(self, prefix: str) -> Dict[str, torch.Tensor]:
        path = self._cache_path(prefix)
        if path.exists():
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        snapshot = self._parser.parse(prefix, incremental=False)
        batch = CudaASTGraphBatch.from_snapshots(
            [snapshot],
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            node_type_vocab_size=self.node_type_vocab_size,
            edge_type_vocab_size=self.edge_type_vocab_size,
            text_vocab_size=self.text_vocab_size,
            max_depth=self.max_depth,
            max_child_index=self.max_child_index,
            node_flag_dim=self.node_flag_dim,
            node_position_dim=self.node_position_dim,
            device=None,
        )
        item = _cache_item_from_batch(batch)
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(item, tmp_path)
        os.replace(tmp_path, path)
        return item

    def _cache_path(self, prefix: str) -> Path:
        assert self.cache_dir is not None
        config = "|".join(
            str(value)
            for value in (
                "ast_graph_v4",
                self.allow_raw_code_fallback,
                self.max_nodes,
                self.max_edges,
                self.node_type_vocab_size,
                self.edge_type_vocab_size,
                self.text_vocab_size,
                self.max_depth,
                self.max_child_index,
                self.node_flag_dim,
                self.node_position_dim,
            )
        )
        digest = hashlib.blake2b(
            f"{config}\0{prefix}".encode("utf-8", errors="replace"),
            digest_size=16,
        ).hexdigest()
        return self.cache_dir / f"{digest}.pt"


def _node_key(node: Any) -> Tuple[Any, ...]:
    return (
        node.type,
        node.start_byte,
        node.end_byte,
        IncrementalTreeSitterCudaAST._point_tuple(node.start_point),
        IncrementalTreeSitterCudaAST._point_tuple(node.end_point),
        IncrementalTreeSitterCudaAST._is_named(node),
    )


def _count_children(snapshot: CudaASTSnapshot) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for edge in snapshot.edges:
        if edge.type == "child":
            counts[edge.source] = counts.get(edge.source, 0) + 1
    return counts


def _parents(snapshot: CudaASTSnapshot) -> Dict[int, int]:
    parents: Dict[int, int] = {}
    for edge in snapshot.edges:
        if edge.type == "child":
            parents[edge.target] = edge.source
    return parents


def _ancestor_ids(snapshot: CudaASTSnapshot, node_id: int) -> set[int]:
    parents = _parents(snapshot)
    ancestors = {node_id}
    current = node_id
    while current in parents:
        current = parents[current]
        if current in ancestors:
            break
        ancestors.add(current)
    return ancestors


def _find_cursor_node(snapshot: CudaASTSnapshot, max_nodes: int) -> Optional[int]:
    nodes = snapshot.nodes[:max_nodes]
    if not nodes:
        return None
    cursor = snapshot.source_bytes
    candidates = [
        node
        for node in nodes
        if node.start_byte <= cursor <= node.end_byte
    ]
    if not candidates:
        candidates = [node for node in nodes if node.end_byte <= cursor]
    if not candidates:
        return snapshot.root_id
    best = min(candidates, key=lambda node: (max(node.end_byte - node.start_byte, 0), -node.start_byte))
    return best.id


def _derive_depths_and_child_indices(
    snapshot: CudaASTSnapshot,
    max_nodes: int,
    max_depth: int,
    max_child_index: int,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    kept = {node.id for node in snapshot.nodes[:max_nodes]}
    depths = {snapshot.root_id: 0}
    child_indices = {snapshot.root_id: 0}
    changed = True
    while changed:
        changed = False
        for edge in snapshot.edges:
            if edge.type != "child" or edge.source not in kept or edge.target not in kept:
                continue
            parent_depth = depths.get(edge.source)
            if parent_depth is None:
                continue
            child_depth = min(parent_depth + 1, max_depth)
            if depths.get(edge.target, max_depth + 1) > child_depth:
                depths[edge.target] = child_depth
                changed = True
            if edge.child_index is not None:
                child_indices[edge.target] = min(max(edge.child_index + 1, 0), max_child_index)
    return depths, child_indices


def _node_text_id(node: CudaASTNode, child_count: int, text_vocab_size: int) -> int:
    text = (node.text or "").strip()
    if not text:
        return 0
    if child_count > 0 and node.type not in {"identifier", "field_identifier", "number_literal", "string_literal"}:
        return 0
    if "\n" in text or len(text) > 80:
        return 0
    return stable_hash_id(f"text:{text}", text_vocab_size)


def _log_norm(value: int, cap: int) -> float:
    import math

    return math.log1p(min(max(value, 0), cap)) / math.log1p(cap)


def _cache_item_from_batch(batch: CudaASTGraphBatch) -> Dict[str, torch.Tensor]:
    return {
        "node_type_ids": batch.node_type_ids[0].cpu(),
        "node_text_ids": batch.node_text_ids[0].cpu(),
        "node_depths": batch.node_depths[0].cpu(),
        "node_child_indices": batch.node_child_indices[0].cpu(),
        "node_flags": batch.node_flags[0].cpu(),
        "node_positions": batch.node_positions[0].cpu(),
        "node_mask": batch.node_mask[0].cpu(),
        "edge_sources": batch.edge_sources[0].cpu(),
        "edge_targets": batch.edge_targets[0].cpu(),
        "edge_type_ids": batch.edge_type_ids[0].cpu(),
        "edge_mask": batch.edge_mask[0].cpu(),
    }


def _batch_from_cache_items(
    items: Sequence[Dict[str, torch.Tensor]],
    device: torch.device | str | None = None,
) -> CudaASTGraphBatch:
    def stack(name: str) -> torch.Tensor:
        tensor = torch.stack([item[name] for item in items])
        return tensor.to(device=device) if device is not None else tensor

    return CudaASTGraphBatch(
        node_type_ids=stack("node_type_ids"),
        node_text_ids=stack("node_text_ids"),
        node_depths=stack("node_depths"),
        node_child_indices=stack("node_child_indices"),
        node_flags=stack("node_flags"),
        node_positions=stack("node_positions"),
        node_mask=stack("node_mask"),
        edge_sources=stack("edge_sources"),
        edge_targets=stack("edge_targets"),
        edge_type_ids=stack("edge_type_ids"),
        edge_mask=stack("edge_mask"),
    )
