from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


PROGRAM_REGIONS = {
    "unknown": 0,
    "signature": 1,
    "setup": 2,
    "indexing": 3,
    "guard": 4,
    "shared_memory": 5,
    "compute": 6,
    "write_back": 7,
}

GENERATED_PREFIX_MARKER = "\n<<<SCEM_GENERATED_PREFIX>>>\n"


@dataclass(frozen=True)
class CudaProgramState:
    """Compact CUDA program state for one decoding prefix."""

    program_region: int = 0
    may_need_guard: bool = False
    may_need_shared_memory: bool = False
    may_need_synchronization: bool = False
    may_need_reduction: bool = False
    may_need_atomic: bool = False
    may_need_texture: bool = False
    may_need_multidim_indexing: bool = False
    may_need_math: bool = False
    has_task_description: bool = False
    has_input_spec: bool = False
    has_output_spec: bool = False
    harness_kernel_only: bool = False
    has_index_definition: bool = False
    has_thread_index: bool = False
    has_block_index: bool = False
    has_open_guard: bool = False
    has_shared_memory: bool = False
    has_syncthreads: bool = False
    has_write_back: bool = False
    writes_declared_output: bool = False
    has_kernel_signature: bool = False
    has_kernel_body: bool = False
    has_required_kernel_name: bool = False
    has_loop: bool = False
    has_atomic: bool = False
    has_device_helper: bool = False
    braces_balanced: bool = True
    statement_stable: bool = True
    brace_depth: float = 0.0
    paren_depth: float = 0.0
    code_length: float = 0.0
    line_count: float = 0.0
    signature_arg_count: float = 0.0
    pointer_arg_count: float = 0.0
    scalar_arg_count: float = 0.0
    output_ref_count: float = 0.0


@dataclass
class CudaProgramStateBatch:
    program_region: torch.LongTensor
    static_flags: torch.Tensor
    prefix_flags: torch.Tensor
    numeric_features: torch.Tensor

    def to(self, device: torch.device | str) -> "CudaProgramStateBatch":
        return CudaProgramStateBatch(
            program_region=self.program_region.to(device),
            static_flags=self.static_flags.to(device),
            prefix_flags=self.prefix_flags.to(device),
            numeric_features=self.numeric_features.to(device),
        )

    @classmethod
    def from_states(
        cls,
        states: Sequence[CudaProgramState],
        device: torch.device | str | None = None,
    ) -> "CudaProgramStateBatch":
        program_region = torch.tensor([s.program_region for s in states], dtype=torch.long, device=device)
        static_flags = torch.tensor(
            [
                [s.may_need_guard, s.may_need_shared_memory, s.may_need_synchronization]
                + [
                    s.may_need_reduction,
                    s.may_need_atomic,
                    s.may_need_texture,
                    s.may_need_multidim_indexing,
                    s.may_need_math,
                    s.has_task_description,
                    s.has_input_spec,
                    s.has_output_spec,
                    s.harness_kernel_only,
                ]
                for s in states
            ],
            dtype=torch.float32,
            device=device,
        )
        prefix_flags = torch.tensor(
            [
                [
                    s.has_index_definition,
                    s.has_thread_index,
                    s.has_block_index,
                    s.has_open_guard,
                    s.has_shared_memory,
                    s.has_syncthreads,
                    s.has_write_back,
                    s.writes_declared_output,
                    s.has_kernel_signature,
                    s.has_kernel_body,
                    s.has_required_kernel_name,
                    s.has_loop,
                    s.has_atomic,
                    s.has_device_helper,
                    s.braces_balanced,
                    s.statement_stable,
                ]
                for s in states
            ],
            dtype=torch.float32,
            device=device,
        )
        numeric_features = torch.tensor(
            [
                [
                    s.brace_depth,
                    s.paren_depth,
                    s.code_length,
                    s.line_count,
                    s.signature_arg_count,
                    s.pointer_arg_count,
                    s.scalar_arg_count,
                    s.output_ref_count,
                ]
                for s in states
            ],
            dtype=torch.float32,
            device=device,
        )
        return cls(
            program_region=program_region,
            static_flags=static_flags,
            prefix_flags=prefix_flags,
            numeric_features=numeric_features,
        )


class CudaProgramStateExtractor:
    """Heuristic prefix-state extractor used before a learned/parser extractor exists."""

    def extract_batch(self, prefixes: Iterable[str]) -> List[CudaProgramState]:
        return [self.extract(prefix) for prefix in prefixes]

    def extract(self, prefix: str) -> CudaProgramState:
        context_text, generated_text = self._split_context_and_generated(prefix)
        full_text = context_text + "\n" + generated_text
        code_prefix = self._focus_cuda_prefix(generated_text)
        signature, kernel_name = self._extract_required_signature(full_text)
        arg_count, pointer_arg_count, scalar_arg_count = self._signature_stats(signature)
        output_names = self._extract_output_names(full_text)
        lowered = code_prefix.lower()
        full_lowered = full_text.lower()
        brace_depth = max(0, code_prefix.count("{") - code_prefix.count("}"))
        paren_depth = max(0, code_prefix.count("(") - code_prefix.count(")"))
        has_guard_token = bool(re.search(r"\bif\s*\(", code_prefix)) or any(
            token in code_prefix for token in ("&&", "||")
        )
        has_write_back = self._has_write_back(code_prefix)
        has_shared = "__shared__" in code_prefix or "extern __shared__" in code_prefix
        has_sync = "__syncthreads" in code_prefix
        has_thread_index = "threadIdx." in code_prefix
        has_block_index = "blockIdx." in code_prefix
        has_atomic = any(token in code_prefix for token in ("atomicAdd", "atomicMax", "atomicMin", "atomicCAS"))
        has_index = any(
            token in code_prefix
            for token in (
                "threadIdx.",
                "blockIdx.",
                "blockDim.",
                "int idx",
                "int i",
                "int row",
                "int col",
            )
        )
        writes_declared_output = self._writes_declared_output(code_prefix, output_names)
        has_kernel_signature = "__global__" in code_prefix
        has_kernel_body = has_kernel_signature and "{" in code_prefix
        has_required_kernel_name = bool(kernel_name and kernel_name in code_prefix)
        has_loop = bool(re.search(r"\b(for|while)\s*\(", code_prefix))
        has_device_helper = "__device__" in code_prefix

        region = self._infer_region(code_prefix, has_guard_token, has_shared, has_write_back)
        return CudaProgramState(
            program_region=region,
            may_need_guard=self._infer_may_need_guard(full_text, code_prefix, has_guard_token),
            may_need_shared_memory=any(token in full_lowered for token in ("matmul", "matrix", "tile", "shared", "convolution", "stencil")),
            may_need_synchronization=has_shared or any(token in full_lowered for token in ("shared", "tile", "reduction", "sum", "prefix")),
            may_need_reduction=any(token in full_lowered for token in ("sum", "mean", "reduce", "reduction", "max", "min", "argmax", "loss")),
            may_need_atomic=any(token in full_lowered for token in ("atomic", "histogram", "scatter", "accumulate", "sum", "reduction")),
            may_need_texture=any(token in full_lowered for token in ("texture", "tex2d", "image", "pixel", "rgba", "uchar4")),
            may_need_multidim_indexing=any(token in full_lowered for token in ("shape = (", "height", "width", "row", "col", "matrix", "image", "tensor")),
            may_need_math=any(token in full_lowered for token in ("exp", "log", "sqrt", "sin", "cos", "pow", "softmax", "sigmoid", "loss")),
            has_task_description="task description" in full_lowered or "task:" in full_lowered,
            has_input_spec="input:" in full_lowered or "inputs:" in full_lowered,
            has_output_spec="output:" in full_lowered or "outputs:" in full_lowered,
            harness_kernel_only="fixed test harness" in full_lowered or "required kernel signature" in full_lowered,
            has_index_definition=has_index,
            has_thread_index=has_thread_index,
            has_block_index=has_block_index,
            has_open_guard=self._has_open_guard(code_prefix),
            has_shared_memory=has_shared,
            has_syncthreads=has_sync,
            has_write_back=has_write_back,
            writes_declared_output=writes_declared_output,
            has_kernel_signature=has_kernel_signature,
            has_kernel_body=has_kernel_body,
            has_required_kernel_name=has_required_kernel_name,
            has_loop=has_loop,
            has_atomic=has_atomic,
            has_device_helper=has_device_helper,
            braces_balanced=brace_depth == 0,
            statement_stable=self._is_statement_stable(code_prefix),
            brace_depth=float(min(brace_depth, 32)) / 32.0,
            paren_depth=float(min(paren_depth, 32)) / 32.0,
            code_length=float(min(len(code_prefix), 4096)) / 4096.0,
            line_count=float(min(code_prefix.count("\n") + 1 if code_prefix else 0, 256)) / 256.0,
            signature_arg_count=float(min(arg_count, 32)) / 32.0,
            pointer_arg_count=float(min(pointer_arg_count, 16)) / 16.0,
            scalar_arg_count=float(min(scalar_arg_count, 16)) / 16.0,
            output_ref_count=float(min(self._count_output_refs(code_prefix, output_names), 16)) / 16.0,
        )

    @staticmethod
    def _infer_region(prefix: str, has_guard: bool, has_shared: bool, has_write_back: bool) -> int:
        if "__global__" in prefix and "{" not in prefix:
            return PROGRAM_REGIONS["signature"]
        if has_write_back:
            return PROGRAM_REGIONS["write_back"]
        if has_guard:
            return PROGRAM_REGIONS["guard"]
        if has_shared:
            return PROGRAM_REGIONS["shared_memory"]
        if any(token in prefix for token in ("threadIdx.", "blockIdx.", "blockDim.")):
            return PROGRAM_REGIONS["indexing"]
        if "{" in prefix:
            return PROGRAM_REGIONS["compute"]
        return PROGRAM_REGIONS["setup"]

    @staticmethod
    def _focus_cuda_prefix(prefix: str) -> str:
        """Return the generated CUDA/code portion instead of prompt scaffolding."""

        if not prefix:
            return ""

        fences = list(re.finditer(r"```[^\n`]*\n?", prefix))
        if fences:
            if len(fences) % 2 == 1:
                return prefix[fences[-1].end() :]
            return prefix[fences[-2].end() : fences[-1].start()]

        cuda_markers = [
            "__global__",
            "__device__",
            "__host__",
            "#include",
            "extern \"C\"",
        ]
        starts = [prefix.find(marker) for marker in cuda_markers if marker in prefix]
        if starts:
            return prefix[min(starts) :]
        return prefix

    @staticmethod
    def _split_context_and_generated(text: str) -> Tuple[str, str]:
        if GENERATED_PREFIX_MARKER in text:
            context, generated = text.rsplit(GENERATED_PREFIX_MARKER, 1)
            return context, generated
        return "", text

    @staticmethod
    def _has_write_back(prefix: str) -> bool:
        if any(token in prefix for token in ("atomicAdd", "atomicMax", "atomicMin", "atomicCAS")):
            return True
        return bool(re.search(r"\b[a-zA-Z_]\w*(?:\s*\[[^\]]+\])+\s*=", prefix))

    @staticmethod
    def _has_open_guard(prefix: str) -> bool:
        matches = list(re.finditer(r"\bif\s*\(", prefix))
        if not matches:
            return False
        last_if = matches[-1].start()
        return last_if > prefix.rfind("}")

    @staticmethod
    def _is_statement_stable(prefix: str) -> bool:
        stripped = prefix.rstrip()
        if not stripped:
            return True
        return stripped.endswith((";", "{", "}"))

    @staticmethod
    def _extract_required_signature(text: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            r"Required kernel signature:\s*(.+?)(?:\n```|\n\n|\nFixed main|\Z)",
            r"(__global__\s+void\s+[A-Za-z_]\w*\s*\([^;\{]*\))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            signature = " ".join(match.group(1).strip().split())
            name_match = re.search(r"\bvoid\s+([A-Za-z_]\w*)\s*\(", signature)
            return signature, name_match.group(1) if name_match else None
        return None, None

    @staticmethod
    def _signature_stats(signature: Optional[str]) -> Tuple[int, int, int]:
        if not signature:
            return 0, 0, 0
        args_match = re.search(r"\((.*)\)", signature)
        if not args_match:
            return 0, 0, 0
        args = [arg.strip() for arg in args_match.group(1).split(",") if arg.strip() and arg.strip() != "void"]
        pointer_args = sum("*" in arg for arg in args)
        scalar_args = len(args) - pointer_args
        return len(args), pointer_args, scalar_args

    @staticmethod
    def _extract_output_names(text: str) -> List[str]:
        names = []
        for line in text.splitlines():
            lowered = line.lower()
            if "output" not in lowered and not lowered.lstrip().startswith("-"):
                continue
            for match in re.finditer(r"['`]([A-Za-z_]\w*)['`]", line):
                names.append(match.group(1))
            bullet = re.match(r"\s*-\s*([A-Za-z_]\w*)\s*:", line)
            if bullet:
                names.append(bullet.group(1))
        for fallback in ("output", "out", "g_odata", "result", "loss"):
            if re.search(rf"\b{re.escape(fallback)}\b", text):
                names.append(fallback)
        deduped = []
        seen = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        return deduped[:8]

    @staticmethod
    def _writes_declared_output(prefix: str, output_names: Sequence[str]) -> bool:
        for name in output_names:
            if re.search(rf"\b{re.escape(name)}\s*(?:\[|=)", prefix):
                return True
        return False

    @staticmethod
    def _count_output_refs(prefix: str, output_names: Sequence[str]) -> int:
        return sum(len(re.findall(rf"\b{re.escape(name)}\b", prefix)) for name in output_names)

    @staticmethod
    def _infer_may_need_guard(full_text: str, code_prefix: str, has_guard: bool) -> bool:
        if has_guard:
            return True
        if not any(token in code_prefix for token in ("threadIdx.", "blockIdx.", "blockDim.")):
            return False
        prefix = full_text
        return any(
            token in prefix
            for token in (
                " int n",
                "int n,",
                "int n)",
                " size",
                " length",
                " width",
                " height",
                " num",
                " N",
            )
        )


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


class IncrementalTreeSitterCudaAST:
    """Incrementally parse generated CUDA code and expose the full AST graph.

    This component intentionally does not emit ``CudaProgramState`` and is not
    wired into train/eval. It keeps the parser-native tree structure instead of
    compressing the AST into hand-written CUDA metrics.
    """

    def __init__(self, include_text: bool = True, text_limit: int = 160):
        self.parser, self.provider = self._make_parser()
        self.include_text = include_text
        self.text_limit = text_limit
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
                "IncrementalTreeSitterCudaAST requires tree_sitter and tree_sitter_cuda. "
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
        node_ids = {self._node_key(node): index for index, node in enumerate(raw_nodes)}
        nodes = [
            CudaASTNode(
                id=node_ids[self._node_key(node)],
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
            for node in raw_nodes
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
            root_id=node_ids[self._node_key(tree.root_node)],
            root_type=tree.root_node.type,
            nodes=nodes,
            edges=edges,
            node_type_counts={key: node_type_counts[key] for key in sorted(node_type_counts)},
            edge_type_counts={key: edge_type_counts[key] for key in sorted(edge_type_counts)},
        )

    @staticmethod
    def _focus_code(text: str) -> str:
        if GENERATED_PREFIX_MARKER in text:
            text = text.rsplit(GENERATED_PREFIX_MARKER, 1)[1]
        fences = list(re.finditer(r"```[^\n`]*\n?", text))
        if fences:
            if len(fences) % 2 == 1:
                return text[fences[-1].end() :]
            return text[fences[-2].end() : fences[-1].start()]
        markers = [text.find(marker) for marker in ("__global__", "__device__", "__host__", "#include") if marker in text]
        return text[min(markers) :] if markers else text

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
            parent_id = node_ids[cls._node_key(parent)]
            children = list(parent.children)
            for index, child in enumerate(children):
                child_id = node_ids[cls._node_key(child)]
                field_name = cls._field_name_for_child(parent, index)
                edges.append(
                    CudaASTEdge(
                        source=parent_id,
                        target=child_id,
                        type="child",
                        field_name=field_name,
                        child_index=index,
                    )
                )
                edges.append(
                    CudaASTEdge(
                        source=child_id,
                        target=parent_id,
                        type="parent",
                        field_name=field_name,
                        child_index=index,
                    )
                )
                if field_name:
                    edges.append(
                        CudaASTEdge(
                            source=parent_id,
                            target=child_id,
                            type=f"field:{field_name}",
                            field_name=field_name,
                            child_index=index,
                        )
                )
                if index > 0:
                    previous_id = node_ids[cls._node_key(children[index - 1])]
                    edges.append(
                        CudaASTEdge(
                            source=previous_id,
                            target=child_id,
                            type="next_sibling",
                            child_index=index,
                        )
                    )
                    edges.append(
                        CudaASTEdge(
                            source=child_id,
                            target=previous_id,
                            type="prev_sibling",
                            child_index=index - 1,
                        )
                    )
        return edges

    @classmethod
    def _node_key(cls, node: Any) -> Tuple[Any, ...]:
        return (
            node.type,
            node.start_byte,
            node.end_byte,
            cls._point_tuple(node.start_point),
            cls._point_tuple(node.end_point),
            cls._is_named(node),
        )

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
