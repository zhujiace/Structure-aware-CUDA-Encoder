from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


SAMPLE_CODE = """#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
"""


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUDA AST Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --border: #d8dee8;
      --text: #18212f;
      --muted: #667084;
      --accent: #0f766e;
      --error: #b42318;
      --warn: #b54708;
      --code: #101828;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 650;
      letter-spacing: 0;
    }
    #status {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 13px;
    }
    .badge {
      border: 1px solid var(--border);
      background: #f9fafb;
      border-radius: 6px;
      padding: 3px 7px;
      white-space: nowrap;
    }
    .badge.ok { color: var(--accent); border-color: #99d6cf; background: #f0fdfa; }
    .badge.err { color: var(--error); border-color: #fda29b; background: #fff1f0; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 0.95fr) minmax(420px, 1.05fr);
      gap: 12px;
      height: calc(100vh - 56px);
      padding: 12px;
    }
    section {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .section-head {
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
    }
    textarea {
      flex: 1;
      width: 100%;
      min-height: 0;
      resize: none;
      border: 0;
      outline: 0;
      padding: 14px;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--code);
      background: #fbfcfe;
      tab-size: 4;
    }
    #ast {
      flex: 1;
      min-height: 0;
      overflow: auto;
      padding: 10px 12px 22px;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #fcfcfd;
    }
    details {
      margin-left: 14px;
      border-left: 1px solid #edf0f5;
      padding-left: 9px;
    }
    summary {
      cursor: pointer;
      list-style-position: outside;
      min-height: 22px;
    }
    .node-type { color: #175cd3; font-weight: 650; }
    .field { color: #7a5af8; }
    .span { color: var(--muted); }
    .text { color: #027a48; }
    .error { color: var(--error); font-weight: 650; }
    .missing { color: var(--warn); font-weight: 650; }
    .leaf {
      margin-left: 22px;
      min-height: 22px;
    }
    .empty, .message {
      color: var(--muted);
      padding: 12px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    button {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      min-height: 28px;
      padding: 0 9px;
      cursor: pointer;
    }
    button:hover { border-color: #aab4c5; background: #f9fafb; }
    @media (max-width: 900px) {
      header { height: auto; min-height: 56px; align-items: flex-start; gap: 8px; flex-direction: column; padding: 10px 12px; }
      main { grid-template-columns: 1fr; height: auto; min-height: calc(100vh - 76px); }
      section { min-height: 45vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>CUDA AST Viewer</h1>
    <div id="status">
      <span class="badge">waiting</span>
    </div>
  </header>
  <main>
    <section>
      <div class="section-head">
        <span>CUDA source</span>
        <div class="toolbar">
          <button id="sample" type="button">Sample</button>
          <button id="clear" type="button">Clear</button>
        </div>
      </div>
      <textarea id="source" spellcheck="false"></textarea>
    </section>
    <section>
      <div class="section-head">
        <span>Incremental AST</span>
        <div class="toolbar">
          <button id="expand" type="button">Expand</button>
          <button id="collapse" type="button">Collapse</button>
        </div>
      </div>
      <div id="ast" aria-live="polite"></div>
    </section>
  </main>
  <script>
    const sampleCode = __SAMPLE_JSON__;
    const source = document.getElementById("source");
    const ast = document.getElementById("ast");
    const statusBar = document.getElementById("status");
    let timer = null;
    let requestSeq = 0;

    function escapeText(value) {
      return String(value ?? "");
    }

    function setStatus(data, stateText) {
      if (!data) {
        statusBar.innerHTML = `<span class="badge">${stateText}</span>`;
        return;
      }
      const errorClass = data.root_has_error ? "err" : "ok";
      const reusedClass = data.incremental_reused ? "ok" : "";
      statusBar.innerHTML = [
        `<span class="badge ${errorClass}">root error: ${data.root_has_error}</span>`,
        `<span class="badge ${reusedClass}">incremental: ${data.incremental_reused}</span>`,
        `<span class="badge">nodes: ${data.node_count}</span>`,
        `<span class="badge">edges: ${data.edge_count}</span>`,
        `<span class="badge">bytes: ${data.source_bytes}</span>`
      ].join("");
    }

    function appendSpan(parent, className, text) {
      const span = document.createElement("span");
      span.className = className;
      span.textContent = text;
      parent.appendChild(span);
    }

    function nodeLabel(node, parent) {
      if (node.field) {
        appendSpan(parent, "field", node.field + ": ");
      }
      appendSpan(parent, "node-type", node.type);
      if (node.has_error) appendSpan(parent, "error", " error");
      if (node.is_missing) appendSpan(parent, "missing", " missing");
      appendSpan(parent, "span", ` [${node.start_point[0]}:${node.start_point[1]}-${node.end_point[0]}:${node.end_point[1]}]`);
      if (node.text) appendSpan(parent, "text", ` ${JSON.stringify(node.text)}`);
    }

    function renderNode(node, depth) {
      if (node.truncated) {
        const div = document.createElement("div");
        div.className = "message";
        div.textContent = node.message;
        return div;
      }
      const children = node.children || [];
      if (!children.length) {
        const div = document.createElement("div");
        div.className = "leaf";
        nodeLabel(node, div);
        return div;
      }
      const details = document.createElement("details");
      details.open = depth < 3;
      const summary = document.createElement("summary");
      nodeLabel(node, summary);
      details.appendChild(summary);
      for (const child of children) {
        details.appendChild(renderNode(child, depth + 1));
      }
      return details;
    }

    function renderAst(data) {
      ast.replaceChildren();
      if (!data.root) {
        const div = document.createElement("div");
        div.className = "empty";
        div.textContent = "No parse tree.";
        ast.appendChild(div);
        return;
      }
      ast.appendChild(renderNode(data.root, 0));
    }

    async function parseNow() {
      const seq = ++requestSeq;
      setStatus(null, "parsing");
      try {
        const response = await fetch("/parse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: source.value })
        });
        const data = await response.json();
        if (seq !== requestSeq) return;
        if (!response.ok || data.error) throw new Error(data.error || response.statusText);
        setStatus(data);
        renderAst(data);
      } catch (err) {
        setStatus(null, "error");
        ast.replaceChildren();
        const div = document.createElement("div");
        div.className = "message";
        div.textContent = escapeText(err.message || err);
        ast.appendChild(div);
      }
    }

    function scheduleParse() {
      window.clearTimeout(timer);
      timer = window.setTimeout(parseNow, 120);
    }

    source.addEventListener("input", scheduleParse);
    source.addEventListener("keydown", event => {
      if (event.key === "Tab") {
        event.preventDefault();
        const start = source.selectionStart;
        const end = source.selectionEnd;
        source.value = source.value.slice(0, start) + "    " + source.value.slice(end);
        source.selectionStart = source.selectionEnd = start + 4;
        scheduleParse();
      }
    });
    document.getElementById("sample").addEventListener("click", () => {
      source.value = sampleCode;
      scheduleParse();
      source.focus();
    });
    document.getElementById("clear").addEventListener("click", () => {
      source.value = "";
      scheduleParse();
      source.focus();
    });
    document.getElementById("expand").addEventListener("click", () => {
      ast.querySelectorAll("details").forEach(item => { item.open = true; });
    });
    document.getElementById("collapse").addEventListener("click", () => {
      ast.querySelectorAll("details").forEach((item, index) => { item.open = index === 0; });
    });
    source.value = sampleCode;
    parseNow();
  </script>
</body>
</html>
"""


def make_language() -> Any:
    try:
        from tree_sitter import Language
        import tree_sitter_cuda
    except ImportError as exc:
        raise RuntimeError(
            "Missing parser dependency. Install with: "
            "/home/zhujiace/anaconda3/envs/llama/bin/pip install tree-sitter tree-sitter-cuda"
        ) from exc

    try:
        return Language(tree_sitter_cuda.language())
    except TypeError:
        return tree_sitter_cuda.language()


def make_parser() -> Any:
    from tree_sitter import Parser

    parser = Parser()
    language = make_language()
    if hasattr(parser, "set_language"):
        parser.set_language(language)
    else:
        parser.language = language
    return parser


def point_at_byte(source: bytes, byte_offset: int) -> Any:
    row = source[:byte_offset].count(b"\n")
    last_newline = source.rfind(b"\n", 0, byte_offset)
    column = byte_offset if last_newline < 0 else byte_offset - last_newline - 1
    try:
        from tree_sitter import Point

        return Point(row, column)
    except (ImportError, TypeError):
        return (row, column)


def point_tuple(point: Any) -> Tuple[int, int]:
    if hasattr(point, "row") and hasattr(point, "column"):
        return int(point.row), int(point.column)
    if point:
        return int(point[0]), int(point[1])
    return 0, 0


def node_has_error(node: Any) -> bool:
    value = getattr(node, "has_error", False)
    return bool(value() if callable(value) else value)


def node_is_missing(node: Any) -> bool:
    value = getattr(node, "is_missing", False)
    return bool(value() if callable(value) else value)


def node_is_named(node: Any) -> bool:
    value = getattr(node, "is_named", False)
    return bool(value() if callable(value) else value)


def field_name_for_child(parent: Any, child_index: int) -> Optional[str]:
    method = getattr(parent, "field_name_for_child", None)
    if not method:
        return None
    try:
        return method(child_index)
    except (TypeError, IndexError):
        return None


def common_edit_span(old: bytes, new: bytes) -> Tuple[int, int, int]:
    start = 0
    max_prefix = min(len(old), len(new))
    while start < max_prefix and old[start] == new[start]:
        start += 1

    old_end = len(old)
    new_end = len(new)
    while old_end > start and new_end > start and old[old_end - 1] == new[new_end - 1]:
        old_end -= 1
        new_end -= 1
    return start, old_end, new_end


def iter_nodes(root: Any) -> Iterable[Any]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


class IncrementalCudaParser:
    def __init__(self, *, text_limit: int = 120, max_nodes: int = 1500):
        self.parser = make_parser()
        self.text_limit = text_limit
        self.max_nodes = max_nodes
        self._source = b""
        self._tree = None

    def reset(self) -> None:
        self._source = b""
        self._tree = None

    def parse(self, source: str, *, incremental: bool = True) -> Dict[str, Any]:
        source_bytes = source.encode("utf-8", errors="replace")
        reused = False

        if incremental and self._tree is not None:
            start_byte, old_end_byte, new_end_byte = common_edit_span(self._source, source_bytes)
            if start_byte == old_end_byte == new_end_byte and self._source == source_bytes:
                tree = self._tree
                reused = True
            else:
                self._tree.edit(
                    start_byte=start_byte,
                    old_end_byte=old_end_byte,
                    new_end_byte=new_end_byte,
                    start_point=point_at_byte(self._source, start_byte),
                    old_end_point=point_at_byte(self._source, old_end_byte),
                    new_end_point=point_at_byte(source_bytes, new_end_byte),
                )
                tree = self.parser.parse(source_bytes, self._tree)
                reused = True
        else:
            tree = self.parser.parse(source_bytes)

        self._source = source_bytes
        self._tree = tree
        return self.snapshot(source_bytes, tree, reused)

    def snapshot(self, source_bytes: bytes, tree: Any, incremental_reused: bool) -> Dict[str, Any]:
        counter: Counter[str] = Counter()
        edge_count = 0
        for node in iter_nodes(tree.root_node):
            counter[node.type] += 1
            edge_count += len(node.children) * 2
            for index in range(len(node.children)):
                if field_name_for_child(node, index):
                    edge_count += 1
            if len(node.children) > 1:
                edge_count += (len(node.children) - 1) * 2

        used = 0

        def convert(node: Any, depth: int = 0, field: Optional[str] = None) -> Dict[str, Any]:
            nonlocal used
            if used >= self.max_nodes:
                return {
                    "type": "truncated",
                    "truncated": True,
                    "message": f"AST display truncated at {self.max_nodes} nodes.",
                }
            used += 1

            children = list(node.children)
            item: Dict[str, Any] = {
                "type": node.type,
                "field": field,
                "named": node_is_named(node),
                "has_error": node_has_error(node),
                "is_missing": node_is_missing(node),
                "start_byte": int(node.start_byte),
                "end_byte": int(node.end_byte),
                "start_point": point_tuple(node.start_point),
                "end_point": point_tuple(node.end_point),
            }
            text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            if not children or len(text) <= self.text_limit:
                item["text"] = text if len(text) <= self.text_limit else text[: self.text_limit] + "..."
            if children:
                item["children"] = [
                    convert(child, depth + 1, field_name_for_child(node, index))
                    for index, child in enumerate(children)
                ]
            return item

        return {
            "provider": "tree_sitter_cuda",
            "incremental_reused": incremental_reused,
            "source_bytes": len(source_bytes),
            "root_type": tree.root_node.type,
            "root_has_error": node_has_error(tree.root_node),
            "node_count": sum(counter.values()),
            "edge_count": edge_count,
            "node_type_counts": dict(counter.most_common(40)),
            "root": convert(tree.root_node),
        }


class CudaASTViewer:
    def __init__(self, *, text_limit: int, max_nodes: int):
        self.parser = IncrementalCudaParser(text_limit=text_limit, max_nodes=max_nodes)
        self.lock = threading.Lock()

    def parse(self, source: str) -> Dict[str, Any]:
        with self.lock:
            return self.parser.parse(source, incremental=True)


def make_handler(app: CudaASTViewer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - - %s\n" % (self.address_string(), format % args))

        def send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            html = HTML_PAGE.replace("__SAMPLE_JSON__", json.dumps(SAMPLE_CODE))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/parse":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                source = str(request.get("source", ""))
                self.send_json(200, app.parse(source))
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

    return Handler


def print_tree(node: Dict[str, Any], *, indent: str = "") -> None:
    if node.get("truncated"):
        print(f"{indent}{node['message']}")
        return
    field = f"{node['field']}: " if node.get("field") else ""
    flags = []
    if node.get("has_error"):
        flags.append("ERROR")
    if node.get("is_missing"):
        flags.append("MISSING")
    flag_text = f" [{' '.join(flags)}]" if flags else ""
    span = f"{node['start_point'][0]}:{node['start_point'][1]}-{node['end_point'][0]}:{node['end_point'][1]}"
    text = node.get("text")
    text_suffix = ""
    if text and "\n" not in text and len(text) <= 80:
        text_suffix = f" {text!r}"
    print(f"{indent}{field}{node['type']}{flag_text} [{span}]{text_suffix}")
    for child in node.get("children", []):
        print_tree(child, indent=indent + "  ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight CUDA AST viewer using tree_sitter_cuda.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--text-limit", type=int, default=120)
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--stdin", action="store_true", help="Parse CUDA source from stdin and print a text AST.")
    parser.add_argument("--json", action="store_true", help="With --stdin, print the full AST payload as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = CudaASTViewer(text_limit=args.text_limit, max_nodes=args.max_nodes)

    if args.stdin:
        source = sys.stdin.read()
        payload = app.parse(source)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            summary = (
                f"provider={payload['provider']} "
                f"incremental={payload['incremental_reused']} "
                f"root_error={payload['root_has_error']} "
                f"nodes={payload['node_count']} "
                f"edges={payload['edge_count']} "
                f"bytes={payload['source_bytes']}"
            )
            print(summary)
            print_tree(payload["root"])
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(f"CUDA AST viewer serving at {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CUDA AST viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
