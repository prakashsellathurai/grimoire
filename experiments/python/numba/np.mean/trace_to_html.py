#!/usr/bin/env python3
"""
Convert a Python trace output file into an HTML call tree by file transitions.

The trace format expected is similar to `python -m trace --trace` output, e.g.:
    --- modulename: foo, funcname: bar
    foo.py(12):  x = 1

The resulting HTML shows a hierarchical tree of file -> line -> file transitions
based on observed execution flow in the trace stream.
"""

from __future__ import annotations

import argparse
import html
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


FILE_RE = re.compile(
    r"^\s*(?P<file><[^>]+>|.+\.pyw?)\((?P<line>\d+)\)(?::\s*(?P<code>.*))?$"
)


@dataclass
class Node:
    kind: str
    name: str
    line_no: Optional[int] = None
    code: Optional[str] = None
    count: int = 0
    children: Dict[str, "Node"] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    def get_line_child(self, line_no: int, code: Optional[str]) -> Tuple["Node", bool]:
        key = f"L:{line_no}"
        child = self.children.get(key)
        if child is None:
            child = Node(kind="line", name=str(line_no), line_no=line_no, code=code)
            self.children[key] = child
            self.order.append(key)
            return child, True
        if child.code is None and code:
            child.code = code
        return child, False

    def get_file_child(self, file_name: str) -> Tuple["Node", bool]:
        key = f"F:{file_name}"
        child = self.children.get(key)
        if child is None:
            child = Node(kind="file", name=file_name)
            self.children[key] = child
            self.order.append(key)
            return child, True
        return child, False

    def iter_children(self) -> Iterable["Node"]:
        for key in self.order:
            yield self.children[key]


@dataclass
class Stats:
    file_events: int = 0
    nodes_created: int = 0
    max_depth: int = 0


def normalize_file(name: str, basename_only: bool) -> str:
    if not basename_only:
        return name
    if name.startswith("<") and name.endswith(">"):
        return name
    return os.path.basename(name)


def parse_trace(path: str, basename_only: bool) -> Tuple[Optional[Node], Stats]:
    stats = Stats()
    root: Optional[Node] = None
    file_stack: List[Node] = []
    current_line: Optional[Node] = None

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = FILE_RE.search(line)
            if not match:
                continue
            stats.file_events += 1
            file_name = normalize_file(match.group("file").strip(), basename_only)
            line_no = int(match.group("line"))
            code = match.group("code")
            if code is not None:
                code = code.rstrip()

            if not file_stack:
                if root is None:
                    root = Node(kind="file", name=file_name)
                    stats.nodes_created += 1
                root.count += 1
                line_node, created = root.get_line_child(line_no, code)
                if created:
                    stats.nodes_created += 1
                line_node.count += 1
                current_line = line_node
                file_stack = [root]
                stats.max_depth = max(stats.max_depth, 1)
                continue

            if file_name == file_stack[-1].name:
                line_node, created = file_stack[-1].get_line_child(line_no, code)
                if created:
                    stats.nodes_created += 1
                line_node.count += 1
                current_line = line_node
                stats.max_depth = max(stats.max_depth, 2 * len(file_stack) - 1)
                continue

            # Return to a parent frame if the file is already on the stack.
            found_idx = None
            for idx in range(len(file_stack) - 1, -1, -1):
                if file_stack[idx].name == file_name:
                    found_idx = idx
                    break
            if found_idx is not None:
                file_stack = file_stack[: found_idx + 1]
                line_node, created = file_stack[-1].get_line_child(line_no, code)
                if created:
                    stats.nodes_created += 1
                line_node.count += 1
                current_line = line_node
                stats.max_depth = max(stats.max_depth, 2 * len(file_stack) - 1)
                continue

            parent_line = current_line
            if parent_line is None:
                parent_line, created = file_stack[-1].get_line_child(line_no, code)
                if created:
                    stats.nodes_created += 1
                parent_line.count += 1
                current_line = parent_line

            child_file, created = parent_line.get_file_child(file_name)
            if created:
                stats.nodes_created += 1
            child_file.count += 1
            file_stack.append(child_file)

            line_node, created = child_file.get_line_child(line_no, code)
            if created:
                stats.nodes_created += 1
            line_node.count += 1
            current_line = line_node
            stats.max_depth = max(stats.max_depth, 2 * len(file_stack) - 1)

    return root, stats


def render_node(
    node: Node,
    depth: int,
    max_depth: Optional[int],
    min_count: int,
    open_depth: int,
) -> str:
    children = [
        child for child in node.iter_children() if child.count >= min_count
    ]
    if node.kind == "file":
        safe_name = html.escape(node.name)
        label = (
            f"<span class=\"name\">{safe_name}</span> "
            f"<span class=\"count\">({node.count})</span>"
        )
    else:
        code_text = html.escape(node.code) if node.code else ""
        line_label = str(node.line_no) if node.line_no is not None else "?"
        label_parts = [f"<span class=\"line-no\">L{line_label}</span>"]
        if code_text:
            label_parts.append(f"<span class=\"code\">{code_text}</span>")
        label_parts.append(f"<span class=\"count\">({node.count})</span>")
        label = " ".join(label_parts)

    if max_depth is not None and depth >= max_depth:
        return (
            f"<div class=\"node leaf {node.kind}\">"
            f"{label}</div>"
        )

    if not children:
        return (
            f"<div class=\"node leaf {node.kind}\">"
            f"{label}</div>"
        )

    open_attr = " open data-default-open" if depth < open_depth else ""
    parts = [
        f"<details class=\"node {node.kind}\"{open_attr}><summary>{label}</summary>"
    ]
    parts.append("<div class=\"children\">")
    for child in children:
        parts.append(
            render_node(child, depth + 1, max_depth, min_count, open_depth)
        )
    parts.append("</div></details>")
    return "\n".join(parts)


def build_html(
    root: Node,
    stats: Stats,
    trace_path: str,
    max_depth: Optional[int],
    min_count: int,
) -> str:
    title = f"Trace Call Tree - {os.path.basename(trace_path)}"
    root_html = render_node(
        root, 0, max_depth, min_count, open_depth=2
    )
    summary = (
        f"File events: {stats.file_events} | "
        f"Nodes: {stats.nodes_created} | "
        f"Max depth: {stats.max_depth}"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f2ec;
      --panel: #fffaf3;
      --ink: #2f2a25;
      --muted: #6f655c;
      --accent: #b66d3b;
      --line: #e6d8c7;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Sans 3", "Noto Sans", sans-serif;
      line-height: 1.4;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 32px auto 48px;
      padding: 0 24px;
    }}
    h1 {{
      font-size: 1.4rem;
      margin: 0 0 6px;
      letter-spacing: 0.01em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 18px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
    }}
    .controls input[type="search"] {{
      flex: 1 1 320px;
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffdf8;
      font-size: 0.95rem;
      color: var(--ink);
      box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    .controls input[type="search"]::placeholder {{
      color: #a29589;
    }}
    .meta-small {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .tree {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px 18px;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
    }}
    details.node {{
      margin: 4px 0;
      border-left: 2px solid var(--line);
      padding-left: 12px;
    }}
    .node > summary {{
      cursor: pointer;
      list-style: none;
      font-weight: 600;
    }}
    .node > summary::-webkit-details-marker {{
      display: none;
    }}
    .node > summary::before {{
      content: ">";
      color: var(--accent);
      display: inline-block;
      width: 1em;
      margin-right: 6px;
      transform: rotate(0deg);
      transition: transform 0.15s ease;
    }}
    details[open] > summary::before {{
      transform: rotate(90deg);
    }}
    .children {{
      margin-left: 14px;
      margin-top: 6px;
    }}
    .leaf {{
      margin: 4px 0;
      padding-left: 26px;
      position: relative;
    }}
    .leaf::before {{
      content: "*";
      color: var(--accent);
      position: absolute;
      left: 10px;
    }}
    details.node.line {{
      border-left-style: dashed;
    }}
    .node.line > summary {{
      font-weight: 500;
    }}
    .node.leaf.line {{
      color: #51463d;
    }}
    .line-no {{
      color: var(--muted);
      font-weight: 600;
    }}
    .code {{
      font-family: "IBM Plex Mono", "Source Code Pro", ui-monospace, monospace;
      font-size: 0.9rem;
      color: #4a3f35;
      background: #f3e6d6;
      padding: 1px 6px;
      border-radius: 6px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .count {{
      color: var(--muted);
      font-weight: 500;
    }}
    .hidden {{
      display: none !important;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <div class="meta">{html.escape(trace_path)}<br />{html.escape(summary)}</div>
    <div class="controls">
      <input id="search" type="search" placeholder="Filter files or code..." spellcheck="false" />
      <div id="search-meta" class="meta-small"></div>
    </div>
    <div class="tree">
      {root_html}
    </div>
  </div>
  <script>
    const input = document.getElementById("search");
    const meta = document.getElementById("search-meta");
    const allNodes = Array.from(document.querySelectorAll(".node"));
    const detailNodes = Array.from(document.querySelectorAll("details.node"));
    const leafNodes = Array.from(document.querySelectorAll(".node.leaf"));
    const searchIndex = new Map();

    allNodes.forEach((node) => {{
      let text = "";
      if (node.tagName === "DETAILS") {{
        const summary = node.querySelector(":scope > summary");
        text = summary ? summary.textContent || "" : node.textContent || "";
      }} else {{
        text = node.textContent || "";
      }}
      searchIndex.set(node, text.toLowerCase());
    }});

    function resetTree() {{
      allNodes.forEach((node) => {{
        node.classList.remove("hidden");
        node.removeAttribute("data-match");
      }});
      detailNodes.forEach((node) => {{
        node.open = node.hasAttribute("data-default-open");
      }});
      meta.textContent = "";
    }}

    function applyFilter() {{
      const query = input.value.trim().toLowerCase();
      if (!query) {{
        resetTree();
        return;
      }}

      allNodes.forEach((node) => {{
        node.classList.remove("hidden");
        node.removeAttribute("data-match");
      }});

      allNodes.forEach((node) => {{
        const text = searchIndex.get(node) || "";
        if (text.includes(query)) {{
          node.setAttribute("data-match", "1");
        }}
      }});

      let visibleCount = 0;
      detailNodes.forEach((node) => {{
        const selfMatch = node.getAttribute("data-match") === "1";
        const descendantMatch = node.querySelector(".node[data-match=\"1\"]");
        if (selfMatch || descendantMatch) {{
          visibleCount += 1;
          if (descendantMatch) {{
            node.open = true;
          }}
        }} else {{
          node.classList.add("hidden");
        }}
      }});

      leafNodes.forEach((node) => {{
        const selfMatch = node.getAttribute("data-match") === "1";
        if (selfMatch) {{
          visibleCount += 1;
        }} else {{
          node.classList.add("hidden");
        }}
      }});

      meta.textContent = visibleCount ? `${{visibleCount}} matches` : "No matches";
    }}

    input.addEventListener("input", applyFilter);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Python trace output into a hierarchical HTML call tree."
    )
    parser.add_argument("trace", help="Path to the trace output file.")
    parser.add_argument(
        "-o",
        "--out",
        help="Output HTML path (default: <trace>.html).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit the rendered tree depth (0 = only root).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Hide child nodes entered fewer than N times.",
    )
    parser.add_argument(
        "--basename",
        action="store_true",
        help="Show only the basename of file paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = args.out or f"{args.trace}.html"

    root, stats = parse_trace(args.trace, args.basename)
    if root is None:
        raise SystemExit("No trace file events found; output not generated.")

    html_text = build_html(
        root,
        stats,
        trace_path=os.path.abspath(args.trace),
        max_depth=args.max_depth,
        min_count=args.min_count,
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
