"""
Nodes in concentric rings by strength; returns positions and ring radii for edge routing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

RING0_RADIUS = 0.5
RING_SPACING = 0.75
RING0_INNER = 0.12
RING2_RADIUS = 0.95
N_RINGS = 3

TOTAL_ANGULAR_SPAN = 5.5
GAP_BETWEEN_GROUPS = 0.2
JITTER_SCALE = 0.12
OUTER_RING_ANGLE_BIAS = -0.12


def _str_hash(s: str) -> int:
    """Deterministic 32-bit hash for angular jitter."""
    h = 0
    for b in s.encode("utf-8"):
        h = (h * 31 + b) & 0xFFFFFFFF
    return h


def _node_jitter(node: str) -> float:
    """Deterministic scalar in [0, 1) from node id."""
    h = _str_hash(str(node)) & 0x7FFFFFFF
    return (h % 1000) / 1000.0


def onion_pos(
    graph: "nx.DiGraph",
    n_rings: int = 3,
    node_to_strength: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Tuple[float, float]], List[float]]:
    """Return (pos, ring_radii); default strength sums outgoing edge abs_weight per node."""
    nodes = list(graph.nodes())
    if not nodes:
        return {}, []

    n_nodes = len(nodes)
    n_rings = min(max(1, n_rings), N_RINGS)

    if node_to_strength is None:
        out_strength = {n: 0.0 for n in nodes}
        for u, v, d in graph.edges(data=True):
            out_strength[u] += float(d.get("abs_weight", 0.0))
        node_to_strength = out_strength

    sorted_nodes = sorted(nodes, key=lambda n: (node_to_strength.get(n, 0.0), str(n)), reverse=True)
    if n_rings == 3:
        ring_radii = [RING0_INNER, RING0_RADIUS, RING2_RADIUS]
    else:
        ring_radii = [RING0_RADIUS + RING_SPACING * i for i in range(n_rings)]

    rings: List[List[str]] = [[] for _ in range(n_rings)]
    if n_rings == 3:
        n_mid = max(0, (n_nodes - 2 + 1) // 2)
        rings[0] = [sorted_nodes[0], sorted_nodes[-1]]
        rings[1] = sorted_nodes[2 : 2 + n_mid] if n_mid > 0 else []
        rings[2] = [sorted_nodes[1]] + sorted_nodes[2 + n_mid : -1]
    else:
        mid = max(1, (n_nodes + 1) // 2)
        for i, node in enumerate(sorted_nodes):
            ring_idx = 0 if i < mid else min(1, n_rings - 1)
            rings[ring_idx].append(node)

    pos = {}
    two_pi = 2.0 * math.pi
    ring_start_offset = [0.0, 0.4, 0.2] if n_rings >= 3 else [0.0, 0.4]

    for ring_idx, ring_nodes in enumerate(rings):
        if not ring_nodes:
            continue
        r = ring_radii[ring_idx]
        start = ring_start_offset[ring_idx] if ring_idx < len(ring_start_offset) else 0.0

        n_groups = min(3, max(2, (len(ring_nodes) + 1) // 2))
        group_size = (len(ring_nodes) + n_groups - 1) // n_groups
        groups: List[List[str]] = []
        for g in range(n_groups):
            lo = g * group_size
            hi = min(lo + group_size, len(ring_nodes))
            if lo < hi:
                groups.append(ring_nodes[lo:hi])

        total_gaps = (len(groups) - 1) * GAP_BETWEEN_GROUPS
        span_for_arcs = TOTAL_ANGULAR_SPAN - total_gaps
        arc_per_node = span_for_arcs / len(ring_nodes) if ring_nodes else 0.0
        angle = start
        for g_idx, group in enumerate(groups):
            if g_idx > 0:
                angle += GAP_BETWEEN_GROUPS
            for node in group:
                jitter = (_node_jitter(node) - 0.5) * JITTER_SCALE
                step = arc_per_node * (1.0 + jitter)
                theta = (angle + step * 0.5) % two_pi
                if ring_idx == 2 and n_rings >= 3:
                    theta = (theta + OUTER_RING_ANGLE_BIAS) % two_pi
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pos[node] = (float(x), float(y))
                angle += step

    return pos, ring_radii


def draw_ring_circles(
    ax,
    ring_radii: List[float],
    color: str = "#cccccc",
    linewidth: float = 0.8,
    zorder: int = 0,
) -> None:
    """Concentric guide circles at ring_radii."""
    from matplotlib.patches import Circle

    for r in ring_radii:
        circle = Circle((0, 0), r, fill=False, edgecolor=color, linewidth=linewidth, zorder=zorder)
        ax.add_patch(circle)