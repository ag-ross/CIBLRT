"""
Square nodes, ring-band edge routing, typography constants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


SIDE_LENGTH_BASE = 0.033
SIDE_LENGTH_MAX = 0.132
NODE_EDGECOLOR = "#2B2B2B"
NODE_LINEWIDTH = 0.25

EDGE_COLOR = "#5BA3D0"
EDGE_ALPHA = 0.3
EDGE_KEEP_PERCENTILE = 75

FONT_SIZE_BODY = 6
FONT_SIZE_TICK = 5
FONT_SIZE_PANEL_LABEL = 8
TITLE_PAD = -4


def set_panel_title(ax: Any, text: str, *, fontsize: float = FONT_SIZE_BODY) -> None:
    """Title anchored to the top of the axes in axes coordinates."""
    ax.set_title(text, fontsize=fontsize, pad=0)
    ax.title.set_transform(ax.transAxes)
    ax.title.set_position((0.5, 1.0))
    ax.title.set_verticalalignment("bottom")


def _wrap_delta(delta: float) -> float:
    """Map angle delta to (-pi, pi]."""
    return float((delta + np.pi) % (2 * np.pi) - np.pi)


def _square_border_distance(side_length: float, direction_angle: float) -> float:
    """Distance from square center to border along a ray (axis-aligned square)."""
    c = abs(float(np.cos(direction_angle)))
    s = abs(float(np.sin(direction_angle)))
    denom = max(c, s, 1e-9)
    return (float(side_length) / 2.0) / denom


def _attach_point_on_square(
    node: Any,
    direction_vec: Sequence[float],
    pos: Dict[Any, Tuple[float, float]],
    node_sizes: Dict[Any, float],
    attach_pad: float = 0.0,
) -> Tuple[float, float]:
    """Point on the square border of the node in the given direction."""
    cx, cy = float(pos[node][0]), float(pos[node][1])
    dx, dy = float(direction_vec[0]), float(direction_vec[1])
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        ang = float(np.arctan2(cy, cx)) if (cx != 0 or cy != 0) else 0.0
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        norm = 1.0
    ux, uy = dx / norm, dy / norm
    ang = float(np.arctan2(uy, ux))
    side = float(node_sizes.get(node, SIDE_LENGTH_BASE))
    d = _square_border_distance(side, ang) + attach_pad
    return (cx + ux * d, cy + uy * d)


def _str_hash(x: Any) -> int:
    """Deterministic 32-bit hash for lane offsets."""
    h = 0
    for b in str(x).encode("utf-8"):
        h = (h * 31 + b) & 0xFFFFFFFF
    return h


def _edge_offset(i: Any, j: Any, n_buckets: int = 9, step: float = 0.04) -> float:
    """Per-edge lateral offset in data units."""
    h = (_str_hash(i) * 1315423911 + _str_hash(j) * 2654435761) & 0xFFFFFFFF
    idx = int(h % n_buckets)
    return (idx - (n_buckets - 1) / 2.0) * step


def data_coord_side_lengths(
    values: Dict[Any, float],
    base: float = SIDE_LENGTH_BASE,
    max_size: float = SIDE_LENGTH_MAX,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Dict[Any, float]:
    """Map scalar node weights to square side lengths in [base, max_size] (optional vmin/vmax)."""
    if not values:
        return {}
    vals = list(values.values())
    if vmin is not None and vmax is not None:
        min_v, max_v = float(vmin), float(vmax)
        if max_v <= min_v:
            min_v, max_v = min(vals), max(vals)
            if max_v <= min_v:
                min_v, max_v = 0.0, 1.0
    else:
        min_v, max_v = min(vals), max(vals)
        if max_v <= min_v:
            min_v, max_v = 0.0, 1.0
    out = {}
    for n, v in values.items():
        v_clamp = max(min_v, min(max_v, float(v)))
        if max_v > min_v:
            norm = (v_clamp - min_v) / (max_v - min_v)
        else:
            norm = 0.5
        out[n] = base + (max_size - base) * norm
    return out


def draw_square_nodes(
    ax,
    pos: Dict[Any, Tuple[float, float]],
    node_sizes: Dict[Any, float],
    node_colors: Dict[Any, str],
    *,
    edgecolor: str = NODE_EDGECOLOR,
    linewidth: float = NODE_LINEWIDTH,
    zorder: int = 10,
) -> None:
    """Axis-aligned squares; side lengths are in data coordinates."""
    for node in pos:
        x, y = pos[node][0], pos[node][1]
        s = float(node_sizes.get(node, SIDE_LENGTH_BASE))
        color = node_colors.get(node, "#3498db")
        rect = Rectangle(
            (x - s / 2, y - s / 2),
            s,
            s,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            antialiased=False,
        )
        ax.add_patch(rect)


def filter_edges_top_percent(
    edges: List[Tuple[Any, Any]],
    weights: List[float],
    percentile: float = EDGE_KEEP_PERCENTILE,
) -> Tuple[List[Tuple[Any, Any]], List[float]]:
    """Keep edges with weight at or above the given percentile; sorting is deterministic."""
    if not edges or not weights or len(edges) != len(weights):
        return (list(edges), list(weights))
    packed = list(zip(edges, weights))
    packed.sort(key=lambda x: (str(x[0][0]), str(x[0][1])))
    edges = [p[0] for p in packed]
    weights = [float(p[1]) for p in packed]
    packed = sorted(zip(edges, weights), key=lambda x: (-float(x[1]), str(x[0][0]), str(x[0][1])))
    edges = [p[0] for p in packed]
    weights = [float(p[1]) for p in packed]
    arr = np.asarray(weights, dtype=float)
    thresh = float(np.percentile(arr, percentile))
    keep = [i for i in range(len(edges)) if weights[i] >= thresh]
    return ([edges[i] for i in keep], [weights[i] for i in keep])


def _ring_index(r: float, ring_radii: List[float]) -> int:
    """Return ring index whose radius is closest to r."""
    if not ring_radii:
        return 0
    best = 0
    best_d = abs(r - ring_radii[0])
    for i in range(1, len(ring_radii)):
        d = abs(r - ring_radii[i])
        if d < best_d:
            best_d = d
            best = i
    return best


def draw_band_edges(
    ax,
    pos: Dict[Any, Tuple[float, float]],
    ring_radii: List[float],
    node_sizes: Dict[Any, float],
    edges: List[Tuple[Any, Any]],
    edge_widths: List[float],
    edge_colors: List[str],
    *,
    edge_color: Optional[str] = None,
    alpha: float = 0.85,
    lane_spacing: float = 0.04,
    n_lanes: int = 3,
) -> None:
    """Polar polylines between onion rings; multi-segment routes when rings differ."""
    max_route_radius = max(ring_radii) + 0.3 if ring_radii else 2.0
    n_rings = len(ring_radii) if ring_radii else 0
    band_0_1 = (float(ring_radii[0]) + float(ring_radii[1])) / 2.0 if n_rings >= 2 else (ring_radii[0] * 0.9 if ring_radii else 1.0)
    band_1_2 = (float(ring_radii[1]) + float(ring_radii[2])) / 2.0 if n_rings >= 3 else band_0_1

    def smoothstep(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def _lerp(a: float, b: float, u: np.ndarray) -> np.ndarray:
        return a + (b - a) * smoothstep(u)

    for idx, (u, v) in enumerate(edges):
        if u not in pos or v not in pos:
            continue
        xi, yi = pos[u][0], pos[u][1]
        xj, yj = pos[v][0], pos[v][1]
        r_i = float(np.hypot(xi, yi))
        r_j = float(np.hypot(xj, yj))
        a_i = float(np.arctan2(yi, xi))
        a_j = float(np.arctan2(yj, xj))
        dtheta = _wrap_delta(a_j - a_i)

        side_i = float(node_sizes.get(u, SIDE_LENGTH_BASE))
        side_j = float(node_sizes.get(v, SIDE_LENGTH_BASE))
        border_i = _square_border_distance(side_i, a_i)
        border_j = _square_border_distance(side_j, a_j)

        lane_off = _edge_offset(u, v, n_buckets=n_lanes, step=lane_spacing)

        if n_rings >= 3:
            ri = _ring_index(r_i, ring_radii)
            rj = _ring_index(r_j, ring_radii)
            if ri == rj:
                if ri == 0:
                    local = max(0.0, (r_i + r_j) / 2.0 - 0.15) + lane_off
                    route_r_list = [max(0.1, min(local, max_route_radius))]
                elif ri == 1:
                    route_r_list = [max(0.1, min(band_0_1 + lane_off, max_route_radius))]
                else:
                    route_r_list = [max(0.1, min(band_1_2 + lane_off, max_route_radius))]
            elif (ri == 0 and rj == 1) or (ri == 1 and rj == 0):
                route_r_list = [max(0.1, min(band_0_1 + lane_off, max_route_radius))]
            elif (ri == 0 and rj == 2) or (ri == 2 and rj == 0):
                route_r_list = [
                    max(0.1, min(band_0_1 + lane_off, max_route_radius)),
                    max(0.1, min(band_1_2 + lane_off, max_route_radius)),
                ]
            else:
                route_r_list = [max(0.1, min(band_1_2 + lane_off, max_route_radius))]
            start_band_r = route_r_list[0]
            end_band_r = route_r_list[-1]
            sign_i = -1.0 if start_band_r < r_i else 1.0
            sign_j = -1.0 if end_band_r < r_j else 1.0
            start_r = max(0.01, r_i + sign_i * border_i)
            end_r = max(0.01, r_j + sign_j * border_j)
        else:
            if not ring_radii or n_rings < 2:
                route_r = 0.9 * (ring_radii[0] if ring_radii else 1.0)
            else:
                route_r = band_0_1
            r_route = max(0.1, min(route_r + lane_off, max_route_radius))
            route_r_list = [r_route]
            start_r = max(0.01, r_i - border_i if r_route < r_i else r_i + border_i)
            end_r = max(0.01, r_j - border_j if r_route < r_j else r_j + border_j)

        n_pts = max(80, int(abs(dtheta) * 50))
        t = np.linspace(0.0, 1.0, n_pts, dtype=float)
        angles = a_i + dtheta * t

        if len(route_r_list) == 1:
            r_route = float(route_r_list[0])
            out_mask = t < 0.18
            in_mask = t > 0.82
            radii = np.full_like(t, r_route, dtype=float)
            radii[out_mask] = start_r + (r_route - start_r) * smoothstep(t[out_mask] / 0.18)
            radii[in_mask] = r_route + (end_r - r_route) * smoothstep((t[in_mask] - 0.82) / 0.18)
        else:
            rA, rB = float(route_r_list[0]), float(route_r_list[1])
            radii = np.full_like(t, rB, dtype=float)
            m1 = t < 0.16
            radii[m1] = _lerp(start_r, rA, t[m1] / 0.16)
            m2 = (t >= 0.16) & (t < 0.44)
            radii[m2] = rA
            m3 = (t >= 0.44) & (t < 0.56)
            radii[m3] = _lerp(rA, rB, (t[m3] - 0.44) / 0.12)
            m4 = (t >= 0.56) & (t < 0.84)
            radii[m4] = rB
            m5 = t >= 0.84
            radii[m5] = _lerp(rB, end_r, (t[m5] - 0.84) / 0.16)
        radii = np.clip(radii, 0.0, max_route_radius)

        line_x = (radii * np.cos(angles)).astype(float)
        line_y = (radii * np.sin(angles)).astype(float)

        p1 = np.array([line_x[1], line_y[1]], dtype=float)
        ci = np.array([xi, yi], dtype=float)
        start_pt = _attach_point_on_square(u, p1 - ci, pos, node_sizes)
        line_x[0], line_y[0] = start_pt[0], start_pt[1]

        p_prev = np.array([line_x[-2], line_y[-2]], dtype=float)
        cj = np.array([xj, yj], dtype=float)
        end_pt = _attach_point_on_square(v, p_prev - cj, pos, node_sizes)
        line_x[-1], line_y[-1] = end_pt[0], end_pt[1]

        w = edge_widths[idx] if idx < len(edge_widths) else 1.0
        color = (edge_color if edge_color is not None else
                 (edge_colors[idx] if idx < len(edge_colors) else "#888888"))
        max_w = max(edge_widths) if edge_widths else 1.0
        w_norm = (w / max_w) if max_w > 0 else 0.5
        linewidth = 0.2 + 0.8 * float(w_norm)
        line = Line2D(
            line_x.tolist(),
            line_y.tolist(),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
        ax.add_line(line)
