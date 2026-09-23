"""Plotting and mesh helpers shared by the SimKit tutorial notebooks.

The physics -- every ``XXX_system`` with its energy / gradient / hessian --
lives inside the notebooks and is built directly from ``simkit``. This module
only holds support code that is not the point of any tutorial:

* Mesh generators with no simkit / libigl equivalent (``tetrahedralized_grid``,
  ``ball_mesh_2d``).
* Matplotlib artists, static plots and one-call animations. Animations return
  ``(fig, anim)``; save them with ``simkit.filesystem.save_animation`` and embed
  the saved file with :func:`embed_video`. Save static figures with
  ``simkit.filesystem.save_figure``.
* Validation visuals every notebook uses: :func:`plot_mesh` and
  :func:`plot_scalar_field`.
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# Mesh generators
# =============================================================================

def tetrahedralized_grid(nx, ny, nz, width=1.0, height=1.0, depth=1.0):
    """Tet-meshed rectangular brick (5-tet-per-hex, parity-flipped to match faces)."""
    xs = np.linspace(-width / 2.0, width / 2.0, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    zs = np.linspace(-depth / 2.0, depth / 2.0, nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    pattern_even = np.array([
        [0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6],
    ])
    pattern_odd = np.array([
        [0, 1, 2, 5], [0, 2, 3, 7], [0, 5, 7, 4], [2, 5, 6, 7], [0, 2, 7, 5],
    ])

    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel()

    def vid(i, j, k):
        return (i * ny + j) * nz + k

    corners = np.stack([
        vid(ii,     jj,     kk    ),
        vid(ii + 1, jj,     kk    ),
        vid(ii + 1, jj + 1, kk    ),
        vid(ii,     jj + 1, kk    ),
        vid(ii,     jj,     kk + 1),
        vid(ii + 1, jj,     kk + 1),
        vid(ii + 1, jj + 1, kk + 1),
        vid(ii,     jj + 1, kk + 1),
    ], axis=1)

    even_mask = ((ii + jj + kk) % 2) == 0
    tets = np.empty((corners.shape[0], 5, 4), dtype=corners.dtype)
    tets[even_mask] = corners[even_mask][:, pattern_even]
    tets[~even_mask] = corners[~even_mask][:, pattern_odd]
    return X, tets.reshape(-1, 4)


def ball_mesh_2d(radius=0.15, n_segments=48):
    """Triangulated 2D disk: 1 center + n_segments boundary vertices."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    boundary = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    X = np.vstack([np.zeros((1, 2)), boundary])
    rim = np.arange(1, n_segments + 1)
    T = np.stack([np.zeros(n_segments, dtype=int), rim, np.roll(rim, -1)], axis=1)
    return X, T



# =============================================================================
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.animation import FuncAnimation

# Embed reasonably long inline animations in notebooks.
plt.rcParams["animation.embed_limit"] = 100.0

# ---- shared colors ----------------------------------------------------------
MAT_COLORS = {
    "Linear":      "#d62728",   # red    - linear elasticity
    "ARAP":        "#1f77b4",   # blue   - as-rigid-as-possible
    "Neo-Hookean": "#2ca02c",   # green  - (stable) neo-hookean
}
SOLVER_COLORS = {
    "Newton":           "#1f77b4",
    "Gradient Descent": "#d62728",
}
MESH_FACE = "#9ecae1"   # light blue fill
MESH_EDGE = "#08519c"   # dark blue edges
TRI_FACE  = "#a1d99b"   # light green fill
TRI_EDGE  = "#00441b"   # dark green edges
REST_EDGE = "#bdbdbd"   # gray ghost of the rest shape
HANDLE_C  = "#e6550d"   # orange handle marker
PIN_C     = "#3182bd"   # blue pin marker


# ---- axis setup -------------------------------------------------------------

def setup_axes(ax, xlim=None, ylim=None, title=None, equal=True, grid=True):
    """Standard 2D scene axis: equal aspect, light grid, optional limits."""
    if equal:
        ax.set_aspect("equal")
    if grid:
        ax.grid(True, color="0.9", linewidth=0.8, zorder=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)
    return ax


def auto_limits(states, pad=0.4):
    """Bounding box over every frame, padded -- so the camera never clips."""
    allpts = np.concatenate([np.asarray(s) for s in states], axis=0)
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    return (float(lo[0] - pad), float(hi[0] + pad)), (float(lo[1] - pad), float(hi[1] + pad))


# ---- artists (each has .update(U)) ------------------------------------------

class TriangleArtist:
    """A single filled triangle + its vertices, optionally ghosting the rest
    shape. ``update(U)`` moves it to a new deformed state."""

    def __init__(self, ax, U, facecolor=TRI_FACE, edgecolor=TRI_EDGE, lw=2.5,
                 rest=None, vertex_color=TRI_EDGE, vertex_size=70, zorder=2):
        U = np.asarray(U, dtype=float)
        self.rest_poly = None
        if rest is not None:
            self.rest_poly = Polygon(np.asarray(rest), closed=True, fill=False,
                                     edgecolor=REST_EDGE, lw=1.5, linestyle="--",
                                     zorder=zorder)
            ax.add_patch(self.rest_poly)
        self.poly = Polygon(U, closed=True, facecolor=facecolor,
                            edgecolor=edgecolor, lw=lw, zorder=zorder + 1, alpha=0.9)
        ax.add_patch(self.poly)
        self.verts = ax.scatter(U[:, 0], U[:, 1], s=vertex_size, color=vertex_color,
                                zorder=zorder + 2)

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.poly.set_xy(U)
        self.verts.set_offsets(U)


class PolyMeshArtist:
    """A triangulated mesh drawn as a ``PolyCollection``. ``update(U)`` moves
    every triangle. Used for the cantilever-beam tutorials."""

    def __init__(self, ax, U, T, facecolor=MESH_FACE, edgecolor=MESH_EDGE,
                 lw=1.0, zorder=2, alpha=0.95):
        self.T = np.asarray(T)
        U = np.asarray(U, dtype=float)
        self.coll = PolyCollection([U[f] for f in self.T], facecolors=facecolor,
                                   edgecolors=edgecolor, linewidths=lw,
                                   zorder=zorder, alpha=alpha)
        ax.add_collection(self.coll)

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.coll.set_verts([U[f] for f in self.T])


SPRING_EDGE = "#08519c"   # dark blue spring segments
SPRING_NODE = "#e6550d"   # orange masses
SPRING_REST = "#bdbdbd"   # gray ghost of the rest shape


class EdgeArtist:
    """A mass-spring network drawn as line segments (the springs) plus a
    scatter of point masses (the nodes). ``update(U)`` moves everything to a
    new state. Optionally ghosts the rest shape so deformation is obvious.
    Used by the mass-spring tutorial."""

    def __init__(self, ax, U, E, *, edgecolor=SPRING_EDGE, node_color=SPRING_NODE,
                 lw=2.5, node_size=60, rest=None, zorder=2):
        self.E = np.asarray(E)
        U = np.asarray(U, dtype=float)
        if rest is not None:
            rest = np.asarray(rest, dtype=float)
            ax.add_collection(LineCollection(
                [rest[e] for e in self.E], colors=SPRING_REST, linewidths=1.5,
                linestyles="--", zorder=zorder))
        self.coll = LineCollection([U[e] for e in self.E], colors=edgecolor,
                                   linewidths=lw, zorder=zorder + 1)
        ax.add_collection(self.coll)
        self.nodes = ax.scatter(U[:, 0], U[:, 1], s=node_size, color=node_color,
                                zorder=zorder + 2)

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.coll.set_segments([U[e] for e in self.E])
        self.nodes.set_offsets(U)


def format_F(F):
    """Pretty 2x2 deformation-gradient string for an in-figure text box."""
    F = np.asarray(F).reshape(2, 2)
    return ("$F$ =\n"
            f"[{F[0,0]:+5.2f}  {F[0,1]:+5.2f}]\n"
            f"[{F[1,0]:+5.2f}  {F[1,1]:+5.2f}]\n"
            f"det $F$ = {np.linalg.det(F):+5.2f}")


def text_box(ax, s, loc="upper left"):
    """Monospace text box pinned to a corner of the axes (data-independent)."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va,
                   family="monospace", fontsize=11,
                   bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


# ---- progressive line plot (the "plot drawn as the scene happens") ----------

class TracePlot:
    """One or more curves revealed progressively. Pre-computes the full series,
    then ``update(i)`` shows samples ``0..i`` plus a moving marker at ``i``.
    """

    def __init__(self, ax, xs, series, colors=None, xlabel="", ylabel="",
                 logy=False, title=None, lw=2.0, ylim=None):
        self.ax = ax
        self.xs = np.asarray(xs, dtype=float)
        self.series = {k: np.asarray(v, dtype=float) for k, v in series.items()}
        colors = colors or {}
        self.lines, self.marks = {}, {}
        for name, ys in self.series.items():
            c = colors.get(name)
            (ln,) = ax.plot([], [], color=c, lw=lw, label=name, zorder=2)
            (mk,) = ax.plot([], [], "o", color=ln.get_color(), ms=7, zorder=3)
            self.lines[name] = ln
            self.marks[name] = mk
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, color="0.9", linewidth=0.8)
        if ylim is not None:                 # explicit range (e.g. a flat-zero curve)
            xmn, xmx = float(self.xs.min()), float(self.xs.max())
            self.ax.set_xlim(xmn, xmx + 1e-12 if xmx == xmn else xmx)
            self.ax.set_ylim(*ylim)
        else:
            self._set_limits(logy)
        if len(self.series) > 1:
            ax.legend(loc="best", fontsize=9)

    def _set_limits(self, logy):
        xmn, xmx = float(self.xs.min()), float(self.xs.max())
        self.ax.set_xlim(xmn, xmx + 1e-12 if xmx == xmn else xmx)
        allv = np.concatenate([v[np.isfinite(v)] for v in self.series.values()])
        if logy:
            pos = allv[allv > 0]
            lo = pos.min() / 3 if pos.size else 1e-8
            hi = allv.max() * 3 if allv.size else 1.0
        else:
            span = (allv.max() - allv.min()) if allv.size else 1.0
            span = span if span > 0 else 1.0
            lo, hi = allv.min() - 0.08 * span, allv.max() + 0.08 * span
        self.ax.set_ylim(lo, hi)

    def update(self, i):
        j = i + 1
        for name, ys in self.series.items():
            self.lines[name].set_data(self.xs[:j], ys[:j])
            self.marks[name].set_data([self.xs[i]], [ys[i]])


# ---- high-level one-call animations -----------------------------------------

def animate_deformation(states, F_list, rest, *, lims=None, fps=20,
                        title="Deformation gradient", interval=None):
    """Triangle deforming with a live 2x2 ``F`` read-out (tutorial 1).

    ``states[i]`` is the deformed triangle, ``F_list[i]`` its deformation
    gradient. The rest shape is ghosted so translations are obvious.
    Returns ``(fig, anim)``.
    """
    if lims is None:
        xlim, ylim = auto_limits(list(states) + [rest])
    else:
        xlim, ylim = lims
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    setup_axes(ax, xlim, ylim, title=title)
    tri = TriangleArtist(ax, states[0], rest=rest)
    txt = text_box(ax, format_F(F_list[0]))

    def update(i):
        tri.update(states[i])
        txt.set_text(format_F(F_list[i]))
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_scene_energy(states, xs, series, *, scene="triangle", T=None,
                         rest=None, lims=None, xlabel="", ylabel="energy",
                         colors=None, logy=False, fps=20, title=None,
                         scene_title="", interval=None, energy_ylim=None):
    """Left: a deforming scene. Right: energy curve(s) traced in lock-step
    (tutorial 2). ``scene`` is ``"triangle"`` or ``"mesh"`` (needs ``T``).
    Pass ``energy_ylim`` to fix the energy-panel y-range (e.g. a flat-zero
    curve that would otherwise auto-zoom into numerical noise).
    Returns ``(fig, anim)``.
    """
    colors = colors or MAT_COLORS
    if lims is None:
        xlim, ylim = auto_limits(states)
    else:
        xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    setup_axes(axL, xlim, ylim, title=scene_title)
    if scene == "mesh":
        art = PolyMeshArtist(axL, states[0], T)
    else:
        art = TriangleArtist(axL, states[0], rest=rest)
    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel,
                      ylabel=ylabel, logy=logy, title=title, ylim=energy_ylim)

    def update(i):
        art.update(states[i])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_mesh(states, T, *, lims=None, fps=20, title="", pin_pts=None,
                 handle_traj=None, target_pts=None, interval=None,
                 figsize=(7, 4)):
    """A deforming triangle mesh, with optional pinned-vertex markers and
    moving handle / target markers (tutorials 4-6). Returns ``(fig, anim)``.

    ``handle_traj[i]`` (if given) is the handle position(s) at frame ``i`` and
    ``target_pts[i]`` the goal(s); each may be a single ``(2,)`` point or an
    ``(k, 2)`` array of several points (e.g. a whole pinned edge).
    """
    if lims is None:
        xlim, ylim = auto_limits(states)
    else:
        xlim, ylim = lims
    fig, ax = plt.subplots(figsize=figsize)
    setup_axes(ax, xlim, ylim, title=title)
    mesh = PolyMeshArtist(ax, states[0], T)
    if pin_pts is not None and len(pin_pts):
        ax.scatter(np.asarray(pin_pts)[:, 0], np.asarray(pin_pts)[:, 1],
                   s=55, color=PIN_C, marker="s", zorder=5, label="pinned")
    hdl = tgt = None
    if handle_traj is not None:
        (hdl,) = ax.plot([], [], "o", color=HANDLE_C, ms=11, zorder=6, label="handle")
    if target_pts is not None:
        (tgt,) = ax.plot([], [], "x", color="0.3", ms=10, mew=2.5, zorder=6,
                         label="target")
    if pin_pts is not None or handle_traj is not None:
        ax.legend(loc="upper right", fontsize=9)

    def update(i):
        mesh.update(states[i])
        if hdl is not None:
            p = np.atleast_2d(handle_traj[i])
            hdl.set_data(p[:, 0], p[:, 1])
        if tgt is not None:
            p = np.atleast_2d(target_pts[i])
            tgt.set_data(p[:, 0], p[:, 1])
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


# ---- static plots -----------------------------------------------------------

def line_plot(xs, series, *, xlabel="", ylabel="", colors=None, logy=False,
              title=None, figsize=(6.5, 4.5), markers=False, ax=None, ylim=None):
    """Quick multi-series line plot with the shared material colors. Returns
    ``(fig, ax)``. ``series`` maps a label to a y-array over ``xs``. Pass an
    explicit ``ylim`` to stop matplotlib from zooming into numerical noise
    (e.g. a curve that is constant zero)."""
    colors = colors or MAT_COLORS
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    style = "-o" if markers else "-"
    for name, ys in series.items():
        ax.plot(xs, ys, style, color=colors.get(name), lw=2, ms=4, label=name)
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, color="0.9", linewidth=0.8)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=9)
    return fig, ax


def deformation_panels(cases, rest, *, lims=None, ncols=4, figsize=None):
    """Row of static triangle panels, each captioned with its name and ``F``
    (tutorial 1 summary). ``cases`` is a list of ``(name, U, F)``."""
    n = len(cases)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if lims is None:
        xlim, ylim = auto_limits([U for _, U, _ in cases] + [rest])
    else:
        xlim, ylim = lims
    figsize = figsize or (3.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for k, (name, U, F) in enumerate(cases):
        ax = axes[k // ncols][k % ncols]
        setup_axes(ax, xlim, ylim, title=name)
        TriangleArtist(ax, U, rest=rest)
        text_box(ax, format_F(F), loc="lower right")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    return fig, axes


def convergence_plot(metrics, *, title="", figsize=(13, 4)):
    """Three side-by-side semilog panels -- energy gap, gradient norm, Newton
    decrement -- one line per solver (tutorial 4).

    ``metrics`` maps a solver name to a dict with keys ``energy`` (per-iter
    objective), ``grad`` (gradient norm), and optionally ``decrement``.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # shared optimum so the energy gap is comparable across solvers
    e_min = min(np.min(m["energy"]) for m in metrics.values())
    panels = [
        ("energy gap  $E_k - E^\\star$", "energy"),
        ("gradient norm  $\\|\\nabla E_k\\|$", "grad"),
        ("Newton decrement  $\\lambda_k$", "decrement"),
    ]
    for ax, (lbl, key) in zip(axes, panels):
        for name, m in metrics.items():
            if key not in m:
                continue
            y = np.asarray(m[key], dtype=float)
            if key == "energy":
                y = y - e_min + 1e-16
            it = np.arange(len(y))
            ax.semilogy(it, np.maximum(y, 1e-16), "-o", ms=3,
                        color=SOLVER_COLORS.get(name), label=name)
        ax.set_xlabel("iteration")
        ax.set_title(lbl)
        ax.grid(True, which="both", color="0.9", linewidth=0.7)
        ax.legend(fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# Helpers for the dynamics / contact / complexity tutorials (6-9)
# =============================================================================

from matplotlib.patches import Circle

INTEGRATOR_COLORS = {
    "Forward Euler":  "#d62728",
    "Backward Euler": "#1f77b4",
    "BDF2":           "#2ca02c",
}
ENERGY_COLORS = {
    "elastic":   "#2ca02c",
    "kinetic":   "#1f77b4",
    "contact":   "#d62728",
    "potential": "#9467bd",
    "total":     "#000000",
}
BALL_FACE = "#f1a340"
BALL_EDGE = "#b35806"


def loglog_plot(xs, series, *, xlabel="", ylabel="", colors=None, title=None,
                ref_slopes=None, figsize=(6.5, 5), markers=True, ax=None):
    """Log-log plot, optionally overlaying dashed reference-slope guide lines.

    ``ref_slopes`` is a dict ``{label: slope}``; each becomes a dashed guide of
    that slope anchored near the data, to eyeball convergence order. Returns
    ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = colors or {}
    xs = np.asarray(xs, dtype=float)
    style = "-o" if markers else "-"
    for name, ys in series.items():
        ax.loglog(xs, np.asarray(ys, dtype=float), style, color=colors.get(name),
                  lw=2, ms=5, label=name)
    if ref_slopes:
        allv = np.concatenate([np.asarray(v, float) for v in series.values()])
        anchor = np.exp(np.mean(np.log(allv[allv > 0]))) if np.any(allv > 0) else 1.0
        xmid = np.exp(np.mean(np.log(xs)))
        for lbl, p in ref_slopes.items():
            guide = anchor * (xs / xmid) ** p
            ax.loglog(xs, guide, "--", color="0.5", lw=1.3, label=lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", color="0.9", linewidth=0.7)
    ax.legend(fontsize=9)
    return fig, ax


def _draw_floor(ax, floor_y, xlim, ylim):
    ax.axhline(floor_y, color="0.3", lw=2.0, zorder=1)
    ax.fill_between(list(xlim), ylim[0], floor_y, color="0.88", zorder=1)


def animate_dynamics(states, T, xs, series, *, lims, xlabel="time (s)",
                     ylabel="energy", colors=None, fps=20, title=None,
                     scene_title="", ball_centers=None, ball_radius=None,
                     sdf=False, floor_y=None, floor_sdf=False,
                     mesh_face=MESH_FACE, mesh_edge=MESH_EDGE,
                     figsize=(11, 5), interval=None):
    """Left: a deforming mesh with optional moving ball (and its signed-distance
    background) and/or a static floor (optionally with its own signed-distance
    background, ``floor_sdf``). Right: one or more energy curves traced over time
    in lock-step (tutorials 8-9). Returns ``(fig, anim)``."""
    colors = colors or ENERGY_COLORS
    xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    setup_axes(axL, xlim, ylim, title=scene_title)

    gx = np.linspace(xlim[0], xlim[1], 140)
    gy = np.linspace(ylim[0], ylim[1], 140)
    GX, GY = np.meshgrid(gx, gy)
    sdf_state = {"cs": None}

    def draw_sdf(c):
        if sdf_state["cs"] is not None:
            sdf_state["cs"].remove()
        D = np.sqrt((GX - c[0]) ** 2 + (GY - c[1]) ** 2) - ball_radius
        vmax = float(np.abs(D).max())
        sdf_state["cs"] = axL.contourf(GX, GY, D, levels=24, cmap="coolwarm",
                                       alpha=0.45, zorder=0, vmin=-vmax, vmax=vmax)

    if floor_sdf and floor_y is not None:
        # static signed distance to the ground plane: phi = y - floor_y (blue below)
        D = GY - floor_y
        vmax = float(np.abs(D).max())
        axL.contourf(GX, GY, D, levels=24, cmap="coolwarm", alpha=0.45, zorder=0,
                     vmin=-vmax, vmax=vmax)
        axL.axhline(floor_y, color="0.3", lw=2.0, zorder=1)
    elif floor_y is not None:
        _draw_floor(axL, floor_y, xlim, ylim)
    mesh = PolyMeshArtist(axL, states[0], T, zorder=3, facecolor=mesh_face,
                          edgecolor=mesh_edge)
    ball = None
    if ball_centers is not None:
        if sdf:
            draw_sdf(ball_centers[0])
        ball = Circle(tuple(ball_centers[0]), ball_radius, facecolor=BALL_FACE,
                      edgecolor=BALL_EDGE, lw=1.5, zorder=4, alpha=0.9)
        axL.add_patch(ball)

    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel,
                      ylabel=ylabel, title=title)

    def update(i):
        mesh.update(states[i])
        if ball is not None:
            if sdf:
                draw_sdf(ball_centers[i])
            ball.center = (float(ball_centers[i][0]), float(ball_centers[i][1]))
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_meshes_grid(panels, *, lims=None, fps=20, figsize=None,
                        suptitle=None, floor_y=None, interval=None):
    """Several deforming meshes side by side, played in sync (tutorials 6, 9).

    ``panels`` is a list of dicts, each ``{"states": [...], "T": T, "title": str}``.
    Panels may have different meshes and different frame counts (shorter ones
    hold on their last frame). Returns ``(fig, anim)``.
    """
    npan = len(panels)
    if lims is None:
        allstates = [s for p in panels for s in p["states"]]
        xlim, ylim = auto_limits(allstates)
    else:
        xlim, ylim = lims
    figsize = figsize or (4.6 * npan, 4.2)
    fig, axes = plt.subplots(1, npan, figsize=figsize, squeeze=False)
    axes = axes[0]
    arts = []
    nframes = max(len(p["states"]) for p in panels)
    for ax, p in zip(axes, panels):
        setup_axes(ax, xlim, ylim, title=p.get("title", ""))
        if floor_y is not None:
            _draw_floor(ax, floor_y, xlim, ylim)
        arts.append(PolyMeshArtist(ax, p["states"][0], p["T"]))

    def update(i):
        for art, p in zip(arts, panels):
            art.update(p["states"][min(i, len(p["states"]) - 1)])
        return ()

    anim = FuncAnimation(fig, update, frames=nframes,
                         interval=interval or 1000 / fps, blit=False)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, anim


def plot_ball_sdf(center, radius, lims, *, U=None, T=None, figsize=(6, 5),
                  title="Signed distance to the ball"):
    """Static filled-contour plot of the signed distance field
    ``phi(x) = |x - center| - radius`` (negative inside the ball), with the
    ball surface as the zero level set. Optionally overlays a mesh. Returns
    ``(fig, ax)``."""
    xlim, ylim = lims
    fig, ax = plt.subplots(figsize=figsize)
    setup_axes(ax, xlim, ylim, title=title)
    gx = np.linspace(xlim[0], xlim[1], 220)
    gy = np.linspace(ylim[0], ylim[1], 220)
    GX, GY = np.meshgrid(gx, gy)
    D = np.sqrt((GX - center[0]) ** 2 + (GY - center[1]) ** 2) - radius
    vmax = float(np.abs(D).max())
    cs = ax.contourf(GX, GY, D, levels=30, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                     zorder=0, alpha=0.85)
    ax.contour(GX, GY, D, levels=[0.0], colors="k", linewidths=1.5, zorder=1)
    fig.colorbar(cs, ax=ax, label=r"signed distance  $\phi(x)=\|x-c\|-r$")
    if U is not None and T is not None:
        PolyMeshArtist(ax, U, T, zorder=3, alpha=0.85)
    ax.add_patch(Circle(tuple(center), radius, fill=False, ec=BALL_EDGE,
                        lw=1.5, zorder=4))
    return fig, ax


# =============================================================================
# Helpers for the mass-spring tutorial (11)
# =============================================================================

def animate_springs_energy(states, E, xs, series, *, rest=None, lims=None,
                           xlabel="", ylabel="energy", colors=None, fps=20,
                           title=None, scene_title="", interval=None,
                           pin_pts=None, energy_ylim=None):
    """Left: a mass-spring network deforming (with an optional ghosted rest
    shape). Right: one or more curves traced in lock-step. Mirror of
    :func:`animate_scene_energy` but for spring networks. Returns ``(fig, anim)``."""
    colors = colors or ENERGY_COLORS
    if lims is None:
        xlim, ylim = auto_limits(states if rest is None else list(states) + [rest])
    else:
        xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    setup_axes(axL, xlim, ylim, title=scene_title)
    art = EdgeArtist(axL, states[0], E, rest=rest)
    if pin_pts is not None and len(pin_pts):
        axL.scatter(np.asarray(pin_pts)[:, 0], np.asarray(pin_pts)[:, 1],
                    s=60, color=PIN_C, marker="s", zorder=6, label="pinned")
        axL.legend(loc="upper right", fontsize=9)
    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel,
                      ylabel=ylabel, title=title, ylim=energy_ylim)

    def update(i):
        art.update(states[i])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_springs_grid(panels, *, lims=None, fps=20, figsize=None,
                         suptitle=None, pin_pts=None, interval=None):
    """Several mass-spring networks side by side, played in sync. Mirror of
    :func:`animate_meshes_grid`. Each panel is a dict
    ``{"states": [...], "E": E, "title": str}``. Returns ``(fig, anim)``."""
    npan = len(panels)
    if lims is None:
        allstates = [s for p in panels for s in p["states"]]
        xlim, ylim = auto_limits(allstates)
    else:
        xlim, ylim = lims
    figsize = figsize or (4.6 * npan, 4.2)
    fig, axes = plt.subplots(1, npan, figsize=figsize, squeeze=False)
    axes = axes[0]
    arts = []
    nframes = max(len(p["states"]) for p in panels)
    for ax, p in zip(axes, panels):
        setup_axes(ax, xlim, ylim, title=p.get("title", ""))
        if pin_pts is not None and len(pin_pts):
            ax.scatter(np.asarray(pin_pts)[:, 0], np.asarray(pin_pts)[:, 1],
                       s=55, color=PIN_C, marker="s", zorder=6)
        arts.append(EdgeArtist(ax, p["states"][0], p["E"]))

    def update(i):
        for art, p in zip(arts, panels):
            art.update(p["states"][min(i, len(p["states"]) - 1)])
        return ()

    anim = FuncAnimation(fig, update, frames=nframes,
                         interval=interval or 1000 / fps, blit=False)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, anim


def phase_plot(trajectories, *, xlabel="position  q", ylabel="momentum  p",
               colors=None, title=None, figsize=(6.5, 6), start_marker=True):
    """Phase-space (position vs momentum) portrait, one curve per integrator.

    ``trajectories`` maps a name to an ``(N, 2)`` array of ``(q, p)`` samples.
    Returns ``(fig, ax)``."""
    colors = colors or INTEGRATOR_COLORS
    fig, ax = plt.subplots(figsize=figsize)
    for name, qp in trajectories.items():
        qp = np.asarray(qp, dtype=float)
        ax.plot(qp[:, 0], qp[:, 1], "-", color=colors.get(name), lw=1.6,
                label=name, zorder=2)
        if start_marker:
            ax.plot([qp[0, 0]], [qp[0, 1]], "o", color=colors.get(name),
                    ms=7, zorder=3)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, color="0.92", linewidth=0.8)
    ax.legend(loc="best", fontsize=9)
    return fig, ax





# =============================================================================
# Validation visuals: meshes, scalar fields, embedded videos
# =============================================================================

def plot_mesh(ax, X, T, *, pins=None, handles=None, title=None, lims=None,
              facecolor=MESH_FACE, edgecolor=MESH_EDGE, lw=0.8, show_vertices=False):
    """Draw a 2D triangle mesh with its edges, marking pinned / handle vertices.

    ``pins`` and ``handles`` are vertex-index arrays. Returns the
    :class:`PolyMeshArtist` so the caller can ``update`` it.
    """
    X = np.asarray(X, dtype=float)
    art = PolyMeshArtist(ax, X, T, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
    if show_vertices:
        ax.scatter(X[:, 0], X[:, 1], s=6, color=edgecolor, zorder=3)
    if pins is not None and len(np.atleast_1d(pins)):
        P = X[np.atleast_1d(pins)]
        ax.scatter(P[:, 0], P[:, 1], s=28, marker="s", color=PIN_C, zorder=4, label="pinned")
    if handles is not None and len(np.atleast_1d(handles)):
        H = X[np.atleast_1d(handles)]
        ax.scatter(H[:, 0], H[:, 1], s=34, color=HANDLE_C, zorder=5, label="handle")
    setup_axes(ax, *(lims if lims is not None else auto_limits([X], pad=0.1)), title=title)
    if pins is not None or handles is not None:
        ax.legend(loc="upper right", fontsize=8)
    return art


def plot_scalar_field(ax, X, T, values, *, title=None, cmap="viridis", lims=None,
                      vmin=None, vmax=None, norm=None, edges=True, colorbar=True, label=None):
    """Color a 2D triangle mesh by a scalar field.

    ``values`` may be per-vertex (length ``len(X)``, smoothly interpolated) or
    per-element (length ``len(T)``, flat per triangle). Returns the mappable.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T)
    values = np.asarray(values, dtype=float).ravel()
    if values.shape[0] == X.shape[0] and values.shape[0] != T.shape[0]:
        m = ax.tripcolor(X[:, 0], X[:, 1], T, values, shading="gouraud", cmap=cmap,
                         vmin=vmin, vmax=vmax, norm=norm)
    else:
        m = PolyCollection([X[f] for f in T], array=values, cmap=cmap, norm=norm)
        if norm is None:
            m.set_clim(vmin, vmax)
        ax.add_collection(m)
    if edges:
        ax.add_collection(PolyCollection([X[f] for f in T], facecolors="none",
                                         edgecolors="0.3", linewidths=0.25, zorder=3))
    setup_axes(ax, *(lims if lims is not None else auto_limits([X], pad=0.1)),
               title=title, grid=False)
    if colorbar:
        ax.figure.colorbar(m, ax=ax, shrink=0.8, label=label)
    return m


def embed_video(path, *, loop=True, autoplay=True):
    """Inline, self-contained HTML5 ``<video>`` (base64) of a saved mp4.

    Pair with ``simkit.filesystem.save_animation``::

        path = simkit.filesystem.save_animation(anim, out_path, fps=FPS, copy_to=media_path)
        plt.close(fig)
        utils.embed_video(path)
    """
    import base64
    from IPython.display import HTML
    data = open(path, "rb").read()
    b64 = base64.b64encode(data).decode("ascii")
    attrs = "controls" + (" autoplay" if autoplay else "") + (" loop" if loop else "") + " muted"
    return HTML(f'<video {attrs} style="max-width:100%;">'
                f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')

# =============================================================================
# Notebook-contributed helpers (non-crucial support code, rule 6)
# =============================================================================


def states_at_times(ts, states, sample_times):
    """Pick, for each requested time, the first recorded state at or after it.

    ``ts`` is the increasing array of recorded times and ``states[k]`` the state at
    ``ts[k]``. Times past the end of the record hold the last state (for example a
    run that stopped early because it blew up). Returns a list, handy as the
    per-frame ``states`` of the ``animate_*`` helpers.
    """
    ts = np.asarray(ts, dtype=float)
    idx = np.searchsorted(ts, np.asarray(sample_times, dtype=float) - 1e-9 * (ts[-1] - ts[0] + 1.0))
    idx = np.clip(idx, 0, len(ts) - 1)
    return [states[i] for i in idx]


def residual_plot(series, xs=None, *, ax=None, colors=None, xlabel="", ylabel="",
                  title=None, ylim=None, markers=False, stagger=True, figsize=(6.5, 4.2)):
    """Semilog-y plot of residual-like curves (constraint violations, gradient norms).

    ``series`` maps a label to a y-array. ``xs`` is a shared x-array, or ``None`` to
    plot each series against its own index (e.g. per-iteration gradient norms of
    solves with different iteration counts). Exact zeros are masked instead of
    dragging the log axis down. With ``stagger`` (default), each later series is
    drawn dashed and thinner, so curves that coincide stay distinguishable.
    ``colors`` maps a label to a color. Returns ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = colors or {}
    styles = ["-", "--", ":", "-."]
    for k, (name, ys) in enumerate(series.items()):
        ys = np.asarray(ys, dtype=float)
        ys = np.where(ys > 0, ys, np.nan)
        x = np.arange(len(ys)) if xs is None else np.asarray(xs, dtype=float)
        ls = styles[k % len(styles)] if stagger else "-"
        lw = 2.6 - 0.6 * min(k, 2) if stagger else 2.0
        ax.semilogy(x, ys, ls, marker="o" if markers else None, ms=4, lw=lw,
                    color=colors.get(name), label=name)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", color="0.9", linewidth=0.7)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=9)
    return fig, ax


class ChamberMeshArtist:
    """A 2D triangle mesh with one closed chamber loop filled and outlined
    (pneumatic-actuation tutorial 015).

    ``loop`` is the chamber wall as an ordered vertex loop (for an edge loop
    ``E_chamber`` pass ``E_chamber[:, 0]``). ``rest`` (optional) draws the rest
    shape as a flat light-gray silhouette underneath; ``pins`` (optional vertex
    indices) are marked at their rest positions. ``update(U)`` moves the mesh
    and the chamber.
    """

    def __init__(self, ax, U, T, loop, *, rest=None, pins=None, facecolor="#cfe8df",
                 edgecolor="#2a7f62", chamber_color="#08519c", lw=0.5, wall_lw=2.0,
                 pin_size=20, zorder=2):
        from matplotlib.patches import Polygon as _Polygon
        U = np.asarray(U, dtype=float)
        self.loop = np.asarray(loop)
        if rest is not None:
            PolyMeshArtist(ax, rest, T, facecolor="0.88", edgecolor="none", lw=0,
                           zorder=zorder - 1, alpha=1.0)
        self.mesh = PolyMeshArtist(ax, U, T, facecolor=facecolor, edgecolor=edgecolor,
                                   lw=lw, zorder=zorder)
        self.chamber = _Polygon(U[self.loop], closed=True, facecolor=chamber_color,
                                edgecolor="none", alpha=0.18, zorder=zorder + 1)
        ax.add_patch(self.chamber)
        closed = np.vstack([U[self.loop], U[self.loop[:1]]])
        (self.wall,) = ax.plot(closed[:, 0], closed[:, 1], "-", color=chamber_color,
                               lw=wall_lw, zorder=zorder + 2, label="chamber wall")
        if pins is not None and len(np.atleast_1d(pins)):
            P = np.asarray(rest if rest is not None else U, dtype=float)[np.atleast_1d(pins)]
            ax.scatter(P[:, 0], P[:, 1], s=pin_size, marker="s", color=PIN_C,
                       zorder=zorder + 3, label="pinned")

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.mesh.update(U)
        self.chamber.set_xy(U[self.loop])
        closed = np.vstack([U[self.loop], U[self.loop[:1]]])
        self.wall.set_data(closed[:, 0], closed[:, 1])


def animate_chamber_trace(states, T, loop, xs, series, *, pins=None, lims=None, fps=20,
                          title="", xlabel="", ylabel="", trace_title=None, colors=None,
                          figsize=(12, 4.2)):
    """Left: a mesh whose chamber inflates (:class:`ChamberMeshArtist`). Right:
    curve(s) ``series`` over ``xs`` traced in lock-step (:class:`TracePlot`).
    ``states[i]`` is frame ``i``; ``pins`` are marked at ``states[0]``.
    Returns ``(fig, anim)``."""
    if lims is None:
        xlim, ylim = auto_limits(states, pad=0.25)
    else:
        xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    setup_axes(axL, xlim, ylim, title=title)
    art = ChamberMeshArtist(axL, states[0], T, loop, pins=pins, lw=0.5)
    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel, ylabel=ylabel,
                      title=trace_title)
    fig.tight_layout()

    def update(i):
        art.update(states[i])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)
    return fig, anim


def animate_chamber_panels(panels, *, lims=None, fps=20, suptitle=None, figsize=None, lw=0.25):
    """Several chamber meshes deforming side by side, played in sync (tutorial 015).

    ``panels`` is a list of dicts ``{"states": [...], "T": T, "loop": loop,
    "title": str}`` with optional ``"pins"`` (vertex indices). Shorter panels
    hold their last frame. Returns ``(fig, anim)``."""
    npan = len(panels)
    if lims is None:
        xlim, ylim = auto_limits([s for p in panels for s in p["states"]], pad=0.1)
    else:
        xlim, ylim = lims
    figsize = figsize or (4.5 * npan, 3.2)
    fig, axes = plt.subplots(1, npan, figsize=figsize, squeeze=False)
    arts = []
    for ax, p in zip(axes[0], panels):
        setup_axes(ax, xlim, ylim, title=p.get("title", ""))
        arts.append(ChamberMeshArtist(ax, p["states"][0], p["T"], p["loop"],
                                      pins=p.get("pins"), lw=lw, pin_size=10))
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    nframes = max(len(p["states"]) for p in panels)

    def update(i):
        for art, p in zip(arts, panels):
            art.update(p["states"][min(i, len(p["states"]) - 1)])
        return ()

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
    return fig, anim


def plot_spring_network(ax, X, E, *, node_values=None, edge_values=None, pins=None,
                        title=None, lims=None, cmap="viridis", vmin=None, vmax=None,
                        colorbar=True, label=None, node_size=30, lw=2.0,
                        edgecolor=SPRING_EDGE, node_color=SPRING_NODE):
    """Static picture of a 2D mass-spring network / polyline (springs ``E`` between points ``X``).

    Optionally colors the nodes by a per-node scalar ``node_values`` (NaN entries are drawn
    gray, e.g. nodes where a per-hinge quantity is undefined) or the springs by a per-edge
    scalar ``edge_values``, and marks pinned nodes (``pins``, vertex indices) with squares.
    Returns the color mappable (``None`` when nothing is colored).
    """
    X = np.asarray(X, dtype=float)
    E = np.asarray(E)
    segs = [X[e] for e in E]
    mappable = None
    if edge_values is not None:
        coll = LineCollection(segs, array=np.asarray(edge_values, float).ravel(), cmap=cmap,
                              linewidths=lw * 1.6, zorder=2)
        coll.set_clim(vmin, vmax)
        mappable = coll
    else:
        coll = LineCollection(segs, colors=edgecolor, linewidths=lw, zorder=2)
    ax.add_collection(coll)
    if node_values is not None:
        vals = np.asarray(node_values, dtype=float).ravel()
        undefined = ~np.isfinite(vals)
        if undefined.any():
            ax.scatter(X[undefined, 0], X[undefined, 1], s=node_size, color="0.6", zorder=3)
        mappable = ax.scatter(X[~undefined, 0], X[~undefined, 1], s=node_size, c=vals[~undefined],
                              cmap=cmap, vmin=vmin, vmax=vmax, zorder=3, edgecolors="none")
    elif node_size:
        ax.scatter(X[:, 0], X[:, 1], s=node_size, color=node_color, zorder=3)
    if pins is not None and len(np.atleast_1d(pins)):
        P = X[np.atleast_1d(pins)]
        ax.scatter(P[:, 0], P[:, 1], s=node_size * 1.8, marker="s", color=PIN_C, zorder=4,
                   label="pinned")
        ax.legend(loc="upper right", fontsize=8)
    setup_axes(ax, *(lims if lims is not None else auto_limits([X], pad=0.2)), title=title)
    if colorbar and mappable is not None:
        ax.figure.colorbar(mappable, ax=ax, shrink=0.8, label=label)
    return mappable


def regular_polygon(n, r=1.0, center=(0.0, 0.0)):
    """``n`` vertices at equal angles, counter-clockwise around ``center``; shape ``(n, 2)``.

    ``r`` is one radius or a length-``n`` array of per-vertex radii (alternating
    radii give a star). Unlike ``simkit.circle_outline`` the first vertex is not
    repeated at the end, so every edge has nonzero length.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = np.broadcast_to(np.asarray(r, dtype=float), (n,))
    return np.column_stack([center[0] + r * np.cos(theta),
                            center[1] + r * np.sin(theta)])


def draw_closed_outline(ax, X, E=None, *, color="#264653", fill=True, lw=2.5, ms=5,
                        alpha=0.18, arrows=False, zorder=3):
    """Draw a closed 2D outline (optionally filled), with its vertices as dots.

    ``E`` is the loop's ordered edge list (default ``[[0, 1], ..., [n-1, 0]]``);
    the loop is walked in the order ``E[:, 0]``. ``arrows=True`` puts an
    arrowhead on every edge to show the winding direction. Returns the loop's
    ``Line2D``.
    """
    X = np.asarray(X, dtype=float)
    order = np.arange(len(X)) if E is None else np.asarray(E)[:, 0]
    P = X[order]
    if fill:
        ax.fill(P[:, 0], P[:, 1], color=color, alpha=alpha, zorder=zorder - 2)
    loop = np.vstack([P, P[:1]])
    (line,) = ax.plot(loop[:, 0], loop[:, 1], "-o", color=color, lw=lw, ms=ms,
                      zorder=zorder)
    if arrows:
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            mid, d = 0.5 * (a + b), b - a
            ax.annotate("", xy=mid + 0.18 * d, xytext=mid - 0.18 * d,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    return line


def animate_outline_trace(states, xs, series, *, E=None, rest=None, color="#2a9d8f",
                          colors=None, dashed=(), logy=False, lims=None, fps=20,
                          xlabel="", ylabel="", title=None, scene_title="",
                          figsize=(11, 5), interval=None):
    """Left: a closed 2D outline moving through ``states``; right: curves traced in lock-step.

    The outline is filled, with small vertex dots; ``rest`` (if given) is ghosted
    as a dashed gray loop on top. ``E`` is the loop's ordered edge list (default
    ``[[0, 1], ..., [n-1, 0]]``). ``series`` maps a label to one value per frame
    (plotted over ``xs``); labels listed in ``dashed`` are drawn dashed.
    Returns ``(fig, anim)``.
    """
    order = np.arange(len(states[0])) if E is None else np.asarray(E)[:, 0]
    if lims is None:
        xlim, ylim = auto_limits(list(states) + ([rest] if rest is not None else []))
    else:
        xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    setup_axes(axL, xlim, ylim, title=scene_title)
    P0 = np.asarray(states[0], dtype=float)[order]
    fill = Polygon(P0, closed=True, facecolor=color, edgecolor="none", alpha=0.18, zorder=1)
    axL.add_patch(fill)
    (loop,) = axL.plot([], [], "-o", color=color, lw=2.5, ms=4, zorder=3)
    if rest is not None:
        R = np.asarray(rest, dtype=float)[order]
        R = np.vstack([R, R[:1]])
        axL.plot(R[:, 0], R[:, 1], "--", color="0.35", lw=1.2, zorder=4, label="start")
        axL.legend(loc="upper right", fontsize=9)
    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel, ylabel=ylabel,
                      logy=logy, title=title)
    for name in dashed:
        trace.lines[name].set_linestyle("--")
    if len(series) > 1:
        axR.legend(loc="best", fontsize=9)   # rebuild so the legend shows the dashes

    def update(i):
        P = np.asarray(states[i], dtype=float)[order]
        fill.set_xy(P)
        L = np.vstack([P, P[:1]])
        loop.set_data(L[:, 0], L[:, 1])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def best_time(fn, inner=1, rep=7):
    """Wall-clock seconds of one ``fn()`` call: the minimum over ``rep`` batches of
    ``inner`` back-to-back calls, divided by ``inner``.

    Taking the minimum filters out scheduler noise, and batching several calls
    makes sub-millisecond timings (small dense solves) meaningful.
    """
    import time
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - t0) / inner)
    return best


class TendonMeshArtist:
    """A 2D triangle mesh with actuator edges (tendons) drawn on top as thick
    colored segments with dotted endpoints (mass-spring actuation tutorial 016).

    ``E_act`` is an ``(m, 2)`` array of actuator edges; ``colors`` and ``labels``
    give one entry per edge. ``rest`` (optional) ghosts the rest mesh as light-gray
    outlines underneath; ``pins`` (optional vertex indices) are marked at their
    rest positions (``rest`` if given, else ``U``). ``update(U)`` moves the mesh
    and the tendons.
    """

    def __init__(self, ax, U, T, E_act, *, colors=None, labels=None, rest=None, pins=None,
                 facecolor="#cfe8df", edgecolor="#2a7f62", node_color="#e6550d", lw=0.6,
                 tendon_lw=3.0, node_size=6, pin_size=30, zorder=2):
        U = np.asarray(U, dtype=float)
        self.E_act = np.asarray(E_act)
        m = len(self.E_act)
        colors = colors or ["#c0392b"] * m
        labels = labels or [None] * m
        if rest is not None:
            PolyMeshArtist(ax, rest, T, facecolor="none", edgecolor="0.8", lw=0.5,
                           zorder=zorder - 1, alpha=1.0)
        self.mesh = PolyMeshArtist(ax, U, T, facecolor=facecolor, edgecolor=edgecolor,
                                   lw=lw, zorder=zorder)
        self.lines = []
        for (i, j), c, lbl in zip(self.E_act, colors, labels):
            (ln,) = ax.plot(U[[i, j], 0], U[[i, j], 1], "-", color=c, lw=tendon_lw,
                            zorder=zorder + 2, label=lbl)
            self.lines.append(ln)
        ends = self.E_act.ravel()
        (self.nodes,) = ax.plot(U[ends, 0], U[ends, 1], "o", color=node_color, ms=node_size,
                                zorder=zorder + 3)
        if pins is not None and len(np.atleast_1d(pins)):
            P = np.asarray(rest if rest is not None else U, dtype=float)[np.atleast_1d(pins)]
            ax.scatter(P[:, 0], P[:, 1], s=pin_size, marker="s", color=PIN_C,
                       zorder=zorder + 4, label="pinned")

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.mesh.update(U)
        for ln, (i, j) in zip(self.lines, self.E_act):
            ln.set_data(U[[i, j], 0], U[[i, j], 1])
        ends = self.E_act.ravel()
        self.nodes.set_data(U[ends, 0], U[ends, 1])


def animate_tendon_trace(states, T, E_act, xs, series, *, tendon_colors=None, pins=None,
                         lims=None, fps=20, title="", xlabel="", ylabel="", trace_title=None,
                         trace_colors=None, figsize=(12, 4.2)):
    """Left: a mesh bending under its tendons (:class:`TendonMeshArtist`). Right:
    curve(s) ``series`` over ``xs`` traced in lock-step (:class:`TracePlot`), with a
    gray zero line. ``states[i]`` is frame ``i``; ``pins`` are marked at
    ``states[0]``. Returns ``(fig, anim)``."""
    if lims is None:
        xlim, ylim = auto_limits(states, pad=0.25)
    else:
        xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    setup_axes(axL, xlim, ylim, title=title)
    art = TendonMeshArtist(axL, states[0], T, E_act, colors=tendon_colors, pins=pins)
    trace = TracePlot(axR, xs, series, colors=trace_colors, xlabel=xlabel, ylabel=ylabel,
                      title=trace_title)
    axR.axhline(0, color="0.6", lw=1.0, zorder=1)
    fig.tight_layout()

    def update(i):
        art.update(states[i])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)
    return fig, anim


def animate_tendon_panels(panels, *, lims=None, fps=20, suptitle=None, figsize=None,
                          tendon_colors=None):
    """Several tendon-driven meshes deforming side by side, played in sync
    (tutorial 016).

    ``panels`` is a list of dicts ``{"states": [...], "T": T, "E_act": E_act,
    "title": str}`` with optional ``"pins"`` (vertex indices). Shorter panels
    hold their last frame. Returns ``(fig, anim)``."""
    npan = len(panels)
    if lims is None:
        xlim, ylim = auto_limits([s for p in panels for s in p["states"]], pad=0.15)
    else:
        xlim, ylim = lims
    figsize = figsize or (4.6 * npan, 4.0)
    fig, axes = plt.subplots(1, npan, figsize=figsize, squeeze=False)
    arts = []
    for ax, p in zip(axes[0], panels):
        setup_axes(ax, xlim, ylim, title=p.get("title", ""))
        arts.append(TendonMeshArtist(ax, p["states"][0], p["T"], p["E_act"],
                                     colors=tendon_colors, pins=p.get("pins"),
                                     lw=0.5, tendon_lw=2.6, node_size=5, pin_size=20))
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    nframes = max(len(p["states"]) for p in panels)

    def update(i):
        for art, p in zip(arts, panels):
            art.update(p["states"][min(i, len(p["states"]) - 1)])
        return ()

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
    return fig, anim


def animate_mesh_arrows(states, T, arrow_base, arrow_vecs, *, lims=None, fps=20, title="",
                        facecolor=MESH_FACE, edgecolor=None, lw=0.3, arrow_color=HANDLE_C,
                        arrow_width=0.005, figsize=(6.5, 5.5), axis_off=True, interval=None):
    """A deforming triangle mesh plus a set of arrows that change every frame.

    ``states[i]`` is the mesh at frame ``i`` and ``arrow_vecs[i]`` the ``(k, 2)``
    arrows drawn at the fixed base points ``arrow_base`` ``(k, 2)``, in data units.
    ``edgecolor=None`` draws edges in the face color, so a fine mesh reads as a
    smooth shape. Returns ``(fig, anim)``.
    """
    xlim, ylim = lims if lims is not None else auto_limits(states, pad=0.12)
    fig, ax = plt.subplots(figsize=figsize)
    setup_axes(ax, xlim, ylim, title=title, grid=not axis_off)
    if axis_off:
        ax.axis("off")
    mesh = PolyMeshArtist(ax, states[0], T, facecolor=facecolor,
                          edgecolor=facecolor if edgecolor is None else edgecolor, lw=lw)
    base = np.asarray(arrow_base, dtype=float)
    v0 = np.asarray(arrow_vecs[0], dtype=float)
    quiv = ax.quiver(base[:, 0], base[:, 1], v0[:, 0], v0[:, 1], angles="xy",
                     scale_units="xy", scale=1, color=arrow_color, width=arrow_width, zorder=5)

    def update(i):
        mesh.update(states[i])
        v = np.asarray(arrow_vecs[i], dtype=float)
        quiv.set_UVC(v[:, 0], v[:, 1])
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_triangle_blend(V, corner_vecs, points, point_vecs, texts, *, fps=20,
                           figsize=(6.6, 6.0), corner_labels=None, facecolor=TRI_FACE,
                           edgecolor=TRI_EDGE, corner_color=PIN_C, point_color=HANDLE_C,
                           interval=None):
    """One triangle with a fixed arrow at each corner and a moving point with its own arrow.

    ``V`` is the ``(3, 2)`` triangle and ``corner_vecs`` its ``(3, 2)`` corner arrows.
    ``points[i]`` / ``point_vecs[i]`` are the moving point and its arrow at frame
    ``i``, and ``texts[i]`` is a caption shown above the triangle (for example the
    point's barycentric weights). Returns ``(fig, anim)``.
    """
    V = np.asarray(V, dtype=float)
    corner_vecs = np.asarray(corner_vecs, dtype=float)
    points = np.asarray(points, dtype=float)
    point_vecs = np.asarray(point_vecs, dtype=float)
    corner_labels = corner_labels or [rf"$u_{i}$" for i in range(3)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_patch(Polygon(V, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                         lw=2.5, alpha=0.55, zorder=1))
    for i in range(3):
        ax.quiver(V[i, 0], V[i, 1], corner_vecs[i, 0], corner_vecs[i, 1], angles="xy",
                  scale_units="xy", scale=1, color=corner_color, width=0.012, zorder=4)
        ax.scatter(*V[i], s=90, color=edgecolor, zorder=5)
        ax.annotate(corner_labels[i], V[i] + np.array([0.04, 0.04]), fontsize=13,
                    color=corner_color)
    xlim, ylim = auto_limits([V, V + corner_vecs, points + point_vecs], pad=0.2)
    setup_axes(ax, xlim, ylim, grid=False)
    ax.axis("off")
    quiv = ax.quiver(points[0, 0], points[0, 1], point_vecs[0, 0], point_vecs[0, 1],
                     angles="xy", scale_units="xy", scale=1, color=point_color,
                     width=0.015, zorder=7)
    (dot,) = ax.plot([points[0, 0]], [points[0, 1]], "o", color=point_color, ms=9, zorder=8)
    txt = ax.text(0.02, 1.02, texts[0], transform=ax.transAxes, va="bottom",
                  family="monospace", fontsize=11,
                  bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    def update(i):
        quiv.set_offsets(points[i])
        quiv.set_UVC(point_vecs[i, 0], point_vecs[i, 1])
        dot.set_data([points[i, 0]], [points[i, 1]])
        txt.set_text(texts[i])
        return ()

    anim = FuncAnimation(fig, update, frames=len(points),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_fields_trace(panels, T, xs, series, *, lims=None, cmap="viridis", norm=None,
                         vmin=None, vmax=None, cbar_label=None, pin_pts=None, colors=None,
                         xlabel="", ylabel="", title=None, ref_line=None, pose=None,
                         fps=20, figsize=None, interval=None):
    """Deforming meshes colored by a per-element scalar field, plus a lock-step trace.

    ``panels`` is a list of dicts ``{"states": [...], "fields": [...], "title": str}``:
    ``states[i]`` is the deformed mesh at frame ``i`` and ``fields[i]`` its per-element
    values, drawn with :func:`plot_scalar_field` (one colorbar, on the last mesh panel).
    The right-most axes traces ``series`` over ``xs`` with :class:`TracePlot`, with an
    optional dashed horizontal ``ref_line``. ``pin_pts`` (k, 2) are drawn as blue squares.
    ``pose`` shows that frame right away, so a still can be saved with ``save_figure``
    before the animation is written. Returns ``(fig, anim)``.
    """
    if lims is None:
        lims = auto_limits([s for p in panels for s in p["states"]], pad=0.2)
    npan = len(panels)
    figsize = figsize or (5.2 * npan + 5.0, 4.4)
    fig, axes = plt.subplots(1, npan + 1, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.0] * npan + [0.95]})
    colls = []
    for k, (ax, p) in enumerate(zip(axes[:-1], panels)):
        m = plot_scalar_field(ax, p["states"][0], T, p["fields"][0], title=p.get("title"),
                              cmap=cmap, lims=lims, vmin=vmin, vmax=vmax, norm=norm,
                              edges=False, colorbar=(k == npan - 1), label=cbar_label)
        m.set_edgecolor("0.25")
        m.set_linewidth(0.3)
        if pin_pts is not None and len(pin_pts):
            P = np.asarray(pin_pts)
            ax.scatter(P[:, 0], P[:, 1], s=26, color=PIN_C, marker="s", zorder=4)
        colls.append(m)
    trace = TracePlot(axes[-1], xs, series, colors=colors, xlabel=xlabel, ylabel=ylabel,
                      title=title)
    if ref_line is not None:
        axes[-1].axhline(ref_line, color="0.5", ls="--", lw=1.2, zorder=1)
    Tn = np.asarray(T)

    def update(i):
        for m, p in zip(colls, panels):
            m.set_verts([np.asarray(p["states"][i])[f] for f in Tn])
            m.set_array(np.asarray(p["fields"][i]).ravel())
        trace.update(i)
        return ()

    fig.tight_layout()
    if pose is not None:
        update(int(pose))
    anim = FuncAnimation(fig, update, frames=len(panels[0]["states"]),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_meshes_trace(panels, xs, series, *, lims=None, colors=None, xlabel="", ylabel="",
                         trace_title=None, logy=False, fps=20, figsize=None, suptitle=None,
                         interval=None):
    """Mesh panels played in sync with a progressively traced curve panel on the right.

    ``panels`` is a list of dicts ``{"states": [...], "T": T, "title": str}`` with optional
    ``"ghost"`` (per-frame reference states drawn filled in light gray underneath, e.g. a
    target pose), ``"color"`` (draw the mesh as unfilled edges of this color), ``"label"``
    and ``"ghost_label"`` (legend entries). ``series`` maps a label to a y-array over
    ``xs`` (one sample per frame, see :class:`TracePlot`). Returns ``(fig, anim)``; the
    animation also gets a ``set_frame(i)`` method that poses the figure at frame ``i``
    (for saving a still with ``save_figure`` after the video).
    """
    npan = len(panels)
    if lims is None:
        allstates = [s for p in panels for key in ("states", "ghost") for s in p.get(key, [])]
        xlim, ylim = auto_limits(allstates)
    else:
        xlim, ylim = lims
    figsize = figsize or (4.8 * (npan + 1), 4.4)
    fig, axes = plt.subplots(1, npan + 1, figsize=figsize,
                             gridspec_kw={"width_ratios": [1] * npan + [1.15]})
    arts = []
    for ax, p in zip(axes[:-1], panels):
        setup_axes(ax, xlim, ylim, title=p.get("title", ""))
        ghost = None
        if p.get("ghost") is not None:
            ghost = PolyMeshArtist(ax, p["ghost"][0], p["T"], facecolor="0.9", edgecolor="0.65", lw=0.4)
            ax.plot([], [], color="0.65", lw=2, label=p.get("ghost_label", "reference"))
        c = p.get("color")
        if c is None:
            mesh = PolyMeshArtist(ax, p["states"][0], p["T"])
        else:
            mesh = PolyMeshArtist(ax, p["states"][0], p["T"], facecolor="none", edgecolor=c, lw=1.0)
        if p.get("label"):
            ax.plot([], [], color=c or MESH_EDGE, lw=2, label=p["label"])
            ax.legend(loc="upper right", fontsize=8)
        arts.append((mesh, ghost, p))
    trace = TracePlot(axes[-1], xs, series, colors=colors, xlabel=xlabel, ylabel=ylabel,
                      logy=logy, title=trace_title)

    def update(i):
        for mesh, ghost, p in arts:
            mesh.update(p["states"][min(i, len(p["states"]) - 1)])
            if ghost is not None:
                ghost.update(p["ghost"][min(i, len(p["ghost"]) - 1)])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(xs), interval=interval or 1000 / fps, blit=False)
    anim.set_frame = update
    if suptitle:
        fig.suptitle(suptitle)
    return fig, anim


def animate_mode_shapes(X, T, B, *, titles=None, n_frames=36, amp=0.3, fps=20,
                        suptitle=None, figsize=None, pad=0.08, interval=None):
    """Play every column of a modal basis on its rest mesh, one panel per mode.

    ``B`` is ``(n*dim, k)`` with vertex-stacked columns. Mode ``j`` is shown as
    ``X + a_j sin(2 pi t) u_j`` over one period of ``n_frames`` frames, with
    ``a_j`` chosen so its largest vertex displacement is ``amp`` (mass-normalized
    modes otherwise differ wildly in size). Panels are stacked vertically and
    played in sync, over a light-gray ghost of the rest shape so frozen regions
    (which hide the ghost) and moving ones stand apart. Returns ``(fig, anim)``.
    """
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    k = B.shape[1]
    titles = titles or [f"mode {j + 1}" for j in range(k)]
    phases = np.sin(2.0 * np.pi * np.arange(n_frames) / n_frames)
    disps = []
    for j in range(k):
        d = np.asarray(B[:, j], dtype=float).reshape(n, dim)
        disps.append(d * (amp / (np.linalg.norm(d, axis=1).max() + 1e-12)))
    xlim, ylim = auto_limits([X + d for d in disps] + [X - d for d in disps], pad=pad)
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    figsize = figsize or (7.0, k * (6.6 * aspect + 0.35) + (0.5 if suptitle else 0.1))
    fig, axes = plt.subplots(k, 1, figsize=figsize, squeeze=False)
    arts = []
    for ax, d, title in zip(axes[:, 0], disps, titles):
        setup_axes(ax, xlim, ylim, grid=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, loc="left")
        ax.add_collection(PolyCollection([X[f] for f in T], facecolors="0.88",
                                         edgecolors="none", zorder=1))
        arts.append(PolyMeshArtist(ax, X, T, lw=0.4, zorder=2))
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()

    def update(i):
        for art, d in zip(arts, disps):
            art.update(X + phases[i] * d)
        return ()

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_mesh_labeled(states, T, labels, *, rest=None, pin_pts=None, lims=None, fps=20,
                         title="", lw=0.1, facecolor=MESH_FACE, edgecolor=MESH_EDGE,
                         pin_size=10, figsize=(6.5, 5.5), interval=None):
    """A deforming 2D triangle mesh with a per-frame caption (e.g. solver iterations).

    ``states[i]`` is the mesh at frame ``i`` and ``labels[i]`` its caption, shown in a
    monospace box in the upper-left corner. ``rest`` (optional) is ghosted in light gray
    underneath, and ``pin_pts`` (optional ``(k, 2)`` array) are drawn as small squares.
    Thin edges (``lw``) keep a fine mesh readable. Returns ``(fig, anim)``.
    """
    xlim, ylim = lims if lims is not None else auto_limits(states, pad=0.3)
    fig, ax = plt.subplots(figsize=figsize)
    setup_axes(ax, xlim, ylim, title=title)
    if rest is not None:
        PolyMeshArtist(ax, rest, T, facecolor="#efefef", edgecolor="#d9d9d9", lw=lw, zorder=1)
    mesh = PolyMeshArtist(ax, states[0], T, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
    if pin_pts is not None and len(pin_pts):
        P = np.asarray(pin_pts, dtype=float)
        ax.scatter(P[:, 0], P[:, 1], s=pin_size, marker="s", color=PIN_C, zorder=5,
                   label="clamped")
        ax.legend(loc="upper right", fontsize=9)
    txt = text_box(ax, labels[0], loc="upper left")
    fig.tight_layout()

    def update(i):
        mesh.update(states[i])
        txt.set_text(labels[i])
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim
