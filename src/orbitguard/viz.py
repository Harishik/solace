"""Static report figures (plotly, self-contained HTML)."""

from __future__ import annotations

import plotly.graph_objects as go

BODY_COLORS = ["#ff5c7a", "#5cc8ff", "#9aa5b8"]  # projectile, bumper, rear wall


def impact_animation(frames: list[dict], scenario, path: str) -> None:
    """Animated scatter of an SPH Whipple-shield run, coloured by body."""

    def traces(f):
        out = []
        for b, name in enumerate(["projectile", "bumper", "rear wall"]):
            m = f["body"] == b
            out.append(go.Scattergl(x=f["pos"][m, 0], y=f["pos"][m, 1], mode="markers", name=name,
                                    marker=dict(size=2, color=BODY_COLORS[b])))
        return out

    w = scenario.width
    y0 = -scenario.bumper_thickness - scenario.standoff - scenario.rear_wall_thickness - 2
    fig = go.Figure(
        data=traces(frames[0]),
        frames=[go.Frame(data=traces(f), name=f"{f['t']:.2f}") for f in frames],
        layout=go.Layout(
            title=f"Whipple shield, {scenario.projectile_diameter} mm Al at {scenario.speed} km/s (2-D SPH)",
            xaxis=dict(range=[-w / 2, w / 2], title="mm", scaleanchor="y"),
            yaxis=dict(range=[y0, scenario.projectile_diameter * 2 + 1], title="mm"),
            template="plotly_dark",
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate",
                                                            args=[None, {"frame": {"duration": 120}}])])],
            sliders=[dict(steps=[dict(label=fr.name + " µs", method="animate",
                                      args=[[fr.name], {"mode": "immediate"}]) for fr in
                                 [go.Frame(name=f"{f['t']:.2f}") for f in frames]])],
        ),
    )
    fig.write_html(path, include_plotlyjs="cdn")
