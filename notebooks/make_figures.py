"""Draw the README's Qini figure. Run with: python -m notebooks.make_figures

This is a script rather than a notebook because Jupyter is not installed in this project.
"""

import numpy as np

from src import config, evaluate, learners, load  # config first: it sets the Matplotlib cache folder.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colours, published as colour-blind safe. Line styles carry identity as well.
STYLES = {
    "t_learner": {"label": "T-learner", "color": "#0072B2", "linestyle": "-"},
    "x_learner": {"label": "X-learner", "color": "#D55E00", "linestyle": "--"},
    "response": {"label": "Response model", "color": "#009E73", "linestyle": "-."},
}
RANDOM_COLOR = "#767676"
INK = "#333333"
OUTPUT = config.ROOT / "docs" / "figures" / "qini_hillstrom.png"


def draw():
    df = load.load_hillstrom()
    features = learners.hillstrom_features(df)
    train, holdout = learners.split_hillstrom(df)
    emails = {config.HILLSTROM_MENS: "Men's email vs no email", config.HILLSTROM_WOMENS: "Women's email vs no email"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for axis, (email, title) in zip(axes, emails.items()):
        train_rows, t_train = learners._email_rows(train, email)
        holdout_rows, t_holdout = learners._email_rows(holdout, email)
        y_holdout = holdout_rows["conversion"].to_numpy()
        scores = learners.fit_and_score(
            features.loc[train_rows.index], train_rows["conversion"].to_numpy(), t_train, features.loc[holdout_rows.index]
        )
        for name, style in STYLES.items():
            customers, incremental = evaluate.qini_curve(y_holdout, t_holdout, scores[name])
            axis.plot(100 * customers / customers[-1], incremental, linewidth=2, **style)
        axis.plot(
            [0, 100], [0, incremental[-1]], color=RANDOM_COLOR, linewidth=1.5, linestyle=":", label="Random targeting"
        )
        axis.set_title(title, loc="left", fontsize=11, color=INK)
        axis.set_xlabel("Share of hold-out customers targeted (%)", color=INK)
        axis.set_xlim(0, 100)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#BBBBBB")

    axes[0].set_ylabel("Cumulative added conversions (Qini)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, labelcolor=INK)
    fig.suptitle("On conversion, neither uplift model clearly beats the response model", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    print("Saved", OUTPUT)


if __name__ == "__main__":
    draw()
