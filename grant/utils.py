import pickle
import numpy as np

from collections.abc import Iterable

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import time
from functools import wraps


def save_pickle(thing, fn):
    with open("{}.pickle".format(fn), "wb") as handle:
        pickle.dump(thing, handle, protocol=3)


def load_pickle(fn):
    with open("{}.pickle".format(fn), "rb") as handle:
        thing = pickle.load(handle)
    return thing


def set_right_cbar(fig, axes):
    """Take a fig, axes pair (from grant.grant.easy_subplots() may I suggest)
    and make the RIGHTMOST column of figures into a cbar axis.

    For best results:
    * Set gridspec_kw with width_ratios where the last axis is very thin (1:10, perhaps)

    Args:
        fig (matplotlib figure)
        axes (ndarray): numpy array of axes
        ncols and nrows are ints
    """

    gs = axes[0, 0].get_gridspec()

    # For the ENTIRE last column
    for ax in axes[0:, -1]:
        # Remove all of those axes
        ax.remove()

    # Take up the space you removed with a single large axis
    cbar_ax = fig.add_subplot(gs[0:, -1])

    return fig, axes, cbar_ax


def calculate_inverse_simpson(X):
    ivs = X.apply(lambda r: 1 / np.sum(r**2), axis=1)
    return ivs


def draw_biplot_arrows(pca, pca_embedding, features, ax=None, th=False):

    if ax is None:
        fig, ax = plt.subplots()

    xvector = pca.components_[0]
    yvector = pca.components_[1]

    xs = pca_embedding[:, 0]
    ys = pca_embedding[:, 1]

    if th is True:
        arrow_lengths = np.sqrt(
            np.square(pca.components_[0, :]), np.square(pca.components_[1, :])
        )

        idx_to_draw = np.arange(len(arrow_lengths))[
            arrow_lengths > np.percentile(arrow_lengths, 99)
        ]

    elif th is False:
        idx_to_draw = np.arange(len(xvector))

    elif (isinstance(th, int) or isinstance(th, float)) and (th < 100):
        arrow_lengths = np.sqrt(
            np.square(pca.components_[0, :]), np.square(pca.components_[1, :])
        )

        idx_to_draw = np.arange(len(arrow_lengths))[
            arrow_lengths > np.percentile(arrow_lengths, th)
        ]

    else:
        idx_to_draw = np.arange(len(xvector))

    for i in idx_to_draw:

        plt.arrow(
            0,
            0,
            xvector[i] * max(xs),
            yvector[i] * max(ys),
            color="r",
            width=0.0005,
            head_width=0.005,
        )
        plt.text(
            xvector[i] * max(xs) * 1.2,
            yvector[i] * max(ys) * 1.2,
            list(features)[i],
            color="r",
        )

    result = {}
    for i in idx_to_draw:
        result[features[i]] = {
            "vector": np.array((xvector[i], yvector[i])),
            "arrow_length": np.sqrt(xvector[i] ** 2 + yvector[i] ** 2),
        }

    return result


def easy_multi_heatmap(
    ncols=1, nrows=1, base_figsize=None, width_ratios=None, **kwargs
):
    """returns (fig, axes, cbar_ax)"""

    if width_ratios is None:
        width_ratios = [1] * (ncols + 1)
        width_ratios[-1] = 0.1

        update_dict = {"gridspec_kw": {"width_ratios": width_ratios}}

        kwargs.update(update_dict)

    if base_figsize is None:
        base_figsize = (6, 6)

    ncols = ncols + 1  # to account for new cbar_ax

    fig, axes = easy_subplots(ncols, nrows, base_figsize=base_figsize, **kwargs)

    axes = axes.reshape((nrows, ncols))

    fig, axes, cbar_ax = set_right_cbar(fig, axes)

    if 1 in axes.shape:
        axes = axes.reshape(-1)

    return fig, axes, cbar_ax


def timefn(fn):
    """wrapper to time the enclosed function"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: {} took {} seconds".format(fn.__name__, t2 - t1))
        return result

    return measure_time


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def easy_subplots(ncols=1, nrows=1, base_figsize=None, **kwargs):

    if base_figsize is None:
        base_figsize = (8, 5)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(base_figsize[0] * ncols, base_figsize[1] * nrows),
        **kwargs,
    )

    # Lazy way of doing this
    try:
        axes = axes.reshape(-1)
    except:
        pass

    return fig, axes


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def savefig(fig, filename, ft=None):
    if ft is None:
        ft = "jpg"
    fig.savefig("{}.{}".format(filename, ft), dpi=300, bbox_inches="tight")


def custom_legend(
    n_entries,
    names,
    color_array=None,
    ax=None,
    line_weight=4,
    marker="o",
    linestyle="None",
    **kwargs,
):
    """Creates a custom legend on current graphic"""

    try:
        names = list(names)
    except Exception as e:
        raise Exception(
            "\n\n".join(
                [
                    str(e),
                    "'names' must be a list, or be able to be cast as a list",
                ]
            )
        )

    legend = []

    if color_array is not None:
        try:
            color_array = list(color_array)
        except Exception as e:
            raise Exception(
                "\n\n".join(
                    [
                        str(e),
                        "'color_array' must be a list, or be able to be cast as a list",
                    ]
                )
            )

        for i in range(n_entries):
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=color_array[i],
                    lw=line_weight,
                    marker=marker,
                    linestyle=linestyle,
                )
            )

    elif color_array is None:
        for i in range(n_entries):
            legend.append(
                Line2D([0], [0], lw=line_weight), marker=marker, linestyle=linestyle
            )

    if ax is None:
        plt.legend(legend, names, **kwargs)
    else:
        ax.legend(legend, names, **kwargs)
