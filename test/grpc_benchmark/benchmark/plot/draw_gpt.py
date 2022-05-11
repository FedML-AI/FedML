import matplotlib.pyplot as plt
import numpy as np

colors = [
    [0.3, 0.3, 0.3],
    [0.6, 0.6, 0.6],
    [239 / 256.0, 74 / 256.0, 40 / 256.0],
]

WIDTH = 0.36
SHOW = False
FONT = {"fontname": "Times New Roman", "size": 22}

# chunks [1, 2, 4, 8, 16, 32, 64, 128]
#

data_fwd_mean = [
    [
        0,
        0,
        0,
        0,
        122653.87 / 1e3,
    ],  # CPU RPC
    [
        537652.98 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CUDA RPC
    [
        0,
        53.7,
        36.59,
        39.35,
        49.89,
    ],  # Pipeline
]

data_fwd_stdv = [
    [
        0,
        0,
        0,
        0,
        102.89 / 1e3,
    ],  # CPU RPC
    [
        771.32 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CUDA RPC
    [
        0,
        1.53,
        0.88,
        3.99,
        1.27,
    ],  # Pipeline
]

data_comm_mean = [
    [
        0,
        0,
        0,
        0,
        78920.51 / 1e3,
    ],  # CPU RPC
    [
        19293.20 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CUDA RPC
    [
        0,
        0,
        0,
        0,
        0,
    ],  # Pipeline
]

data_comm_stdv = [
    [
        0,
        0,
        0,
        0,
        89.53 / 1e3,
    ],  # CPU RPC
    [
        70.64 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CUDA RPC
    [
        0,
        0,
        0,
        0,
        0,
    ],  # Pipeline
]

data_bwd_mean = [
    [
        0,
        0,
        0,
        0,
        330225.75 / 1e3,
    ],  # CPU RPC
    [
        797521.04 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CUDA RPC
    [
        0,
        152.22,
        115.86,
        106.92,
        116.95,
    ],  # Pipeline
]

data_bwd_stdv = [
    [
        209.06 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CPU RPC
    [
        317.87 / 1e3,
        0,
        0,
        0,
        0,
    ],  # CUDA RPC
    [
        0,
        3.24,
        3.39,
        1.91,
        1.5,
    ],  # Pipeline
]


def plot_nlp(x_name, y_lim):
    plt.figure(figsize=(5.3, 4))
    xs = np.asarray(range(5))

    for i in range(1, 3):
        fwd = np.flip(np.asarray(data_fwd_mean[i]))
        com = np.flip(np.asarray(data_comm_mean[i]))
        bwd = np.flip(np.asarray(data_bwd_mean[i]))
        fwd_stdv = np.flip(np.asarray(data_fwd_stdv[i]))
        com_stdv = np.flip(np.asarray(data_comm_stdv[i]))
        bwd_stdv = np.flip(np.asarray(data_bwd_stdv[i]))

        fwd += com

        configs = {
            "width": WIDTH,
            "color": colors[i],
            "edgecolor": "black",
            "capsize": 6,
        }

        plt.bar(xs + (i - 1.5) * WIDTH, fwd, yerr=fwd_stdv, hatch="///", **configs)
        plt.bar(
            xs + (i - 1.5) * WIDTH,
            bwd,
            yerr=bwd_stdv,
            hatch="\\\\\\",
            bottom=fwd,
            **configs,
        )

    color_handles = []
    color_handles.append(plt.bar([20], [0], color=colors[1]))
    color_handles.append(plt.bar([2], [0], color=colors[2]))
    color_names = ["RPC", "Pipeline"]

    hatch_handles = []
    hatch_handles.append(plt.bar([2], [0], hatch="///", color="white"))
    hatch_handles.append(plt.bar([2], [0], hatch="\\\\\\", color="white"))
    hatch_names = ["FWD", "BWD"]

    def interleave(l1, l2):
        return [val for pair in zip(l1, l2) for val in pair]

    plt.legend(
        handles=interleave(color_handles, hatch_handles),
        loc="upper left",
        labels=interleave(color_names, hatch_names),
        prop={"family": FONT["fontname"], "size": FONT["size"] - 2},
        ncol=2,
        # bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )

    plt.xticks(xs, ["1", "2", "4", "8", "128"], **FONT)
    plt.yticks(**FONT)

    plt.xlabel(x_name, **FONT)
    plt.ylabel("Delay (Second)", **FONT)

    plt.ylim(y_lim)
    plt.xlim([-0.5, 4.5])

    # plt.yscale('log')

    # plt.show()
    plt.savefig(f"../images/gpt.pdf", bbox_inches="tight")


plot_nlp(x_name="Micro-Batch Size", y_lim=[0, 2000])
