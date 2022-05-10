import matplotlib.pyplot as plt
import numpy as np


colors = [
    [0.3, 0.3, 0.3],
    [0.6, 0.6, 0.6],
    [239 / 256.0, 74 / 256.0, 40 / 256.0],
]

WIDTH = 0.3
SHOW = False
FONT = {"fontname": "Times New Roman", "size": 22}

fetch_delay_mean = [
    [
        0.04694,
        0.03694,
        0.03620,
        0.03764,
        0.0391242159737481,
        0.052636506822374134,
        0.06071656545003255,
        0.08869274192386203,
    ],  # grpc
    [
        0.00774,
        0.00885,
        0.00872,
        0.00900,
        0.011370658874511719,
        0.011322021484375,
        0.013988137245178223,
        0.032533695962693954,
    ],  # CPU
    [
        0.00535,
        0.00549,
        0.00575,
        0.00575,
        0.005764425597671006,
        0.005885667933358087,
        0.006081581483077672,
        0.005898183475558956,
    ],  # CUDA
]

fetch_delay_stdv = [
    [
        0.00399,
        0.00547,
        0.00323,
        0.00586,
        0.01159347876102909,
        0.044711036383304974,
        0.030639126816620704,
        0.0792876920181466,
    ],
    [
        0.00111,
        0.00140,
        0.00130,
        0.00139,
        0.006590051563060455,
        0.006793594530467246,
        0.011212384597925687,
        0.05478976943389735,
    ],
    [
        0.000515,
        0.00054,
        0.00048,
        0.00047,
        0.0007373455796658365,
        0.0005577237905989097,
        0.0009866885039473779,
        0.0007528747487282274,
    ],
]

update_delay_mean = [
    [
        0.09087,
        0.08392,
        0.08573,
        0.09151,
        0.4195092447688071,
        0.6269224011496212,
        0.7451699396197715,
        1.0245614507225123,
    ],  # grpc
    [
        0.05269,
        0.05266,
        0.05502,
        0.05459,
        0.6127188550101386,
        0.7235896931754218,
        0.6864565716849433,
        1.638031636344062,
    ],  # CPU
    [
        0.01246,
        0.01290,
        0.01310,
        0.01384,
        0.01433212012052536,
        0.014361156264526977,
        0.015118700069271856,
        0.014623743978639444,
    ],  # GPU
]

update_delay_stdv = [
    [
        0.01115,
        0.01546,
        0.01617,
        0.01811,
        0.1738510783850508,
        0.26840761319235884,
        0.356174453618109,
        0.4584052645960193,
    ],
    [
        0.00218,
        0.00400,
        0.00778,
        0.00640,
        0.2214552835610761,
        0.26733996435657853,
        0.26776838670253855,
        0.5302236634784867,
    ],
    [
        0.00095,
        0.00078,
        0.00081,
        0.00130,
        0.0017324355350418952,
        0.00229041608501797,
        0.001746302112273325,
        0.0018825723757046988,
    ],
]


def plot_rl(f_name, y_lim):
    plt.figure(figsize=(6, 3))
    xs = np.asarray(range(4))

    for i in range(3):
        fetch = np.asarray(fetch_delay_mean[i][:4]) * 1e3
        update = np.asarray(update_delay_mean[i][:4]) * 1e3
        fetch_stdv = np.asarray(fetch_delay_stdv[i][:4]) * 1e3
        update_stdv = np.asarray(update_delay_stdv[i][:4]) * 1e3

        configs = {
            "width": WIDTH,
            "color": colors[i],
            "edgecolor": "black",
            "capsize": 6,
        }

        plt.bar(xs + (i - 1) * WIDTH, fetch, yerr=fetch_stdv, hatch="///", **configs)
        plt.bar(
            xs + (i - 1) * WIDTH,
            update,
            yerr=update_stdv,
            hatch="\\\\\\",
            bottom=fetch,
            **configs,
        )

    color_handles = []
    color_handles.append(plt.bar([20], [0], color=colors[0]))
    color_handles.append(plt.bar([20], [0], color=colors[1]))
    color_handles.append(plt.bar([20], [0], color=colors[2]))
    color_names = ["grpc", "CPU", "CUDA"]

    """
    hatch_handles = []
    hatch_handles.append(plt.bar([20], [0], hatch="///", color="white"))
    hatch_handles.append(plt.bar([20], [0], hatch="\\\\\\", color="white"))
    hatch_names = ["Fetch", "Update"]
    """

    plt.legend(
        handles=color_handles,
        loc="upper left",
        labels=color_names,
        prop={"family": FONT["fontname"], "size": FONT["size"] - 3},
        ncol=3,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        # bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )

    plt.xticks(xs, ["1", "3", "5", "7"], **FONT)
    plt.yticks(**FONT)

    plt.xlabel("Number of Actors", **FONT)
    plt.ylabel("Delay (ms)", **FONT)

    plt.ylim(y_lim)
    plt.xlim([-0.5, 3.5])

    # plt.show()
    plt.savefig(f"../images/{f_name}.pdf", bbox_inches="tight")


plot_rl("mario_single", [0, 220])


def plot_rl(f_name, y_lim):
    plt.figure(figsize=(6, 3))
    xs = np.asarray(range(4))

    for i in range(3):
        fetch = np.asarray(fetch_delay_mean[i][4:]) * 1e3
        update = np.asarray(update_delay_mean[i][4:]) * 1e3
        fetch_stdv = np.asarray(fetch_delay_stdv[i][4:]) * 1e3
        update_stdv = np.asarray(update_delay_stdv[i][4:]) * 1e3

        configs = {
            "width": WIDTH,
            "color": colors[i],
            "edgecolor": "black",
            "capsize": 6,
        }

        plt.bar(xs + (i - 1) * WIDTH, fetch, yerr=fetch_stdv, hatch="///", **configs)
        plt.bar(
            xs + (i - 1) * WIDTH,
            update,
            yerr=update_stdv,
            hatch="\\\\\\",
            bottom=fetch,
            **configs,
        )

    props = {"family": FONT["fontname"], "size": FONT["size"] - 3}

    for i in range(4, 8):
        delay = (fetch_delay_mean[2][i] + update_delay_mean[2][i]) * 1e3
        plt.text(0.22 + i - 4, 1000, f"{delay:.2f}", props, rotation=-90)
        plt.arrow(
            0.33 + i - 4, 700, 0, -300, head_width=0.05, head_length=30, fc="k", ec="k"
        )

    """
    color_handles = []
    color_handles.append(plt.bar([20], [0], color=colors[0]))
    color_handles.append(plt.bar([20], [0], color=colors[1]))
    color_handles.append(plt.bar([20], [0], color=colors[2]))
    color_names = ["grpc", "CPU", "CUDA"]
    """

    hatch_handles = []
    hatch_handles.append(plt.bar([20], [0], hatch="///", color="white"))
    hatch_handles.append(plt.bar([20], [0], hatch="\\\\\\", color="white"))
    hatch_names = ["Fetch", "Update"]

    plt.legend(
        handles=hatch_handles,
        loc="upper left",
        labels=hatch_names,
        prop=props,
        ncol=2,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        # bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )

    plt.xticks(xs, ["9", "11", "13", "15"], **FONT)
    plt.yticks(**FONT)

    plt.xlabel("Number of Actors", **FONT)
    plt.ylabel("Delay (ms)", **FONT)

    plt.ylim(y_lim)
    plt.xlim([-0.5, 3.5])

    # plt.show()
    plt.savefig(f"../images/{f_name}.pdf", bbox_inches="tight")


plot_rl("mario_multi", [0, 2200])
