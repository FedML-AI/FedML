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


data_fwd_mean = [
    [
        3.1925,
        5.3389,
    ],  # DDP
    [
        44.222187995910645 - 0.7689118385314941,
        87.3267412185669 - 0.4015803337097168,
    ],  # CPU RPC
    [
        5.0170183181762695 - 2.988492810726166,
        34.88408327102661 - 12.825427198410035,
    ],  # CUDA RPC
]

data_fwd_stdv = [
    [
        0.6692,
        2.4257,
    ],  # DDP
    [
        32.92566779663072 - 0.05272164627466966,
        110.28518650791624 - 0.02627479562734089,
    ],
    [
        0.3196359096421508,
        1.6836104408202686 - 1.5476508630812806,
    ],
]

data_comm_mean = [
    [
        42.1918,
        173.6781,
    ],
    [
        30.799853801727295,
        67.9653525352478,
    ],
    [
        4.43674875497818,
        32.97247829437255,
    ],
]

data_comm_stdv = [
    [
        0.3729,
        0.9426,
    ],
    [
        24.04035565363716,
        87.34213383147451,
    ],
    [0.3698690169656274, 3.0801068468562987],
]

data_bwd_mean = [
    [
        51.3728,
        182.0691,
    ],
    [
        366.519570350647 - 30.0309419632,
        831.6826105117798 - 67.9653525352478,
    ],
    [
        3.733503818511963 - 1.44825594425,
        36.26296520233154 - 20.147051096,
    ],
]

data_bwd_stdv = [
    [
        13.9692,
        18.1527,
    ],
    [
        82.46750627159679 - 23.9876340074,
        174.4926704697184 - 87.34213383147451,
    ],
    [
        0.3396226211714276,
        2.200198987343897 - 1.53245598378,
    ],
]


def plot_nlp(x_name, f_name, y_lim):
    plt.figure(figsize=(4, 4))
    xs = np.asarray(range(2))

    for i in [0, 2]:
        fwd = np.asarray(data_fwd_mean[i]) / 1e3
        com = np.asarray(data_comm_mean[i]) / 1e3
        bwd = np.asarray(data_bwd_mean[i]) / 1e3
        fwd_stdv = np.asarray(data_fwd_stdv[i]) / 1e3
        com_stdv = np.asarray(data_comm_stdv[i]) / 1e3
        bwd_stdv = np.asarray(data_bwd_stdv[i]) / 1e3

        bwd -= com
        com *= 2

        configs = {
            "width": WIDTH,
            "color": colors[i],
            "edgecolor": "black",
            "capsize": 6,
        }

        plt.bar(xs + (i / 2 - 0.5) * WIDTH, fwd, yerr=fwd_stdv, hatch="///", **configs)
        plt.bar(
            xs + (i / 2 - 0.5) * WIDTH,
            com,
            yerr=com_stdv,
            hatch="\\\\\\",
            bottom=fwd,
            **configs,
        )
        plt.bar(
            xs + (i / 2 - 0.5) * WIDTH,
            bwd,
            yerr=bwd_stdv,
            hatch="...",
            bottom=fwd + com,
            **configs,
        )

    color_handles = []
    color_handles.append(plt.bar([4], [0], color=colors[0]))
    # color_handles.append(plt.bar([4], [0], color=colors[1]))
    color_handles.append(plt.bar([4], [0], color=colors[2]))
    color_names = ["DDP", "CUDA"]

    hatch_handles = []
    hatch_handles.append(plt.bar([4], [0], hatch="///", color="white"))
    hatch_handles.append(plt.bar([4], [0], hatch="\\\\\\", color="white"))
    hatch_handles.append(plt.bar([4], [0], hatch="...", color="white"))
    hatch_names = ["FWD", "COMM", "BWD"]

    def interleave(l1, l2):
        return [val for pair in zip(l1, l2) for val in pair]

    plt.legend(
        handles=hatch_handles + color_handles,
        loc="upper left",
        labels=hatch_names + color_names,
        prop={"family": FONT["fontname"], "size": FONT["size"] - 2},
        ncol=1,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        # bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )

    plt.xticks(xs, ["4", "16"], **FONT)
    plt.yticks(**FONT)

    plt.xlabel(x_name, **FONT)
    plt.ylabel("Delay (Second)", **FONT)

    plt.ylim(y_lim)
    plt.xlim([-0.5, 1.5])
    # plt.yscale('log')

    if SHOW:
        plt.show()
    else:
        plt.savefig(f"../images/{f_name}.pdf", bbox_inches="tight")


plot_nlp(x_name="Number of GPUs", f_name="rec", y_lim=[0, 0.5])


WIDTH = 0.5
plt.figure(figsize=(4, 4))

iter_delay = [
    6.443297863006592,
    6.856632232666016,
    7.402908802032471,
    7.4396491050720215,
    8.179962635040283,
    8.215594291687012,
]

iter_stdev = [
    0.21451405133227391,
    0.2458064285360685,
    0.27179209255802483,
    0.24238109624973778,
    0.3054622322309643,
    0.35940199993917693,
]


configs = {
    "width": WIDTH,
    "color": colors[2],
    "edgecolor": "black",
    "capsize": 6,
}
xs = np.asarray(range(6))
plt.bar(xs, iter_delay, yerr=iter_stdev, hatch="///", **configs)


plt.xticks(xs, ["20", "40", "60", "80", "100", "120"], **FONT)
plt.yticks(**FONT)

plt.xlabel("Million Embeddings", **FONT)
plt.ylabel("Delay (ms)", **FONT)

plt.ylim([0, 10])
plt.xlim([-0.5, 5.5])
# plt.yscale('log')

if SHOW:
    plt.show()
else:
    plt.savefig(f"../images/rec_shard.pdf", bbox_inches="tight")
