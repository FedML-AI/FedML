# light vs heavy
# 0: GCPP - grpc cpu Python PCIe
# 1: GCSP - grpc cpu script PCIe
# 2: GCPE - grpc cpu Python EFA
# 3: GCSE - grpc cpu script EFA
# 4: GGPP - grpc GPU Python PCIe
# 5:
# 2: grpc cuda
# 3: grpc cuda script
# 4: ptrpc cpu
# 5: ptrpc cpu script
# 6: ptrpc cuda
# 7: ptrpc cuda script
# 8: ptrpc cuda NVLink
# 9: ptrpc cuda NVLink
# 10: ptrpc cuda IB
# 11: ptrpc cuda script IB

# [GP] grpc vs PT-RPC
# [CG] CPU vs GPU
# [PS] Python vs Script
# [PENI] PCIe vs EFA vs NVLink vs IB

# light vs heavy
# single machine [PCIe vs NVLink] vs multi machine [EFA vs IB]

# 0: GCP - grpc + CPU + Python
# 1: GCS - grpc + CPU + Script
# 2: GGP - grpc + GPU + Python
# 3: GGS - grpc + GPU + Script
# 4: PCP - PT-RPC + CPU + Python
# 5: PCS - PT-RPC + CPU + Script
# 6: PGP - PT-RPC + GPU + Python
# 7: PGS - PT-RPC + GPU + Script

# Figure:
#  1k *  1k + light + single
#  1k *  1k + light + multi
# 10k * 10k + light + single
# 10k * 10k + light + multi
#  1k *  1k + heavy + single
#  1k *  1k + heavy + multi
# 10k * 10k + heavy + single
# 10k * 10k + heavy + multi

"""
files = [
    "logs/single_aws_2/single_pt_rpc_%d.log",
    "logs/learnfair/single_pt_rpc_%d.log",
    "logs/multi_aws_2/single_pt_rpc_%d.log",
    "logs/single_aws_2/single_grpc_%d.log",
    "logs/multi_aws_2/single_grpc_%d.log",
]



"""

import matplotlib.pyplot as plt
import numpy as np


colors = [
    [0.3, 0.3, 0.3],
    [239 / 256.0, 74 / 256.0, 40 / 256.0],
    [0.6, 0.6, 0.6],
]

WIDTH = 0.3
SHOW = False
FONT = {"fontname": "Times New Roman", "size": 22}


def plot_bar(name, ylim):
    mean = np.asarray(data[name + "_mean"]) * 1e3
    stdv = np.asarray(data[name + "_stdv"]) * 1e3
    xs = np.arange(4)
    # plt.figure(figsize=(6, 3))
    handles = []
    handles.append(
        plt.bar(
            xs - WIDTH / 2.0,
            mean[0],
            yerr=stdv[0],
            color=colors[0],
            width=WIDTH,
            capsize=6,
        )
    )
    handles.append(
        plt.bar(
            xs + WIDTH / 2.0,
            mean[1],
            yerr=stdv[1],
            color=colors[1],
            width=WIDTH,
            capsize=6,
        )
    )

    plt.xticks(xs, ["CP", "CS", "GP", "GS"], **FONT)
    plt.yticks(**FONT)

    plt.ylabel("Delay (ms)", **FONT)

    plt.legend(
        handles=handles,
        loc="upper left",
        labels=["grpc", "PT"],
        prop={"family": FONT["fontname"], "size": FONT["size"]},
        ncol=2,
        # bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )

    plt.ylim(ylim)
    plt.grid()

    """
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"../images/{name}.pdf", bbox_inches='tight')
    """


def plot_bar3(name, ylim, ax):
    mean = np.asarray(data[name + "_mean"]) * 1e3
    stdv = np.asarray(data[name + "_stdv"]) * 1e3
    xs = np.arange(4)
    # plt.figure(figsize=(6, 3))
    handles = []
    handles.append(
        plt.bar(
            xs - WIDTH, mean[0], yerr=stdv[0], color=colors[0], width=WIDTH, capsize=6
        )
    )
    handles.append(
        plt.bar(xs, mean[1], yerr=stdv[1], color=colors[1], width=WIDTH, capsize=6)
    )
    handles.append(
        plt.bar(
            xs + WIDTH, mean[2], yerr=stdv[2], color=colors[2], width=WIDTH, capsize=6
        )
    )

    plt.xticks(xs, ["CP", "CS", "GP", "GS"], **FONT)
    # plt.yticks(**FONT)

    # plt.ylabel("Delay (ms)", **FONT)

    plt.legend(
        handles=[handles[2]],
        loc="upper left",
        labels=["PT IB"],
        prop={"family": FONT["fontname"], "size": FONT["size"]},
    )

    # plt.ylim(ylim)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.grid()

    """
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"../images/{name}.pdf", bbox_inches='tight')
    """


data = {}

data["small_light_single_mean"] = [
    [
        0.026708102226257323,  # GCP
        0.03247213363647461,  # GCS
        0.0599402910232544,  # GGP
        0.030090252685546874,  # GGS
    ],  # grpc
    [
        0.00871570110321045,  # PCP
        0.008874011039733887,  # PCS
        0.004539344000816345,
        0.0014466304063796998,
    ],  # pt rpc
]

data["small_light_single_stdv"] = [
    [
        0.004715544243583349,
        0.010037730927172632,
        0.02789967304355296,
        0.006728921952568284,
    ],  # grpc
    [
        0.003113925632485608,
        0.003417710351898507,
        0.001146082866262916,
        0.00019359466164992894,
    ],  # pt rpc
]

data["small_light_multi_mean"] = [
    [
        0.06157054901123047,
        0.05341341495513916,
        0.11209506454467774,
        0.03241867504119873,
    ],  # grpc
    [
        0.005541062355041504,
        0.005627202987670899,
        0.04050200967788696,
        0.03573512620925904,
    ],  # PT RPC EFA
    [
        0,  # 0.03316252231597901,
        0,  # 0.03235681056976318,
        0.016081782436370852,
        0.005295679926872253,
    ],  # PT RPC IB
]

data["small_light_multi_stdv"] = [
    [
        0.019342111628262035,
        0.016638521784208063,
        0.036828845998343275,
        0.008091798959968794,
    ],
    [
        0.0016513753253001736,
        0.0018334709060509695,
        0.0190571934776313,
        0.01899493488811709,
    ],
    [
        0,  # 0.011050395280665903,
        0,  # 0.013056317959832133,
        0.00752183610847404,
        0.0013382178476675506,
    ],
]


name = "small_light"
plt.figure(figsize=(10, 3))
ax1 = plt.subplot(121)
plot_bar(f"{name}_single", [0, 200])
plt.text(2.7, 120, "intra", **FONT)

ax2 = plt.subplot(122, sharey=ax1)
plot_bar3(f"{name}_multi", [0, 200], ax2)
plt.text(-0.5, 120, "cross", **FONT)

plt.subplots_adjust(wspace=0)

if SHOW:
    plt.show()
else:
    plt.savefig(f"../images/{name}.pdf", bbox_inches="tight")


########

data["small_heavy_single_mean"] = [
    [
        0.20356476306915283,  # GCP
        0.1753929615020752,  # GCS
        0.1090344779968262,  # GGP
        0.03455644454956055,  # GGS
    ],  # grpc
    [
        0.08173201084136963,
        0.08039040565490722,
        0.01840829429626465,
        0.019564937973022462,
    ],  # PT RPC
]

data["small_heavy_single_stdv"] = [
    [
        0.030740158246616446,
        0.026400505243055778,
        0.012998388008905444,
        0.0077261431947167475,
    ],
    [
        0.008086306850952114,
        0.008788756231678428,
        0.0030035480556141986,
        0.0015929879224909264,
    ],
]

data["small_heavy_multi_mean"] = [
    [
        0.15635180473327637,
        0.10107812881469727,
        0.11622803573608398,
        0.04124575309753418,
    ],
    [
        0.08081684112548829,
        0.07558789253234863,
        0.04198958997726441,
        0.0370933913230896,
    ],
    [
        0,  # 0.10263054370880127,
        0,  # 0.10391204357147217,
        0.023845241641998288,
        0.017658985614776614,
    ],  # PT IB
]

data["small_heavy_multi_stdv"] = [
    [
        0.011790788841651524,
        0.022212092601855485,
        0.01622634971258239,
        0.008715935601335283,
    ],
    [
        0.012027888882285216,
        0.007028608622937932,
        0.01915620890073913,
        0.01899507439187126,
    ],
    [
        0,  # 0.010964331419899845,
        0,  # 0.01258920269134317,
        0.004454980744851911,
        0.0034703982699525286,
    ],  # PT IB
]

name = "small_heavy"
plt.figure(figsize=(10, 3))
ax1 = plt.subplot(121)
plot_bar(f"{name}_single", [0, 350])
plt.text(2.7, 120 * 3.5 / 2, "intra", **FONT)

ax2 = plt.subplot(122, sharey=ax1)
plot_bar3(f"{name}_multi", [0, 350], ax2)
plt.text(-0.5, 120 * 3.5 / 2, "cross", **FONT)

plt.subplots_adjust(wspace=0)

if SHOW:
    plt.show()
else:
    plt.savefig(f"../images/{name}.pdf", bbox_inches="tight")


#######


def large_plot_bar(name, ylim):
    mean = np.asarray(data[name + "_mean"])
    stdv = np.asarray(data[name + "_stdv"])
    xs = np.arange(4)
    # plt.figure(figsize=(6, 3))
    handles = []
    handles.append(
        plt.bar(
            xs - WIDTH / 2.0,
            mean[0],
            yerr=stdv[0],
            color=colors[0],
            width=WIDTH,
            capsize=6,
        )
    )
    handles.append(
        plt.bar(
            xs + WIDTH / 2.0,
            mean[1],
            yerr=stdv[1],
            color=colors[1],
            width=WIDTH,
            capsize=6,
        )
    )

    plt.xticks(xs, ["CP", "CS", "GP", "GS"], **FONT)
    plt.yticks(**FONT)

    plt.ylabel("Delay (Second)", **FONT)

    plt.legend(
        handles=handles,
        loc="upper left",
        labels=["grpc", "PT"],
        prop={"family": FONT["fontname"], "size": FONT["size"]},
        ncol=2,
        # bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )

    plt.ylim(ylim)
    plt.grid()

    """
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"../images/{name}.pdf", bbox_inches='tight')
    """


def large_plot_bar3(name, ylim, ax):
    mean = np.asarray(data[name + "_mean"])
    stdv = np.asarray(data[name + "_stdv"])
    xs = np.arange(4)
    # plt.figure(figsize=(6, 3))
    handles = []
    handles.append(
        plt.bar(
            xs - WIDTH, mean[0], yerr=stdv[0], color=colors[0], width=WIDTH, capsize=6
        )
    )
    handles.append(
        plt.bar(xs, mean[1], yerr=stdv[1], color=colors[1], width=WIDTH, capsize=6)
    )
    handles.append(
        plt.bar(
            xs + WIDTH, mean[2], yerr=stdv[2], color=colors[2], width=WIDTH, capsize=6
        )
    )

    plt.xticks(xs, ["CP", "CS", "GP", "GS"], **FONT)
    # plt.yticks(**FONT)

    # plt.ylabel("Delay (ms)", **FONT)

    plt.legend(
        handles=[handles[2]],
        loc="upper left",
        labels=["PT IB"],
        prop={"family": FONT["fontname"], "size": FONT["size"]},
    )

    # plt.ylim(ylim)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.grid()

    """
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"../images/{name}.pdf", bbox_inches='tight')
    """


data["large_light_single_mean"] = [
    [
        22.20263538360596,  # GCP
        24.788425254821778,  # GCS
        23.351867480468748,  # GGP
        11.011896142578125,  # GGS
    ],
    [
        1.2488246440887452,
        1.3589040994644166,
        0.1270621353149414,
        0.10892848949432372,
    ],
]

data["large_light_single_stdv"] = [
    [5.771916091747043, 5.364484586934269, 6.918799080128208, 1.9855945166246234],
    [
        0.5846850829686981,
        0.6423697106855647,
        0.05806563249041085,
        0.054903939847492186,
    ],
]

data["large_light_multi_mean"] = [
    [25.670038175582885, 26.178440260887147, 26.272700683593747, 13.444128710937502],
    [
        0.4566845417022705,
        0.45766301155090333,
        4.171952026367188,
        3.683140618896484,
    ],
    [
        0,  # 2.3643112659454344,
        0,  # 2.3529667139053343,
        0.3194052200317382,
        0.24071654739379883,
    ],  # IB
]

data["large_light_multi_stdv"] = [
    [
        7.146803514600233,
        6.839167249104777,
        7.496780532630554,
        2.777911749383032,
    ],
    [
        0.2184543757717796,
        0.2227305837441535,
        1.941776007191919,
        1.9379969055436004,
    ],
    [
        0,  # 1.0914719969608395,
        0,  # 1.0934558992804269,
        0.14819209802419867,
        0.1078176178281113,
    ],
]

name = "large_light"
plt.figure(figsize=(10, 3))
ax1 = plt.subplot(121)
ax1.set_yscale("log")
large_plot_bar(f"{name}_single", [0.05, 10000])
plt.text(2.7, 80, "intra", **FONT)

ax2 = plt.subplot(122, sharey=ax1)
ax2.set_yscale("log")
large_plot_bar3(f"{name}_multi", [0.05, 10000], ax2)
plt.text(-0.5, 80, "cross", **FONT)

plt.subplots_adjust(wspace=0)

if SHOW:
    plt.show()
else:
    plt.savefig(f"../images/{name}.pdf", bbox_inches="tight")


data["large_heavy_single_mean"] = [
    [
        42.54107656478882,
        38.679797768592834,
        25.43493349609375,
        14.1742890625,
    ],
    [
        16.5695259809494,
        14.258692002296447,
        1.5742439208984376,
        1.4494814453125,
    ],
]


data["large_heavy_single_stdv"] = [
    [
        7.901092728867473,
        9.498046712534931,
        7.488048587705518,
        3.6846603869313106,
    ],
    [6.58542294507806, 5.0298745484719705, 0.31908617359309316, 0.3272961587238953],
]

data["large_heavy_multi_mean"] = [
    [
        38.442934083938596,
        35.58412787914276,
        27.784189160156252,
        14.961043750000002,
    ],
    [
        16.519581270217895,
        14.204672360420227,
        4.366681433105469,
        3.8775099121093755,
    ],
    [
        0,  # 27.967683577537535,
        0,  # 25.08467490673065,
        1.8139447509765625,
        1.7794348754882816,
    ],
]

data["large_heavy_multi_stdv"] = [
    [
        4.210697320599725,
        8.901779052553074,
        7.133498496783156,
        3.0455976968829157,
    ],
    [
        1.9035645472548066,
        2.264535141637365,
        1.9410518689572116,
        1.9375170968683162,
    ],
    [
        0,  # 4.343233661234864,
        0,  # 3.177754522705607,
        0.29057995712800344,
        0.25226875866321113,
    ],
]

name = "large_heavy"
plt.figure(figsize=(10, 3))
ax1 = plt.subplot(121)
ax1.set_yscale("log")
large_plot_bar(f"{name}_single", [0.05, 10000])
plt.text(2.7, 80, "intra", **FONT)

ax2 = plt.subplot(122, sharey=ax1)
ax2.set_yscale("log")
large_plot_bar3(f"{name}_multi", [0.05, 10000], ax2)
plt.text(-0.5, 80, "cross", **FONT)

plt.subplots_adjust(wspace=0)

if SHOW:
    plt.show()
else:
    plt.savefig(f"../images/{name}.pdf", bbox_inches="tight")
