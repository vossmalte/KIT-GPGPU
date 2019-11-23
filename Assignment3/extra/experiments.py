import matplotlib.pyplot as plt
import numpy as np

results = {
    "2x": {
        "2": 1.60914,
        "4": 1.66838,
        "8":1.61511,
        "16":1.51495
    },
    "4x": {
        "2": 1.65692,
        "4":1.62547,
        "8":1.55718,
        "16":1.5327,
    },
    "8x": {
        "2": 1.63148,
        "4":1.64538,
        "8":1.61035,
        "16":1.51034
    },
    "16x": {
        "2": 1.61688,
        "4":1.66603,
        "8":1.61982,
        "16":1.54467
    }
}


all_values = [list(vert.values()) for vert in list(results.values())]
plt.xticks([0,1,2,3], labels=list(results["2x"]))
plt.xlabel="V_RESULT_STEPS"
plt.ylabel="throughput [Gpixels/s]"
for i in range(len(all_values)):
    plt.plot(all_values[i], label=list(results)[i])
plt.legend()
plt.tight_layout()

plt.show()
plt.savefig("figure.png")