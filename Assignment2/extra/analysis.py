import matplotlib.pyplot as plt

f = open("log", 'r')
lines = f.readlines()
f.close()

lws="LocalWorkSize "
d = {}
for i in range(len(lines)):
    if lines[i].startswith(lws):
        size = int(lines[i][len(lws):-2])
        metrics = []
        for j in range(1,6):
            metrics.append(float(lines[i+2*j].split(" ")[-2]))
        d[size] = metrics

print(d)

labels = ["interleavedAddressing", "sequentialAddressing", "kernelDecomposition", 
"kernelDecompositionUnroll", "kernelDecompositionAtomics"]

for size in list(d):
    plt.plot(d[size], label=str(size))

plt.xticks([e for e in range(5)], labels = labels, rotation=10,ha="center",
             rotation_mode="anchor")
plt.ylabel="Gelem/s"
plt.title="Different work-group sizes running the reduction approaches"
plt.legend()
plt.show()