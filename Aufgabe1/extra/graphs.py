import matplotlib.pyplot as plt

threads = [1<<11, 1<<20, 1<<20, 1<<25]
worksize=[16,64,256,1024]
lines = []
analysis = {t: [] for t in worksize}
with open("build/log","r") as f:
    lines = f.readlines()
for i in range(min(96,len(lines))):
    line = lines[i]
    if not line.startswith("Computing GPU result."):
        continue
    if line.startswith("Running matrix"):
        break
    line = line.split("Executing ")[1].split(" threads in ")
    thread = int(line[0])
    groups, size = [int(s) for s in line[1].split(" groups of size ")]
    time = float(lines[i+1][38:-4])
    analysis[size].append(time)

print(analysis)

x = [e for e in range(1,5)]
width=0.2
b = len(worksize)
fig, ax = plt.subplots()
pos = [-1.5*width, -width/2, width/2, 1.5*width]
for i in range(4):
    x_axis = [xi+pos[i] for xi in x]
    print(x_axis)
    ax.bar(x_axis, analysis[worksize[i]], width, label="Worksize " +str(worksize[i]))

plt.yscale("log")
plt.title("Malte Voss: Runtimes of different group sizes for different vector sizes")
ax.set_ylabel("Execution time [ms]")
ax.set_xlabel("Vector size")
ax.set_xticks(x)
ax.set_xticklabels(threads)
ax.legend()

plt.show()

