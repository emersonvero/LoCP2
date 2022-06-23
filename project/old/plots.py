import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

fig,ax = plt.subplots(figsize=[15,7])

N = 5

for i in range(N):

    ax.plot([0,1],np.random.rand(2),linestyle='--',label=("$\lambda_{%s}(t)$" % i))
    


ax.set_ylabel("$ \lambda $")
ax.set_xlabel("t")
ax.set_xticks([0,1])
ax.set_xticklabels(["0","T"])
ax.set_yticks([])
ax.legend()

plt.savefig("./img/linear_lambda.png")
plt.show()
