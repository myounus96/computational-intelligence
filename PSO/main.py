import random, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fit_func = lambda x, y: 100 * (x ** 2 - y) ** 2 + (1 - x) ** 2

# random.seed(786)

G = 10;
n = 25;
c1 = c2 = 2

nList = [];

# init population
for i in range(n):
    x = random.uniform(-2, 2)
    y = random.uniform(-1, 3)
    fit_val = fit_func(x, y)
    nList.append([x, y, fit_val])


l_best = [];
g_best = 0;
p_best = 0;
p_best_list = []
velocities = []

for i in range(n):
    velocities.append({'x': 0, 'y': 0})

for ind, j in enumerate(nList):
    p_best_list.append({})
    p_best_list[ind][j[2]] = j

l_best.append((sorted(nList, key=lambda i: i[2], reverse=True))[0])

g_best = (sorted(l_best, key=lambda i: i[2], reverse=True))[0]

fig=plt.figure()

ax1 = fig.add_subplot(111)

def animate(i):
    global g_best

    for i in range(G):

        xs = [x for x in range(len(nList))]
        ys = [x[2] for x in nList]

        print(g_best)

        for ind, j in enumerate(nList):
            # get x component of best of particle's whole iterations
            p_best_x = p_best_list[ind][max(p_best_list[ind].keys())][0]

            velocities[ind]['x'] = velocities[ind]['x'] + c1 * random.uniform(0, 1) * (
                p_best_x - j[0]) + c2 * random.uniform(0, 1) * (g_best[0] - j[0])


            if j[0] + velocities[ind]['x'] >= -2 and j[0] + velocities[ind]['x'] <= 2:
                j[0] += velocities[ind]['x']

            p_best_y = p_best_list[ind][max(p_best_list[ind].keys())][1]

            velocities[ind]['y'] = velocities[ind]['y'] + c1 * random.uniform(0, 1) * (
                p_best_y - j[1]) + c2 * random.uniform(0, 1) * (g_best[1] - j[1])

            if j[1] + velocities[ind]['y'] >= -1 and j[1] + velocities[ind]['y'] <= 3:
                j[1] += velocities[ind]['y']

            j[2] = fit_func(j[0], j[1])

            p_best_list[ind][j[2]] = j


        l_best.append((sorted(nList, key=lambda i: i[2], reverse=True))[0])

        g_best = (sorted(l_best, key=lambda i: i[2], reverse=True))[0]


    ax1.clear()
    ax1.scatter(xs, ys)

ani=animation.FuncAnimation(fig,animate,interval=100)
plt.show()