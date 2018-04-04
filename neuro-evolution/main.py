import pandas as pd
import numpy as np
import random, math

df = pd.read_csv('NN-DATA.csv')

# convert data to list of lists
data = df.values.tolist()

# normalization
data = list(map(lambda i: list(map(lambda j: j / 20000, i)), data))


inner_threshold = 0.005;
rate_var = 0;


def fit_func(rec, i):
    x1, x2, x3, y = i[0], i[1], i[2], i[3]

    out4 = (x1 * rec['data']['w14']) + (x2 * rec['data']['w24']) + (x3 * rec['data']['w34']) + rec['data']['o4']
    out4 = 1 / (1 + math.e ** (-out4))

    out5 = (x1 * rec['data']['w15']) + (x2 * rec['data']['w25']) + (x3 * rec['data']['w35']) + rec['data']['o5']
    out5 = 1 / (1 + math.e ** (-out5))

    out6 = (out4 * rec['data']['w46']) + (out5 * rec['data']['w56']) + rec['data']['o6']
    out6 = 1 / (1 + math.e ** (-out6))

    err6 = out6 * (1 - out6) * (y - out6)

    if err6 <= inner_threshold:
        rec['fitness'] += 1


nList = [];
m = 20;
n = 25;
mutation_prob = 75;
mutation = 0.5;
for i in range(n):
    # floating random weights
    w14, w15, w24, w25, w34, w35, w46, w56 = np.random.uniform(-1, 1, 8)
    # floating random biases
    o4, o5, o6 = np.random.uniform(0, 1, 3)
    nList.append({'data': {'w14': w14, 'w15': w15, 'w24': w24, 'w25': w25, 'w34': w34, 'w35': w35, 'w46': w46,
                           'w56': w56, 'o4': o4, 'o5': o5, 'o6': o6}, 'fitness': 0})

#fitness calculation for first time/initial population
for rec in nList:
    for i in data:
        fit_func(rec, i)

count = 0
flag=False
while (1):
    count += 1
    mList = []

    rec_num=0
    for rec in nList:
        print("rec_index: {}".format(rec_num))
        rec_num+=1
        if rec['fitness'] == len(data):
            print('gen: {}'.format(count))
            print('Answer: {}'.format(rec))
            flag=True
            break

    if flag:
        break

    for j in range(int(m / 2)):
        # randomly parent selection(not same)
        # t=temp

        t = random.sample(range(n), 2)

        p1 = nList[t[0]]
        p2 = nList[t[1]]

        # crossover
        ch1 = list(p1['data'].values())[:5] + list(p2['data'].values())[5:]
        ch2 = list(p2['data'].values())[:5] + list(p1['data'].values())[5:]

        # mutation work
        for k in [ch1, ch2]:
            if random.randint(0, 100) <= mutation_prob:
                # get random for mutation in weights
                t = random.sample(range(8), 3)
                # get random for mutation in bias
                t1 = random.randint(8, 10)
                # odd,do +ve
                if random.randint(1, 2) == 1:
                    # mutation in weights
                    if (k[t[0]] + mutation) >= -1 and (k[t[0]] + mutation) <= 1:
                        k[t[0]] += mutation

                    if (k[t[1]] + mutation) >= -1 and (k[t[1]] + mutation) <= 1:
                        k[t[1]] += mutation

                    if (k[t[2]] + mutation) >= -1 and (k[t[2]] + mutation) <= 1:
                        k[t[2]] += mutation

                    # mutation in bias
                    if (k[t1] + mutation) >= 0 and (k[t1] + mutation) <= 1:
                        k[t1] += mutation

                # even,do -ve
                else:
                    # mutation in weights
                    if (k[t[0]] - mutation) >= -1 and (k[t[0]] - mutation) <= 1:
                        k[t[0]] -= mutation

                    if (k[t[1]] - mutation) >= -1 and (k[t[1]] - mutation) <= 1:
                        k[t[1]] -= mutation

                    if (k[t[2]] - mutation) >= -1 and (k[t[2]] - mutation) <= 1:
                        k[t[2]] -= mutation

                    # mutation in bias
                    if (k[t1] - mutation) >= 0 and (k[t1] - mutation) <= 1:
                        k[t1] -= mutation

        ch1 = {'data': {'w14': ch1[0], 'w15': ch1[1], 'w24': ch1[2], 'w25': ch1[3], 'w34': ch1[4], 'w35': ch1[5],
                        'w46': ch1[6],
                        'w56': ch1[7], 'o4': ch1[8], 'o5': ch1[9], 'o6': ch1[10]}, 'fitness': 0}

        ch2 = {'data': {'w14': ch2[0], 'w15': ch2[1], 'w24': ch2[2], 'w25': ch2[3], 'w34': ch2[4], 'w35': ch2[5],
                        'w46': ch2[6],
                        'w56': ch2[7], 'o4': ch2[8], 'o5': ch2[9], 'o6': ch2[10]}, 'fitness': 0}

        #calculate fitness of children
        for rec in [ch1,ch2]:
            for i in data:
                fit_func(rec, i)

        mList.append(ch1)
        mList.append(ch2)


    # combine both mList and nList and sort it with respect fit func value,also ovverides the nList,so we used in next iteration

    nList = sorted(nList + mList, key=lambda item: item['fitness'], reverse=True)[:25]
