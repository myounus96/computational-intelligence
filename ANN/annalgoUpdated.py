import pandas as pd
import numpy as np
import random, math

df = pd.read_csv('NN-DATA.csv')

#convert data to list of lists
data = df.values.tolist()

#normalization
data = list(map(lambda i: list(map(lambda j: j / 20000, i)), data))

# #break data in training & testing
# train_data=data[:500]
# test_data=data[500:]

# outer_threshold = 0.001;
inner_threshold = 0.005;
rate_var = 0;

final_ansr=[]

# floating random weights
w14, w15, w24, w25, w34, w35, w46, w56 = np.random.uniform(-1, 1, 8)
# floating random biases
o4, o5, o6 = np.random.uniform(0, 1, 3)

while (1):
    # avg_total = []
    rec_index = 0
    # learning rate
    n = 1 / (len(data) + rate_var)
    print("gen:{}".format(rate_var))
    rate_var += 1

    #randomly get records,use this only if not first iteration and you not want to check threshold on avg
    # data=random.sample(data,len(data))

    for i in data:
        rec_index += 1
        # print("record:{}".format(i))
        x1, x2, x3, y = i[0], i[1], i[2], i[3]

        while (1):
            # print("record index:{}".format(rec_index))
            # print("rate_var:{}".format(rate_var))
            out4 = (x1 * w14) + (x2 * w24) + (x3 * w34) + o4
            out4 = 1 / (1 + math.e ** (-out4))

            out5 = (x1 * w15) + (x2 * w25) + (x3 * w35) + o5
            out5 = 1 / (1 + math.e ** (-out5))

            out6 = (out4 * w46) + (out5 * w56) + o6
            out6 = 1 / (1 + math.e ** (-out6))

            err6 = out6 * (1 - out6) * (y - out6)

            if abs(err6) <= inner_threshold:
                # avg_total.append(err6)
                break

            # weight & bias updation/back propagation....d means delta
            d_w46 = n * err6 * out4
            w46 += d_w46

            d_w56 = n * err6 * out5
            w56 += d_w56

            o6 += n * err6

            err4 = out4 * (1 - out4) * (err6) * (w46)
            err5 = out5 * (1 - out5) * (err6) * (w56)

            d_w14 = n * err4 * x1
            w14 += d_w14

            d_w15 = n * err5 * x1
            w15 += d_w15

            d_w24 = n * err4 * x2
            w24 += d_w24

            d_w25 = n * err5 * x2
            w25 += d_w25

            d_w34 = n * err4 * x3
            w34 += d_w34

            d_w35 = n * err5 * x3
            w35 += d_w35

            o4 += n * err4
            o5 += n * err5

    # in this list "calculated error" of each record should be appended
    # if (sum(avg_total) / len(data)) <= 0.001:
    #     break

    #validation code

    test=0

    for i in data:
        x1, x2, x3, y = i[0], i[1], i[2], i[3]

        out4 = (x1 * w14) + (x2 * w24) + (x3 * w34) + o4
        out4 = 1 / (1 + math.e ** (-out4))

        out5 = (x1 * w15) + (x2 * w25) + (x3 * w35) + o5
        out5 = 1 / (1 + math.e ** (-out5))

        out6 = (out4 * w46) + (out5 * w56) + o6
        out6 = 1 / (1 + math.e ** (-out6))

        err6 = out6 * (1 - out6) * (y - out6)

        if abs(err6) <= inner_threshold:
            # final_ansr.append(out6 * 20000)
            test=1
        else:
            test = 0
            break

    if test:
        break

    if rate_var > 300:
        break

print("\nFinal Weights and biases\n")
print(f'w14:{w14}, w15:{w15}, w24:{w24}, w25:{w25}, w34:{w34}, w35:{w35}, w46:{w46}, w56:{w56},o4:{o4}, o5:{o5}, o6:{o6}')

#writing data on the basis of fittest
for i in data:

    x1, x2, x3, y = i[0], i[1], i[2], i[3]

    out4 = (x1 * w14) + (x2 * w24) + (x3 * w34) + o4
    out4 = 1 / (1 + math.e ** (-out4))

    out5 = (x1 * w15) + (x2 * w25) + (x3 * w35) + o5
    out5 = 1 / (1 + math.e ** (-out5))

    out6 = (out4 * w46) + (out5 * w56) + o6
    out6 = 1 / (1 + math.e ** (-out6))

    err6 = out6 * (1 - out6) * (y - out6)

    final_ansr.append(out6 * 20000)


# final_ansr=final_ansr[-len(data):]

df['Y-ANN']=pd.DataFrame(final_ansr)
df.to_csv('NN-DATA2.csv',index=False)