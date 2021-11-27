import random

f = open('datasets/multi_dataset_isp2.csv', 'r')
f1 = open('datasets/testmultiv2.txt', 'w')
a = []
t = ""
op = []
ha = 0
for line in f:
    if ha == 0:
        ha += 1
        continue
    op = line[line.index(',') + 1:].split(",")
    text = op[0]
    if len(text) > 3:
        for i in range(len(text)-2):
            t += text[i:i+3] + " "
    else:
        t = text + " "
    a.append(t[:-1] + "," + op[1])
    t = ""
random.shuffle(a)
for line in a:
    f1.write(line)
f.close()
f1.close()
