from matplotlib import  pyplot as plt

with open('mprofile_20210812005348.dat') as f:
    mem = [float(line.split()[1]) for line in f.readlines()[1:]]
    plt.plot(mem)