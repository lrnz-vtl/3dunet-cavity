from matplotlib import pyplot as plt
import pstats
from pstats import SortKey

fname = '/home/lorenzo/3dunet-cavity/logs/test_refactor.prof'

# p = pstats.Stats('restats')
p = pstats.Stats(fname)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)


# with open('mprofile_20210812005348.dat') as f:
#     mem = [float(line.split()[1]) for line in f.readlines()[1:]]
#     plt.plot(mem)