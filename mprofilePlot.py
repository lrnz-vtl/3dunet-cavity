import pstats
p = pstats.Stats('prof.out')
# mode = 'cumulative'
mode = 'tottime'
p.sort_stats(mode).print_stats(30)