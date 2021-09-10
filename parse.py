import re

fname = '/home/lorenzo/3dunet-cavity/logs/out.txt'
fname = '/home/lorenzo/3dunet-cavity/logs/output.text'

epochs = set()
lines = []

with open(fname) as f:
    for line in f.readlines():
        m = re.search('Epoch \[(\d+)/999\]', line)
        if m is not None:
            epoch = int(m.group(1))
            if epoch not in epochs:
                epochs.add(epoch)
                lines.append(line)

        m = re.search('Slice builder config', line)
        if m is not None:
            lines.append(line)

print('\n'.join(lines))