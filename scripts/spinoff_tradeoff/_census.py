import glob, json, collections, numpy as np
C = collections.defaultdict(collections.Counter); N = collections.Counter(); sz = collections.Counter()
s = json.load(open('data/splits/split_v2.json')); ids = set(s['train'] + s['val'] + s['test'])
for p in glob.glob('../../data/npz_v2b_body/*/*.npz'):
    pid = p.replace(chr(92), '/').split('/')[-1][:-4]
    if pid not in ids: continue
    site = p.replace(chr(92), '/').split('/')[-2]; N[site] += 1
    m = json.load(open(p[:-4] + '.json'))
    if m['spacing_z_mm'] != 3.0: sz[(site, m['spacing_z_mm'])] += 1
    with np.load(p) as z:
        for k in z.files:
            if k.startswith('mask_') and z[k].any(): C[site][k[5:]] += 1
for st in C: print(st, N[st], dict(C[st]))
print(sz)
