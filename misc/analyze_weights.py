import csv, math

def load(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r['Layer Name'], float(r['Frobenius Norm'])))
    return rows

clic   = load('plots/clic_weight_comparison.csv')
cogvlm = load('plots/cogvlm_weight_comparison.csv')

def stats(rows, label):
    vals = [v for n, v in rows if 'text_projection' not in n]
    mean = sum(vals) / len(vals)
    var  = sum((x - mean)**2 for x in vals) / (len(vals) - 1)
    print(f'\n=== {label} ===')
    print(f'  N (excluding text_proj)  : {len(vals)}')
    print(f'  Mean Frobenius norm      : {mean:.6f}')
    print(f'  Variance                 : {var:.6f}')
    print(f'  Std dev                  : {math.sqrt(var):.6f}')
    print(f'  Total sum                : {sum(vals):.6f}')

stats(clic,   'CLIC / RedCaps')
stats(cogvlm, 'CogVLM')

tp_c = [v for n, v in clic   if n == 'text_projection.weight'][0]
tp_g = [v for n, v in cogvlm if n == 'text_projection.weight'][0]
print(f'\n  text_projection CLIC  : {tp_c:.6f}')
print(f'  text_projection CogVLM: {tp_g:.6f}')

# Layer-by-layer totals
print('\n\n=== Per-layer total Frobenius norm (sum of all sub-params) ===')
print(f'{"Layer":<8}  {"CLIC":>12}  {"CogVLM":>12}  {"CogVLM-CLIC":>12}')
sums_c, sums_g = [], []
for l in range(12):
    pat = f'encoder.layers.{l}.'
    sc = sum(v for n, v in clic   if pat in n)
    sg = sum(v for n, v in cogvlm if pat in n)
    sums_c.append(sc)
    sums_g.append(sg)
    print(f'  L{l:<5}  {sc:>12.6f}  {sg:>12.6f}  {sg - sc:>+12.6f}')

top_c = sorted(range(12), key=lambda i: sums_c[i], reverse=True)[:4]
top_g = sorted(range(12), key=lambda i: sums_g[i], reverse=True)[:4]
print(f'\nTop-4 layers by total Frob norm  CLIC  : {sorted(top_c)}')
print(f'Top-4 layers by total Frob norm  CogVLM: {sorted(top_g)}')

# Component-type breakdown
print('\n\n=== Mean Frob norm by component type (averaged over 12 layers) ===')
comps = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
         'out_proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']
print(f'{"Component":<22}  {"CLIC":>12}  {"CogVLM":>12}  {"CogVLM/CLIC":>12}')
for comp in comps:
    c_v = [v for n, v in clic   if comp in n]
    g_v = [v for n, v in cogvlm if comp in n]
    cm = sum(c_v) / len(c_v)
    gm = sum(g_v) / len(g_v)
    print(f'  {comp:<20}  {cm:>12.6f}  {gm:>12.6f}  {gm/cm:>12.3f}x')
