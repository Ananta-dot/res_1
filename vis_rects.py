import numpy as np, math, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ======= parameters (keep identical to training script) ======================
n, k, BITS_PER_COLOR = 14, 2, 2
def get_batcher_oe_comparators_py(n_):
    comps, p = [], 1
    while p < n_:
        k_ = p
        while k_:
            j_start, j_end, step = k_ % p, n_ - 1 - k_, 2*k_
            for j in range(j_start, j_end+1, step):
                for i in range(min(k_-1, n_ - j - k_ - 1)+1):
                    if (i+j)//(2*p) == (i+j+k_)//(2*p):
                        comps.append((i+j, i+j+k_))
            k_//=2
        p*=2
    return comps
base_len, comps, m = 2*n, get_batcher_oe_comparators_py(2*n), None
m = len(comps)                              # m depends on n

# ======= helpers ==============================================================

def build_base_array(n_):
    return np.repeat(np.arange(1,n_+1),2)

def apply_comps(arr, bits):
    arr = arr.copy()
    for bit,(a,b) in zip(bits, comps):
        if bit: arr[a], arr[b] = arr[b], arr[a]
    return arr

def bits_to_colors(bits):
    return [(bits[2*i]<<1)|bits[2*i+1] for i in range(len(bits)//2)]

# strict version: same-color rectangles never overlap vertically
def height_correct(rects, colors):
    new = rects.copy()
    by_color = {}
    for idx,(_, (y1,_)) in enumerate(rects):
        by_color.setdefault(colors[idx], []).append((y1, idx))
    for lst in by_color.values():
        lst.sort()                                  # bottom-to-top sweep
        for r,(y1_idx, i) in enumerate(lst):
            (x1,x2),(y1,y2) = new[i]
            next_bottom = lst[r+1][0] if r+1<len(lst) else math.inf
            limit = max(1, next_bottom - y1)        # room below next rect
            new[i] = ((x1,x2),(y1, y1+min(y2-y1, limit-1)))
    return new

# ======= generate *any* bitstring  (here we pick random) =====================
DEC    = 2*m + 2*m + n*BITS_PER_COLOR
bits   = np.random.randint(0,2,DEC)
h_bits = bits[:m];  v_bits = bits[m:2*m];  c_bits = bits[2*m:]
colors = bits_to_colors(c_bits)

# ======= decode to rectangles ===============================================
base   = build_base_array(n)
arrH   = apply_comps(base, h_bits)
arrV   = apply_comps(base, v_bits)

rects_original = []
for lab in range(1,n+1):
    xs=[i for i,v in enumerate(arrH) if v==lab]
    ys=[i for i,v in enumerate(arrV) if v==lab]
    rects_original.append(((min(xs),max(xs)),(min(ys),max(ys))))   # (x1,x2),(y1,y2)

# Apply height correction
rects_corrected = height_correct(rects_original, colors)

# ======= print rectangle data ===============================================
print("\n=== BEFORE Height Correction ===")
header = "{:>3}  br(x,y)   tl(x,y)   Color".format("Idx")
print(header)
for i, ((x1, x2), (y1, y2)) in enumerate(rects_original):
    print("{:3}  ({:2},{:2})  ({:2},{:2})    {}".format(
        i, x1, y1, x2, y2, colors[i]))

print("\n=== AFTER Height Correction ===")
print(header)
for i, ((x1, x2), (y1, y2)) in enumerate(rects_corrected):
    print("{:3}  ({:2},{:2})  ({:2},{:2})    {}".format(
        i, x1, y1, x2, y2, colors[i]))
    
# ======= plot both versions side by side ====================================
palette = {0:'#e41a1c',1:'#4daf4a',2:'#377eb8',3:'#ff7f00'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot BEFORE height correction
for i,(((x1,x2),(y1,y2)),col) in enumerate(zip(rects_original,colors)):
    ax1.add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                           facecolor=palette[col], edgecolor='black', alpha=.5))
    ax1.text((x1+x2)/2,(y1+y2)/2, str(i), ha='center', va='center',
            color='black', fontweight='bold', fontsize=9)

ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('BEFORE Height Correction')
ax1.autoscale_view()

# Plot AFTER height correction
for i,(((x1,x2),(y1,y2)),col) in enumerate(zip(rects_corrected,colors)):
    ax2.add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                           facecolor=palette[col], edgecolor='black', alpha=.5))
    ax2.text((x1+x2)/2,(y1+y2)/2, str(i), ha='center', va='center',
            color='black', fontweight='bold', fontsize=9)

ax2.set_aspect('equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('AFTER Height Correction')
ax2.autoscale_view()

plt.tight_layout()
plt.show()
