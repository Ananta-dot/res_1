import matplotlib.pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpContinuous, value

def solve_disjoint_rectangles_pulp(rectangles):
    n = len(rectangles)

    # Generate candidate points (cell centers inside rectangles)
    candidate_points = set()
    for (x1, x2), (y1, y2) in rectangles:
        for x in range(x1, x2):
            for y in range(y1, y2):
                candidate_points.add((x + 0.5, y + 0.5))

    # Find overlapping rectangle indices for each candidate point
    constraints = []
    for point in candidate_points:
        x, y = point
        covers = []
        for i, ((x1, x2), (y1, y2)) in enumerate(rectangles):
            if x1 <= x <= x2 and y1 <= y <= y2:
                covers.append(i)
        if len(covers) > 2:
            return 0.0, 0.0, 0.0
        elif len(covers) == 2:
            constraints.append(tuple(covers))

    # LP Model (Relaxed)
    lp_model = LpProblem("Max_Disjoint_LP", LpMaximize)
    x_vars_lp = [LpVariable(f"x_{i}", 0, 1, LpContinuous) for i in range(n)]
    lp_model += lpSum(x_vars_lp)
    for i, j in constraints:
        lp_model += x_vars_lp[i] + x_vars_lp[j] <= 1
    lp_model.solve()
    lp_val = value(lp_model.objective)

    # ILP Model (Binary)
    ilp_model = LpProblem("Max_Disjoint_ILP", LpMaximize)
    x_vars_ilp = [LpVariable(f"x_{i}", 0, 1, LpBinary) for i in range(n)]
    ilp_model += lpSum(x_vars_ilp)
    for i, j in constraints:
        ilp_model += x_vars_ilp[i] + x_vars_ilp[j] <= 1
    ilp_model.solve()
    ilp_val = value(ilp_model.objective)

    ratio = 0.0 if ilp_val < 1e-6 else lp_val / ilp_val
    return lp_val, ilp_val, ratio

def hash_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index + 1)
    for key in index_dict:
        index_dict[key] = tuple(index_dict[key])
    return index_dict

def calc_score_pulp(arrH, arrV, n):
    result_h = hash_indices(arrH)
    result_v = hash_indices(arrV)

    keys = sorted(set(result_h.keys()) & set(result_v.keys()))
    rectangles = []
    for key in keys:
        h_start, h_end = result_h[key]
        v_start, v_end = result_v[key]
        rectangles.append(((h_start, h_end), (v_start, v_end)))

    lp_val, ilp_val, ratio = solve_disjoint_rectangles_pulp(rectangles)
    print(f"LP: {lp_val}, ILP: {ilp_val}, Ratio: {ratio:.2f}")

    # Plotting
    color_list = ['red', 'orange', 'blue', 'cyan', 'yellow', 'green', 'magenta', 'purple']
    fig, ax = plt.subplots()
    for i, rect in enumerate(rectangles):
        (x1, x2), (y1, y2) = rect
        width = x2 - x1
        height = y2 - y1
        color = color_list[i % len(color_list)]
        patch = plt.Rectangle((x1, y1), width, height, facecolor=color, edgecolor='black', alpha=0.6)
        ax.add_patch(patch)
        ax.text(x1 + width / 2, y1 + height / 2, str(keys[i]), color='black', ha='center', va='center', fontsize=8)

    ax.set_xlim(0, len(arrH))
    ax.set_ylim(0, len(arrV))
    ax.set_xlabel('Horizontal')
    ax.set_ylabel('Vertical')
    ax.set_title('Rectangles Visualization (PuLP)')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# === Example ===
arrH = [3, 8, 6, 8, 4, 5, 6, 1, 1, 2, 7, 3, 2, 4, 5, 7]
arrV = [4, 8, 7, 1, 6, 4, 6, 3, 3, 8, 2, 7, 5, 5, 1, 2]
n = 8

calc_score_pulp(arrH, arrV, n)
