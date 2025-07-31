import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_disjoint_rectangles(rectangles):
    n_rect = len(rectangles)

    # Generate candidate points (centers of unit cells inside rectangles)
    candidate_points = set()
    for (x1, x2), (y1, y2) in rectangles:
        for x in range(x1, x2):
            for y in range(y1, y2):
                candidate_points.add((x + 0.5, y + 0.5))

    # Overlap constraints
    constraints = []
    for point in candidate_points:
        x, y = point
        covering = []
        for i, ((x1, x2), (y1, y2)) in enumerate(rectangles):
            if x1 <= x <= x2 and y1 <= y <= y2:
                covering.append(i)
        if len(covering) > 2:
            return 0.0, 0.0, 0.0
        elif len(covering) == 2:
            constraints.append(covering)

    # LP model
    lp_val = 0.0
    try:
        model_lp = gp.Model("maxDisjointRectangles_LP")
        model_lp.setParam('OutputFlag', 0)
        x_lp = model_lp.addVars(n_rect, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
        model_lp.setObjective(gp.quicksum(x_lp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_lp.addConstr(gp.quicksum(x_lp[i] for i in covers) <= 1)
        model_lp.optimize()
        if model_lp.status == GRB.OPTIMAL:
            lp_val = model_lp.objVal
            for v in model_lp.getVars():
                print(f"Variable: {v.VarName}, Type: {v.VType}, Value: {v.X}")
    except Exception as e:
        print("LP Exception:", e)

    # ILP model
    ilp_val = 0.0
    try:
        model_ilp = gp.Model("maxDisjointRectangles_ILP")
        model_ilp.setParam('OutputFlag', 0)
        x_ilp = model_ilp.addVars(n_rect, vtype=GRB.BINARY, name="x")
        model_ilp.setObjective(gp.quicksum(x_ilp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_ilp.addConstr(gp.quicksum(x_ilp[i] for i in covers) <= 1)
        model_ilp.optimize()
        if model_ilp.status == GRB.OPTIMAL:
            ilp_val = model_ilp.objVal
            for v in model_ilp.getVars():
                print(f"Variable: {v.VarName}, Type: {v.VType}, Value: {v.X}")
    except Exception as e:
        print("ILP Exception:", e)

    ratio = 0 if ilp_val < 1e-9 else lp_val / ilp_val
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

def calc_score(arrH, arrV, n):
    result_h = hash_indices(arrH)
    result_v = hash_indices(arrV)

    keys = sorted(set(result_h.keys()) & set(result_v.keys()))
    rectangles = []
    for key in keys:
        h_start, h_end = result_h[key]
        v_start, v_end = result_v[key]
        rectangles.append(((h_start, h_end), (v_start, v_end)))

    lp_val, ilp_val, ratio = solve_disjoint_rectangles(rectangles)
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
    ax.set_title('Rectangles Visualization')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# === Example ===
arrH = [3, 8, 6, 8, 4, 5, 6, 1, 1, 2, 7, 3, 2, 4, 5, 7]
arrV = [4, 8, 7, 1, 6, 4, 6, 3, 3, 8, 2, 7, 5, 5, 1, 2]
n = 8

calc_score(arrH, arrV, n)
