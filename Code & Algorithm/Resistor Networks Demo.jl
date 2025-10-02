using Revise
include("Scripts.jl")

# -------------------
# Example: 4-node net
# -------------------
# nodes: 1(fixed), 2(free), 3(free), 4(fixed)
branches = [
    (1, 2, 1.0),  # g12
    (2, 3, 2.0),  # g23
    (2, 4, 3.0),  # g24
    (3, 4, 4.0)   # g34
]

free_nodes  = [2, 3]
fixed_nodes = [1, 4]

# Build block matrices
Gff, Gfc = build_blocks(branches, free_nodes, fixed_nodes)

# Voltages of fixed nodes: V1=5.0, V4=0.0
Vc = [5.0, 0.0]
# Injected currents at free nodes
If = [0.0, 0.0]

# Solve for free voltages
Vf = solve_free(Gff, Gfc, Vc, If)