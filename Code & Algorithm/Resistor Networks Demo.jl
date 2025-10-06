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

# More complex example in Resistor Networks.jl

number_of_free_nodes = 4

# (i, j, g_ij) for branch between symmetric nodes i and j with conductance g_ij
branches = [
    (1, 2, only(rand(1))),
    
    (2, 3, only(rand(1))),
    (2, 4, only(rand(1))),
    (2, 5, only(rand(1))),

    (3, 4, only(rand(1))),
    (3, 6, only(rand(1))),

    (4, 6, only(rand(1))),

    (5, 4, only(rand(1))),
    (5, 6, only(rand(1)))
]

free_nodes  = [2, 3, 4, 5]
fixed_nodes = [1, 6]

Gff, Gfc = build_blocks(branches, free_nodes, fixed_nodes)

# Voltages of fixed nodes: V1=5.0, V6=0.0
Vc = [5.0, 0.0]
# Injected currents at free nodes
If = zeros(number_of_free_nodes)

# Solve for free voltages
Vf = solve_free(Gff, Gfc, Vc, If)