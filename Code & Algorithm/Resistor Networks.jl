using Revise
include("Scripts.jl")

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