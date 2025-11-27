using Revise
include("Scripts.jl")

###################
# Example: 3 node #
###################

# nodes: 1(fixed), 2(free), 3(free), 4(fixed)
branches = [
    (1, 2, 1.0),  # g12
    (2, 3, 2.0),  # g23
]

free_nodes  = [2]
fixed_nodes = [1, 3]

# Voltages of fixed nodes: V1=5.0, V4=0.0
Vc = [5.0, 0.0]
# Total currents at free nodes
If = [0.0]

# Solve for free voltages
Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)

###############################################
# More complex example in Resistor Networks.jl#
###############################################

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

# Voltages of fixed nodes: V1=5.0, V6=0.0
Vc = [5.0, 0.0]
If = zeros(number_of_free_nodes)

# Solve for free voltages
Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)

###################
# Another example #
###################

number_of_free_nodes = 3

branches = [
    (1, 2, 1.0),
    (1, 3, 1.0),

    (2, 3, 1.0),
    (2, 4, 1.0),

    (3, 4, 1.0),
    (3, 5, 1.0),
    (3, 6, 1.0),

    (4, 6, 1.0),

    (5, 6, 1.0)
]

free_nodes  = [2, 3, 6]
fixed_nodes = [1, 4, 5]

Vc = [1.0, 0.0, 1.0]
If = zeros(number_of_free_nodes)

Gff, Gfc = build_blocks(branches, free_nodes, fixed_nodes)

Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)

################
# CUBE NETWORK #
################

number_of_free_nodes = 6

branches = [
    (1, 2, 1.0),
    (1, 4, 1.0),
    (1, 5, 1.0),

    (2, 3, 1.0),
    (2, 6, 1.0),

    (3, 4, 1.0),
    (3, 7, 1.0),

    (4, 8, 1.0),

    (5, 6, 1.0),
    (5, 8, 1.0),

    (6, 7, 1.0),
    
    (7, 8, 1.0)
]
free_nodes  = [2, 3, 4, 5, 6, 8]
fixed_nodes = [1, 7]

# Voltages of fixed nodes: V1=0.0, V7=2.0
Vc = [1.0, 0.0]
If = zeros(number_of_free_nodes)

Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)

# CHECK THE CUBE NETWORK
Gff, Gfc = build_blocks(branches, free_nodes, fixed_nodes)
Gff = Int.(Gff)

G_free_free = [15 -5 0 0 -7 0;
                -5 17 -7 0 0 0;
                0 -7 17 0 0 -9;
                0 0 0 11 -1 -2;
                -7 0 0 -1 9 0;
                0 0 -9 -2 0 12
]

issymmetric(G_free_free)

G_free_fixed = [-3 0;
                0 -5;
                -1 0;
                -8 0;
                0 -1;
                0 -1

]

V_fixed = [0.0; 2.0]

Vf_check = inv(G_free_free) * (- G_free_fixed * V_fixed )

Vf_check == Vf