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

number_of_free_nodes = 2

branches = [
    (1, 2, 4.0),
    (2, 3, 7.0),
    (2, 4, 5.0),
    (3, 4, 3.0),
    (1, 4, 10.0),
]
free_nodes  = [2, 4]
fixed_nodes = [1, 3]

Vc = [5.0, 0.0]
If = zeros(number_of_free_nodes)

Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)