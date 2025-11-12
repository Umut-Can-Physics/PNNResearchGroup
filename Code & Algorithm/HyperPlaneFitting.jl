include("Scripts.jl")
using Revise, TriangleMesh
using Plots, LaTeXStrings, Graphs, GraphMakie

###################
# GENERATING DATA #
###################

training_size = 10
σ = 0.1
dim = 10 # dimesion of configuration space
x_train, y_train = generate_tensor_data(σ, training_size, dim)

##############################
# GENERATING TRIANGULAR MESH #
##############################

area_max = 0.05
P, x, y, cells, mesh = generate_uniform_mesh(area_max)
number_of_nodes = size(mesh.point_attribute, 2)
scatter!(P, x, y, label="Nodes", markersize=4, color = :gray)

#####################################
# EXTRACTING BRANCHES FROM THE MESH #
#####################################

# random conductances assigned to edges in the mesh
branches = []
for e in eachcol(mesh.edge)
    i, j = e # neighboring two nodes
    g = rand() * 10.0 # random conductance between 0 and 10
    push!(branches, (i, j, g))
end

# choose randomly input nodes, hidden nodes, output nodes
all_nodes = collect(1:length(x))
randomly_pick_nodes(all_nodes) = all_nodes[rand(1:length(all_nodes))]
output_node = randomly_pick_nodes(all_nodes)
num_of_fixed_nodes = dim
number_of_output_nodes = 1 
input_nodes = []
while true
    rand_pick = randomly_pick_nodes(all_nodes)
    if rand_pick ∉ input_nodes && rand_pick != output_node
        push!(input_nodes, rand_pick)
    end
    if length(input_nodes) == num_of_fixed_nodes
        break                           
    end
end
input_nodes

scatter!(P, x[input_nodes], y[input_nodes], color = :green, markersize=8, label="Input Nodes")
scatter!(P, [x[output_node]], [y[output_node]], color = :blue, markersize=8, label="Output Node")

# In the free phase hidden and output nodes are free nodes (all nodes expect input nodes)
free_nodes = setdiff(all_nodes, input_nodes)
fixed_nodes = input_nodes

If = zeros(length(free_nodes)) # KCL rule at free nodes

target_nodes = free_nodes # nodes to be clamped 

gnew_list = []
P_list = []
iteration_size = 100
α = 1e-1 # learning rate
η = 1 # clamping strength
for step in 1:iteration_size
    rand_position = rand(1:dim)
    target_value = y_train[rand_position] # target (output) voltages
    input_value = x_train[:,:, rand_position] # input voltages
    Vc = vec(input_value) # [V1 (input)]
    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_value; α, η) 
    push!(gnew_list, gnew)
    push!(P_list, P_hist)
end