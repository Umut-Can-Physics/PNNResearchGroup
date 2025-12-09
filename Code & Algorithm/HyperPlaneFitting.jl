include("Scripts.jl")
using Revise, TriangleMesh, Random
using Plots, LaTeXStrings, Graphs, GraphMakie

# TO-DO:
# - Add error messages to prevent wrong inputs and configurations

################### 
# GENERATING DATA #
###################

# inputs form of a tensor #
training_size = 1000 # number of training samples
σ = 0.5
bias = 0.1
number_of_pixels = 2 # number of input dimensions
InputRange = 0:0.001:10 # data range
# generate training data with flattned outputs
x_train, y_train, A, noise = generate_tensor_hyperplane_data(number_of_pixels, 1, training_size, InputRange; σ, bias)

##############################
# GENERATING TRIANGULAR MESH #
##############################

area_max = 0.05
num_of_inputs = 2 + 1 + 1 # dimesion of configuration space (number of fixed nodes)
# 1 bias and 1 ground node
P_network, x, y, cells, mesh = generate_uniform_mesh(area_max)
number_of_nodes = size(mesh.point_attribute, 2)
if number_of_nodes < num_of_inputs + 1 # not enough nodes to fit a hyperplane
    error("Number of nodes in the mesh is less than number of input dimensions + 1. Please increase the mesh size.")
end
scatter!(P_network, x, y, label="Nodes", markersize=4, color = :gray)

#####################################
# EXTRACTING BRANCHES FROM THE MESH #
#####################################

# random conductances assigned to edges in the mesh
branches = []
for e in eachcol(mesh.edge)
    i, j = e # neighboring two nodes
    g = rand() * 2.0 # random conductance between 0 and 2
    push!(branches, (i, j, g))
end

# choose randomly input nodes, hidden nodes, output nodes
all_nodes = collect(1:length(x))
output_node = randomly_pick_nodes(all_nodes) # pick one output node

# filter connections that ignore input-input edges
neighbors = Dict(n => Set{Int}() for n in all_nodes)
for (i, j, _) in branches
    push!(neighbors[i], j)
    push!(neighbors[j], i)
end

input_nodes = pick_nonadjacent_inputs(all_nodes, neighbors, output_node, num_of_inputs)

filtered_branches = []
for (i,j,g) in branches
    if (i in input_nodes) && (j in input_nodes)
        continue   # skip input-input edges
    end
    push!(filtered_branches, (i,j,g))
end
branches = filtered_branches

num_of_fixed_nodes = num_of_inputs
number_of_output_nodes = length([output_node])

scatter!(P_network, x[input_nodes], y[input_nodes], color = :green, markersize=8, label="Input Nodes")
scatter!(P_network, [x[output_node]], [y[output_node]], color = :blue, markersize=8, label="Output Node")

# In the free phase hidden and output nodes are free nodes (all nodes except input nodes)
free_nodes = setdiff(all_nodes, input_nodes) # hidden + output nodes
fixed_nodes = input_nodes

If = zeros(length(free_nodes)) # KCL rule at free nodes (solve all voltages at free nodes)

target_nodes = [output_node] # nodes to be clamped (I've one output node here to clamped)

gnew_list = []
P_list = []
iteration_size = 5000
α = 1e-2 # learning rate
η = 1e-4 # clamping strength
α/(2*η)
#= α = 1e-1 # learning rate
η = 1e-1 # clamping strength =#
C_list = []
for step in 1:iteration_size
    println("\n=== Training Step: ", step, " ===")
    # slelect a random training sample
    rand_position = rand(1:training_size)
    # input and outputs values for a randomly selected training sample
    target_values = [y_train[rand_position]] # target (output) voltages
    input_value = x_train[:,:, rand_position] # image pixel voltages
    # boundary conditions are input voltages
    Vc = vcat(input_value, [1.0, 0.0]) # adding bias and ground voltages
    # Hangileri pixel input hangileri bias ve ground olduğu belirsiz, ancak önemli değil.
    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α, η) 
    push!(gnew_list, gnew)
    push!(P_list, P_hist)
    push!(C_list, C)
end

ylabel = latexstring("Conductance  \$ [ \\Omega^{-1} ] \$")
P_conductances = plot(title="Training Size=$(training_size)", xlabel="Iteration", ylabel=ylabel, legend=false)
for i in 1:length(branches)
    plot!(P_conductances, map(g->g[i], gnew_list))
end
P_conductances

function moving_average(x, window)
    n = length(x)
    y = similar(x)
    for i in 1:n
        i1 = max(1, i - window + 1)
        y[i] = mean(x[i1:i])
    end
    return y
end

smooth_cost = moving_average(C_list, 150)

P_cost = plot(C_list, xlabel="Iteration", ylabel="Cost", title="Training cost", label="Instant cost", alpha=0.3)
plot!(P_cost, smooth_cost, label="Moving avg", lw=2)

# prediction for inputs that form of a tensor
pred_outputss_axis_1 = []
pred_outputss_axis_2 = []
for k in 1:training_size
    println("$(k) out of $(training_size)")
    push!(pred_outputss_axis_1, x_train[:,:,k][1])
    push!(pred_outputss_axis_2, x_train[:,:,k][2])
end
pred_outputS = zeros(Float64, (length(pred_outputss_axis_1), length(pred_outputss_axis_2)))
for xi in 1:training_size, xj in 1:training_size
    input_value = [pred_outputss_axis_1[xi],pred_outputss_axis_2[xj]]
    Vc = vcat(input_value, [1.0, 0.0])
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If) # it use final conductances after training
    V  = full_voltage(free_nodes, fixed_nodes, Vf, Vc) 
    pred_outputS[xj, xi] = V[output_node] 
end
# prediction for inputs that form of a tensor (alternative way)
PredictedResults = []
for k in 1:training_size
    println("$(k) out of $(training_size)")
    input_value = [pred_outputss_axis_1[k],pred_outputss_axis_2[k]]
    Vc = vcat(input_value, [1.0, 0.0])
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If) # it use final conductances after training
    V  = full_voltage(free_nodes, fixed_nodes, Vf, Vc)
    push!(PredictedResults, V[output_node])
end

##############################################################
# TRUE VS PREDICTED PLANE PLOT for inputs that form a vector #
##############################################################

# Extract input coordinates from training data

x1 = [pred_outputss_axis_1[i] for i in 1:training_size]
x2 = [pred_outputss_axis_2[i] for i in 1:training_size]

# Create a grid over the input domain
xs = range(minimum(x1), maximum(x1), length=training_size)
ys = range(minimum(x2), maximum(x2), length=training_size)

z_true = y_train

plotlyjs()
gr()
P_plane = scatter3d(
    pred_outputss_axis_1, pred_outputss_axis_2, z_true;
    alpha = 0.4,
    xlabel = "x₁",
    ylabel = "x₂",
    zlabel = "y",
    title = "True Plane vs Predicted Outputs",
    label = "True plane",
    camera = (0, 0),
    marker = (:diamond, 2,:red)
)

# alternative way to overlay predicted outputs as points
scatter3d!(
    P_plane,
    pred_outputss_axis_1, pred_outputss_axis_2, PredictedResults;
    markersize = 4,
    label = "Predicted outputs",
    marker =(:star5, 2, :blue)
)

# Find a way to check the maximum error between true plane and predicted outputs
# z_true noise olmadan data olmalı
errors = abs.(z_true .- noise .- PredictedResults)
mean_error = mean(errors)^2

P_all = plot(
    P_network,
    P_conductances,
    P_cost,
    P_plane;
    layout = (2, 2),
    size   = (2000, 1100)  # adjust as you like
)

savefig(P_all, "figures/P_All.png")