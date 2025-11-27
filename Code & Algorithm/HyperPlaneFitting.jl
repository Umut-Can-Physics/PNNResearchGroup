include("Scripts.jl")
using Revise, TriangleMesh, Random
using Plots, LaTeXStrings, Graphs, GraphMakie

# TO-DO:
# - Add error messages to prevent wrong inputs and configurations

# FIX:
# - All predicted outputs doesn't cover true plane for two inputs

################### 
# GENERATING DATA #
###################

# inputs form of a tensor #
training_size = 500 # number of training samples
σ = 0
num_of_inputs = 2 # dimesion of configuration space (number of fixed nodes)
InputRange = 0:0.001:20 # data range
x_train, y_train, A, noise = generate_tensor_hyperplane_data(2, 1, training_size, InputRange; σ)

# inputs form of a vector
#= x_train_1 = range(start = 0, stop = 10, length = training_size)
x_train_2 = range(start = 11, stop = 20, length = training_size)
A1 = 0.1
A2 = 0.9
y_train_0 = A1*x_train_1.+ A2*x_train_2
collect(y_train_0) =#
# end of data #

##############################
# GENERATING TRIANGULAR MESH #
##############################

area_max = 0.05
P, x, y, cells, mesh = generate_uniform_mesh(area_max)
number_of_nodes = size(mesh.point_attribute, 2)
if number_of_nodes < num_of_inputs + 1 # not enough nodes to fit a hyperplane
    error("Number of nodes in the mesh is less than number of input dimensions + 1. Please increase the mesh size.")
end
scatter!(P, x, y, label="Nodes", markersize=4, color = :gray)

#####################################
# EXTRACTING BRANCHES FROM THE MESH #
#####################################

# random conductances assigned to edges in the mesh
branches = []
for e in eachcol(mesh.edge)
    i, j = e # neighboring two nodes
    g = rand() * 5.0 # random conductance between 0 and 10
    push!(branches, (i, j, g))
end

# choose randomly input nodes, hidden nodes, output nodes
all_nodes = collect(1:length(x))
output_node = randomly_pick_nodes(all_nodes) # just one output node

# filter input-input connections
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

scatter!(P, x[input_nodes], y[input_nodes], color = :green, markersize=8, label="Input Nodes")
scatter!(P, [x[output_node]], [y[output_node]], color = :blue, markersize=8, label="Output Node")

# In the free phase hidden and output nodes are free nodes (all nodes expect input nodes)
free_nodes = setdiff(all_nodes, input_nodes)
fixed_nodes = input_nodes

If = zeros(length(free_nodes)) # KCL rule at free nodes (solve all voltages at free nodes)

target_nodes = [output_node] # nodes to be clamped (I've one output node here to clamped)

gnew_list = []
P_list = []
iteration_size = 150
α = 1e-1 # learning rate
η = 1e-1 # clamping strength
C_list = []
for step in 1:iteration_size
    println("\n=== Training Step: ", step, " ===")
    # slelect a random training sample
    rand_position = rand(1:training_size)
    # input and outputs values for a randomly selected training sample
    target_values = [y_train[rand_position]] # target (output) voltages
    input_value = x_train[:,:, rand_position] # input voltages
    #input_value = [x_train_1[rand_position],x_train_2[rand_position]] # input voltages
    # boundary conditions are input voltages
    #Vc = vec(input_value) # [V1 (input)]
    Vc = input_value
    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α, η) 
    push!(gnew_list, gnew)
    push!(P_list, P_hist)
    push!(C_list, C)
end

ylabel = latexstring("Conductance  \$ [ \\Omega^{-1} ] \$")
p1 = plot(title="Training Size=$(training_size)", xlabel="Iteration", ylabel=ylabel, legend=false)
for i in 1:length(branches)
    plot!(map(g->g[i], gnew_list))
end
p1

function moving_average(x, window)
    n = length(x)
    y = similar(x)
    for i in 1:n
        i1 = max(1, i - window + 1)
        y[i] = mean(x[i1:i])
    end
    return y
end

smooth_cost = moving_average(C_list, 100)  # 100 iterasyonluk pencere

# prediction for inputs that form of a vector
#pred_outputs = zeros(Float64, (length(x_train_1), length(x_train_2)))

# prediction for inputs that form of a tensor
pred_outputss_axis_1 = []
pred_outputss_axis_2 = []
for k in 1:training_size
    push!(pred_outputss_axis_1, x_train[:,:,k][1])
    push!(pred_outputss_axis_2, x_train[:,:,k][2])
end
pred_outputS = zeros(Float64, (length(pred_outputss_axis_1), length(pred_outputss_axis_2)))

for xi in 1:training_size, xj in 1:training_size
    #input_value = [x_train_1[xi],x_train_2[xj]]
    input_value = [pred_outputss_axis_1[xi],pred_outputss_axis_2[xj]]
    Vc = input_value
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If) # it use final conductances after training
    V  = full_voltage(free_nodes, fixed_nodes, Vf, Vc) 
    #pred_outputs[xj, xi] = V[output_node] 
    pred_outputS[xj, xi] = V[output_node] 
end

# prediction for inputs that form of a tensor (alternative way)
PredictedResults = []
for k in 1:training_size
    input_value = [pred_outputss_axis_1[k],pred_outputss_axis_2[k]]
    Vc = input_value
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If) # it use final conductances after training
    V  = full_voltage(free_nodes, fixed_nodes, Vf, Vc)
    push!(PredictedResults, V[output_node])
end

#= pred_outputs = []
for xi in 1:training_size
    input_value = x_train[:,:,xi] 
    Vc = vec(input_value) 
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If) # it use final conductances after training
    V  = full_voltage(free_nodes, fixed_nodes, Vf, Vc)
    push!(pred_outputs, V[output_node]) 
end =#

#= errors = y_train .- pred_outputs
println("MSE = ", mean(errors.^2))
=#

pC = plot(C_list, xlabel="Iteration", ylabel="Cost", title="Training cost", label="Instant cost", alpha=0.3)
plot!(smooth_cost, label="Moving avg", lw=2)

##############################################################
# TRUE VS PREDICTED PLANE PLOT for inputs that form a vector #
##############################################################

# Extract input coordinates from training data

x1 = [pred_outputss_axis_1[i] for i in 1:training_size]
x2 = [pred_outputss_axis_2[i] for i in 1:training_size]

# Create a grid over the input domain
xs = range(minimum(x1), maximum(x1), length=training_size)
ys = range(minimum(x2), maximum(x2), length=training_size)

# True plane: y = A₁ x₁ + A₂ x₂
#z_true = [A[1]*xx + A[2]*yy for yy in ys, xx in xs]
#z_true = [A1*xx + A2*yy for yy in x_train_2, xx in x_train_1]
z_true = y_train

plotlyjs()
p_plane = scatter3d(
    pred_outputss_axis_1, pred_outputss_axis_2, z_true;
    alpha = 0.4,
    xlabel = "x₁",
    ylabel = "x₂",
    zlabel = "y",
    title = "True Plane vs Predicted Outputs",
    label = "True plane",
    camera = (0, 0)
)

# Overlay predicted outputs as points
#= wireframe!(
    p_plane,
    pred_outputss_axis_1, pred_outputss_axis_2, pred_outputS;
    markersize = 4,
    label = "Predicted outputs"
)
 =#
# alternative way to overlay predicted outputs as points
scatter3d!(
    p_plane,
    pred_outputss_axis_1, pred_outputss_axis_2, PredictedResults;
    markersize = 4,
    label = "Predicted outputs"
)

# Find a way to check the maximum error between true plane and predicted outputs
errors = abs.(z_true .- PredictedResults)
mean_error = mean(errors)^2