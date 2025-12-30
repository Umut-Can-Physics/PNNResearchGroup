include("Scripts.jl")
using Revise, Random
using Plots, LaTeXStrings

# inputs form of a tensor #
training_size = 1000 # number of training samples
epoch_size = 100
σ = 0.5 # noise standard deviation
bias = 0.1
number_of_pixels = 15 # number of input dimensions
α = 1e-1 # learning rate
η = 1e-1 # clamping strength
area_max = 0.01 # max area for triangular mesh elements
α/(2*η)

################### 
# GENERATING DATA #
###################


InputRange = 0:0.001:10 # data range
# generate training data with flattned outputs
x_train, y_train, A, noise = generate_tensor_hyperplane_data(number_of_pixels, 1, training_size, InputRange; σ, bias)




using Triangulate
using Random
using Plots

##############################
# GENERATING TRIANGULAR MESH #
##############################

num_of_inputs = number_of_pixels + 1 + 1 

# Helper to recreate your 'generate_uniform_mesh' using Triangulate.jl
function generate_triangulate_mesh(area_max)
    # Define a simple bounding box (0,0) to (1,1) for the mesh
    # Triangulate requires points and segments to define the domain
    nodes = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]'
    segs = Int32[1 2; 2 3; 3 4; 4 1]'
    
    tin = TriangulateIO()
    tin.pointlist = nodes
    tin.segmentlist = segs
    
    # "p" for PSLG, "q" for quality, "a" for area constraint, "Q" for quiet
    switches = "pqea$(area_max)Q"
    (tout, vorout) = triangulate(switches, tin)
    
    # Extracting x, y for plotting
    x = tout.pointlist[1, :]
    y = tout.pointlist[2, :]
    
    # Create a base plot
    P_network = plot(aspect_ratio=:equal)
    
    return P_network, x, y, tout
end

P_network, x, y, mesh = generate_triangulate_mesh(area_max)

# In Triangulate.jl, point attributes are in pointattributelist
# If no attributes were defined, we check the number of columns in pointlist
number_of_nodes = size(mesh.pointlist, 2)

if number_of_nodes < num_of_inputs + 1
    error("Number of nodes in the mesh is less than number of input dimensions + 1.")
end

scatter!(P_network, x, y, label="Hidden Nodes", markersize=4, color = :gray)

#####################################
# EXTRACTING BRANCHES FROM THE MESH #
#####################################

# Triangulate.jl returns edges in 'edgelist' if requested, 
# but usually we extract them from the triangle connectivity (trianglelist) 
# or use the segmentlist for boundary edges.
branches = []

# If you specifically need all internal edges, it is safest to iterate 
# through trianglelist. Here is the standard way to get unique edges:
unique_edges = Set{Tuple{Int, Int}}()
for col in 1:size(mesh.trianglelist, 2)
    t = mesh.trianglelist[:, col]
    # Triangles have 3 edges: (1,2), (2,3), (3,1)
    for (a, b) in [(t[1], t[2]), (t[2], t[3]), (t[3], t[1])]
        push!(unique_edges, a < b ? (a, b) : (b, a))
    end
end

for (i, j) in unique_edges
    g = rand() * 2.0 
    push!(branches, (Int(i), Int(j), g))
end

# choose randomly input nodes, hidden nodes, output nodes
all_nodes = collect(1:length(x))
output_node = rand(all_nodes) # Simplified pick

# filter connections that ignore input-input edges
neighbors = Dict(n => Set{Int}() for n in all_nodes)
for (i, j, _) in branches
    push!(neighbors[i], j)
    push!(neighbors[j], i)
end

# Using your provided function (ensure it is defined in your scope)
input_nodes = pick_nonadjacent_inputs(all_nodes, neighbors, output_node, num_of_inputs)

filtered_branches = []
for (i, j, g) in branches
    if (i in input_nodes) && (j in input_nodes)
        continue   
    end
    push!(filtered_branches, (i, j, g))
end
branches = filtered_branches

scatter!(P_network, x[input_nodes], y[input_nodes], color = :green, markersize=4, label="Input Nodes")
scatter!(P_network, [x[output_node]], [y[output_node]], color = :blue, markersize=4, label="Output Node")

display(P_network)




# In the free phase hidden and output nodes are free nodes (all nodes except input nodes)
free_nodes = setdiff(all_nodes, input_nodes) # hidden + output nodes
fixed_nodes = input_nodes

If = zeros(length(free_nodes)) # KCL rule at free nodes (solve all voltages at free nodes)

target_nodes = [output_node] # nodes to be clamped (I've one output node here to clamped)

gnew_list = []
P_list = []

random_ordered_indices = randperm(length(y_train))
shuffle_ytrain = y_train[random_ordered_indices]
shuffle_xtrain = x_train[:, :, random_ordered_indices]

C_list = []
for epoch in 1:epoch_size
    println("\n=== Epoch Step: ", epoch, " ===")
    for step in 1:length(y_train)
        # boundary conditions are input voltages
        Vc = vcat(shuffle_xtrain[:, :, step], [1.0, 0.0]) # adding bias and ground voltages
        # Hangileri pixel input hangileri bias ve ground olduğu belirsiz, ancak önemli değil.
        C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, [shuffle_ytrain[step]]; α, η) 
        push!(gnew_list, gnew)
        push!(P_list, P_hist)
        push!(C_list, C)
    end
end

ylabel = latexstring("Conductance  \$ [ \\Omega^{-1} ] \$")
P_conductances = plot(title="Training Size=$(training_size)", xlabel="Iteration", ylabel=ylabel, legend=false)
for i in 1:length(branches)
    plot!(P_conductances, map(g->g[i], gnew_list))
end

#display(P_conductances)

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