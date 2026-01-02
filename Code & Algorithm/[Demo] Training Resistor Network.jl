using Revise, Logging
include("Scripts.jl")

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

V_out_free, V_hidden_free = Vf

println("Free phase:")
println("  V_hidden = $V_hidden_free")
println("  V_output = $V_out_free")

Full_Voltage = full_voltage(free_nodes, fixed_nodes, Vf, Vc)

# 2. Clamped Phase

η = 1e-1
target_nodes = [2] # nodes to be clamped
target_values = [7] # target voltages
VtC   = [Full_Voltage[n] + η*(pt - Full_Voltage[n]) for (n,pt) in zip(target_nodes, target_values)]

free2  = [n for n in free_nodes if !(n in target_nodes)] # remaining free nodes after clamping
fixed2 = vcat(fixed_nodes, target_nodes) # set target nodes as temporary fixed nodes (add targets to fixed nodes)
Vc2    = vcat(Vc, VtC) # add fixed node voltages

If2 = zeros(length(free2))
Vf2 = solve_free(branches, free2, fixed2, Vc2, If2)

Vclamped = full_voltage(free2, fixed2, Vf2, Vc2)

println("\nClamped phase:")
println("  V_hidden = $Vf2")
println("  V_output (clamped) = $(Vc2[end])")

ΔVF = branch_dV3(branches, Full_Voltage)
ΔVC = branch_dV3(branches, Vclamped)

α = 1e-2
kmin = 1e-6

# since update is iterative, we need to run every time by exclamation mark (!) indicating in-place modification
# exclamation mark (!): Modifies original in-place, more memory efficient, but changes the input
for n in eachindex(branches) 
    i, j, g = branches[n] 
    Δg = (α/(2η)) * ((ΔVF[n]^2) - (ΔVC[n]^2)) # Equation (4)
    #gnew = max(g + Δg, kmin) # gnew can't be negative
    branches[n] = (i, j, g + Δg)
end

# 3. Free Phase after weight update (We remove clamping and solve again)
Vf_new = solve_free(branches, free_nodes, fixed_nodes, Vc, If)

println("Free phase after updated conductance:")
println("  V_hidden = ", Vf_new[1])
println("  V_output = ",  Vf_new[2])

# And so on...

# Iterative Version of the same circuit training

using Revise, Plots, LaTeXStrings
include("Scripts.jl")

number_of_free_nodes = 2
If = zeros(number_of_free_nodes)

branches = [
    (1, 2, 4.0),
    (2, 3, 7.0),
    (2, 4, 5.0),
    (3, 4, 3.0),
    (1, 4, 10.0),
]
free_nodes  = [2, 4]
fixed_nodes = [1, 3]
Vc = [10.0, 0.0]
target_nodes = [2] # nodes to be clamped
target_values = [3] # target voltages

epoch_size = 5000
α  = 1e-4
η  = 1e-3
CList = []
for epoch in 1:epoch_size

    C, P_hist, gnew = train_step!(
        branches, free_nodes, fixed_nodes, Vc, If, target_nodes,
        target_values; α, η
    )

    push!(CList, C)

end

ylabel = L"\mathrm{Cost} = \frac{1}{2} (V_{out}^{free} - V_{out}^{target})^2"

plot(CList , xlabel="Iterations", ylabel=ylabel, title="Training Curve", legend=false)

savefig("figures/Training_Curve_Resistor_Network_Iterative.pdf")

# Full Training Loop with Data

using Revise, Plots, Random
include("Scripts.jl")

a_true = 0.4
b_true = 1.2
xs = collect(range(0.0, 10.0, length=100))
ys = a_true .* xs .+ b_true

number_of_free_nodes = 1
If = zeros(number_of_free_nodes)

free_nodes  = [2] # free node as a output node
fixed_nodes = [1, 3, 4] # hidden node is now fixed node
target_nodes = [2] # nodes to be clamped

epoch_size_list = [10, 50, 100, 1000, 1500]
α  = 1e-4
η  = 1e-3
MainCList = []

for epoch_size in epoch_size_list

    branches = [
    (1, 2, 4.0),
    (2, 3, 7.0),
    (2, 4, 5.0),
    (3, 4, 3.0),
    (1, 4, 10.0),
] # reset branches for each training size
    CList = Float64[]
    
    for epoch in 1:epoch_size
        # shuffle dataset each epoch
        perm = randperm(length(xs))
        C_epoch = 0.0

        for k in perm
            x = xs[k]
            y = ys[k]

            # input: V1 = 0, V3 = 5 (bias node), V4 = x (input node)
            Vc = [0.0, 5.0, x]            # fixed_nodes order: [4, 1, 3]
            target_values = [y]           # clamp node-2 to y

            C, P_hist, gnew = train_step!(
                branches, free_nodes, fixed_nodes, Vc, If,
                target_nodes, target_values; α, η
            )

            C_epoch += C

        end

        push!(CList, C_epoch/length(xs)) # Mean Squared Error
    end

    push!(MainCList, CList)
end

ylabel = L"\mathrm{MSE}\ = \frac{1}{2N} \sum_{i=1}^{N} (V_{out,i}^{free} - V_{out,i}^{target})^2"

plot(MainCList[1] , xlabel="Epoch", ylabel=ylabel, title="Training Curve", 
marker=:o, markersize=10, label="Epochs=$(epoch_size_list[1])", legend=:topright)
plot!(MainCList[2], marker=:star, markersize=8, label="Epochs=$(epoch_size_list[2])", legend=:topright)
plot!(MainCList[3], marker=:diamond, markersize=6, label="Epochs=$(epoch_size_list[3])", legend=:topright)
plot!(MainCList[4], marker=:hexagon, markersize=4, label="Epochs=$(epoch_size_list[4])", legend=:topright)
plot!(MainCList[5], marker=:cross, markersize=2, label="Epochs=$(epoch_size_list[5])", legend=:topright)

savefig("figures/Training_Curve_Resistor_Network.pdf")

# Plane Fitting

using Revise, Plots, Random
include("Scripts.jl")

# --- Ground-truth plane ---
a_true = 0.7
b_true = -0.2
c_true = 1.5

# --- Dataset: (x,y) -> z ---
xs = collect(range(-5.0, 5.0, length=10))
ys = collect(range(-5.0, 5.0, length=10))

Z = a_true.*xs .+ b_true.*ys .+ c_true
 
# --- Network definition ---
# Nodes:
# 1: ground (0V)   fixed
# 3: bias (5V)     fixed
# 4: x-input       fixed
# 5: y-input       fixed
# 2: output        free
# 6,7: hidden      free (optional but recommended)

free_nodes  = [2, 6, 7]
fixed_nodes = [1, 3, 4, 5]
target_nodes  = [2]  # clamp output node

number_of_free_nodes = length(free_nodes)
If = zeros(number_of_free_nodes)

# branches: (i, j, g)
# start with some random-ish conductances (positive)
branches = [
    (2, 6, 2.0),
    (2, 7, 2.0),
    (6, 7, 1.0),

    (6, 4, 1.0),  # connect hidden to x-input
    (7, 5, 1.0),  # connect hidden to y-input

    (6, 3, 1.0),  # connect to bias
    (7, 3, 1.0),

    (6, 1, 1.0),  # connect to ground
    (7, 1, 1.0),
    (2, 1, 1.0),
]

epoch_size = 20000
α  = 1e-4
η  = 1e-3

CList = Float64[]

for epoch in 1:epoch_size
    perm = randperm(length(Z))
    C_epoch = 0.0

    for k in perm
        x = X[k]
        y = Y[k]
        z = Z[k]

        # fixed_nodes order: [1,3,4,5] = [gnd, bias, x, y]
        Vc = [0.0, 5.0, x, y]
        target_values = [z]

        C, P_hist, gnew = train_step!(
            branches, free_nodes, fixed_nodes, Vc, If,
            target_nodes, target_values; α, η
        )

        C_epoch += C
    end

    push!(CList, C_epoch/length(Z))
end

plot(CList, xlabel="Epoch", ylabel="MSE (Cost)", title="Plane Regression Training", legend=false)

# Hyper-plane fitting 

using Revise, Plots, Random, Triangulate
include("Scripts.jl")

Vbias = 5.0
Vgnd  = 0.0

d = 5
N = 200
w_true = rand(d)
w_true ./= sum(w_true)
b_true = 0.3

X = rand(0:0.0001:1, N, d)
Z = X*w_true .+ b_true

fixed_voltages = [maximum([vcat(X...); Vbias; Vgnd]), minimum([vcat(X...); Vbias; Vgnd])]

try
    @assert findall(x->x==1, Z.<fixed_voltages[2]) == Int[]
    @assert findall(x->x==1, Z.>fixed_voltages[1]) == Int[]
catch e
    error("Target values Z must be within the range of fixed voltages.")
end

num_of_inputs = d + 1 + 1 # pixels + bias + ground
area_max = 0.01

P_network, x, y, mesh = generate_triangulate_mesh(area_max)

number_of_nodes = size(mesh.pointlist, 2)

if number_of_nodes < num_of_inputs + 1 #  +1 for output node
    error("Number of nodes in the mesh is less than number of input dimensions + 1.")
end

scatter!(P_network, x, y, label="Hidden Nodes", markersize=4, color = :gray, legend = :outertopleft)

# Randomly select input and output nodes
branches, feature_nodes, bias_node, ground_node, output_node, all_nodes = extract_branches_from_mesh(mesh, x, num_of_inputs)
scatter!(P_network, x[feature_nodes], y[feature_nodes], color = :green, markersize=4, label="Input Nodes")
scatter!(P_network, [x[output_node]], [y[output_node]], color = :blue, markersize=4, label="Output Node")
scatter!(P_network, [x[bias_node]], [y[bias_node]], color = :black, markersize=4, label="Bias Node")
scatter!(P_network, [x[ground_node]], [y[ground_node]], color = :red, markersize=4, label="Ground Node")

fixed_nodes = vcat(feature_nodes, [bias_node, ground_node])
free_nodes  = setdiff(all_nodes, fixed_nodes)

number_of_free_nodes = length(free_nodes)
If = zeros(number_of_free_nodes)

epoch_size = 1000
α  = 1e-4
η  = 1e-3

CList = Float64[]

# Some sanity checks
try
    @assert output_node in free_nodes
    @assert output_node ∉ feature_nodes
    @assert bias_node   in fixed_nodes
    @assert ground_node in fixed_nodes
    @assert ground_node != bias_node
    @assert all(n -> n in fixed_nodes, feature_nodes)
    @assert length(feature_nodes) == d
    @assert output_node != bias_node != ground_node
    @assert sort(free_nodes ∪ fixed_nodes) == all_nodes
    @assert intersect(free_nodes, fixed_nodes) == Int[]
catch e
    error("Node selection sanity check failed. Please check the node assignments.")
end

for epoch in 1:epoch_size
    perm = randperm(N)
    C_epoch = 0.0

    for k in perm
        xfeat = X[k, :]                 # length 20
        y     = Z[k]                    # scalar target

        Vc = vcat(xfeat, [Vbias, Vgnd]) # fixed_nodes order must match!
        C, yhat = train_step!(
            branches, free_nodes, fixed_nodes, Vc, If,
            output_node, y; α=α, η=η
        )

        C_epoch += C
    end

    push!(CList, C_epoch/N) # MSE
end

plot(CList, xlabel="Epoch", ylabel="MSE", legend=false)