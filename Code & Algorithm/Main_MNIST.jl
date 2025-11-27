########## Main_MNIST.jl ##########
include("Scripts.jl")
include("MNISTUtils.jl")
using Revise, TriangleMesh, Random
using Plots, LaTeXStrings, Graphs, GraphMakie
using Statistics

###################
# MNIST + PCA DATA #
###################

n_components        = 15   # PCA boyutu = input node sayısı
n_train_per_class   = 200  # toplam N_train = 400
n_test_per_class    = 100  # toplam N_test = 200

X_train, y_train_lbl, X_test, y_test_lbl =
    mnist01_pca_data(n_components, n_train_per_class, n_test_per_class)

N_train = size(X_train, 2)
N_test  = size(X_test,  2)

println("Train samples: $N_train, Test samples: $N_test, Input dim (PCA): $n_components")

##############################
# TRIANGULAR MESH ve NETWORK #
##############################

area_max = 0.005
Pplot, x, y, cells, mesh = generate_uniform_mesh(area_max)
number_of_nodes = size(mesh.point_attribute, 2)

num_of_inputs = n_components
if number_of_nodes < num_of_inputs + 3   # 2 output + 1-2 buffer
    error("Mesh node sayısı, input sayısı + birkaç output için yetersiz. area_max veya mesh'i büyüt.")
end

scatter!(Pplot, x, y, label="Nodes", markersize=3, color=:gray)

# --- branches ---
branches = []
for e in eachcol(mesh.edge)
    i, j = e
    g = rand() * 5.0
    push!(branches, (i, j, g))
end

# --- node seçimleri ---
all_nodes = collect(1:length(x))

# Output için 2 node seç (sabit: 0 sınıfı için, 1 sınıfı için)
output_node0 = randomly_pick_nodes(all_nodes)
output_node1 = randomly_pick_nodes(setdiff(all_nodes, [output_node0]))
target_nodes = [output_node0, output_node1]

# Komşuluk dict'i (input node'ları non-adjacent seçmek için)
neighbors = Dict(n => Set{Int}() for n in all_nodes)
for (i, j, _) in branches
    push!(neighbors[i], j)
    push!(neighbors[j], i)
end

input_nodes = pick_nonadjacent_inputs(all_nodes, neighbors, target_nodes, num_of_inputs)

# input–input bağlantılarını filtrele
filtered_branches = []
for (i, j, g) in branches
    if (i in input_nodes) && (j in input_nodes)
        continue
    end
    push!(filtered_branches, (i, j, g))
end
branches = filtered_branches

scatter!(Pplot, x[input_nodes], y[input_nodes], color=:green,  markersize=6, label="Input Nodes")
scatter!(Pplot, [x[output_node0]], [y[output_node0]], color=:blue,  markersize=7, label="Out: class 0")
scatter!(Pplot, [x[output_node1]], [y[output_node1]], color=:red,   markersize=7, label="Out: class 1")

# free ve fixed node setleri
free_nodes  = setdiff(all_nodes, input_nodes)
fixed_nodes = input_nodes

If = zeros(length(free_nodes))   # free node KCL

########################
# TRAINING HYPERPARAMS #
########################

iteration_size = 50_000
α = 5e-4
η = 5e-3  # α/(2η) ~ 0.05 civarı olsun diye
C_list = Float64[]
gnew_list = Vector{Vector{Float64}}()

######################
# TRAINING DÖNGÜSÜ   #
######################

for step in 1:iteration_size
    # Rastgele bir train örneği seç
    k = rand(1:N_train)

    xk = X_train[:, k]      # boyut: n_components
    label = y_train_lbl[k]  # 0 veya 1

    # input voltajları = bu PCA vektörü
    Vc = copy(xk)  # fixed_nodes sırasıyla aynı boyutta

    # Target voltajları: [V_out0, V_out1]
    if label == 0
        target_values = [1.0, 0.0]
    else
        target_values = [0.0, 1.0]
    end

    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes,
                                  Vc, If, target_nodes, target_values; α=α, η=η)

    push!(C_list, C)
    push!(gnew_list, gnew)

    if step % 100 == 0
        println("Step $step / $iteration_size, Cost = $(C)")
    end
end

##########################
# COST SMOOTH GÖRSELLEŞE #
##########################

function moving_average(x, window)
    n = length(x)
    y = similar(x)
    for i in 1:n
        i1 = max(1, i - window + 1)
        y[i] = mean(x[i1:i])
    end
    return y
end

smooth_cost = moving_average(C_list, 100)

pC = plot(C_list, xlabel="Iteration", ylabel="Cost",
          title="Training cost (MNIST 0 vs 1)", label="Instant cost", alpha=0.3)
plot!(pC, smooth_cost, label="Moving avg", lw=2)

display(pC)

##########################
# TEST & ACCURACY        #
##########################

function predict_label(branches, free_nodes, fixed_nodes, If, target_nodes, x_vec)
    Vc = copy(x_vec)
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)
    V  = full_voltage(free_nodes, fixed_nodes, Vf, Vc)
    v0 = V[target_nodes[1]]
    v1 = V[target_nodes[2]]
    return v0 >= v1 ? 0 : 1
end

# Train accuracy
train_preds = [predict_label(branches, free_nodes, fixed_nodes, If, target_nodes, X_train[:,k])
               for k in 1:N_train]
train_acc = mean(train_preds .== y_train_lbl)
println("Train accuracy = ", train_acc)

# Test accuracy
test_preds = [predict_label(branches, free_nodes, fixed_nodes, If, target_nodes, X_test[:,k])
              for k in 1:N_test]
test_acc = mean(test_preds .== y_test_lbl)
println("Test accuracy = ", test_acc)
