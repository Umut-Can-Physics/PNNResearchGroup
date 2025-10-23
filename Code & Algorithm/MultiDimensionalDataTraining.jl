include("Scripts.jl")
using Revise
using Plots, LaTeXStrings, Graphs, GraphMakie

# DATA GENERATION
# (y=ax)
x = 0.1:0.1:1000
y = sort(0.3 .* x) 

number_of_free_nodes = 1

# 1. Free Phase
branches = [
    (1, 2, 7.0),
    (2, 3, 9.0)
]

free_nodes  = [2]
fixed_nodes = [1, 3]

# (y=ax+b)

x = 0.1:0.1:1000
y = sort(0.3 .* x) .+ 10

branches = [
    (1, 2, 2.0),
    (2, 3, 3.0),
    (3, 4, 1.0)
]
vfree_nodes  = [2]
fixed_nodes = [1, 3, 4]

If = zeros(number_of_free_nodes) # KCL rule at free nodes

target_nodes = [2] # nodes to be clamped (only one node here, meaning target nodes is clamped to one voltage)

gnew_list = []
P_list = []
iteration_size = 500
for step in 1:iteration_size
    rand_position = rand(eachindex(x))
    Vc = [x[rand_position], 0.0, 1.0] # [V1, V3, V4]
    target_values = [y[rand_position]] # target voltages
    # SORU: Her adımda bir önceki gnew'i kullanıyor muyuz? EVET!
    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α=1e-5, η=1e-5) 
    push!(gnew_list, gnew)
    push!(P_list, P_hist)
end

p1 = plot(map(g->g[1], gnew_list), label=L"g_{12}", title="Conductance Evolution, Data Size=$(length(x))", xlabel="Iteration", ylabel="Conductance")
plot!(map(g->g[2], gnew_list), label=L"g_{23}")

# Learned conductances
g12, g23 = branches[1][3], branches[2][3]
pred_y = [xi * g12 / (g12 + g23) for xi in x]

p2 = plot(x, pred_y, seriestype=:scatter, label="Predicted", xlabel="x", ylabel="y", title="Resistor Network Predictions, Iteration Size=$(iteration_size)")
plot!(x, y, label="True", lw=2)

p3 = plot(map(p->p[1], P_list), label="Power Dissipation", title="Power Dissipation over Iterations", xlabel="Iteration", ylabel="Power")

plot(p1, p2, p3, layout=(3,1), size=(800,800))