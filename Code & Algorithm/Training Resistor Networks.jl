include("Scripts.jl")
using Revise
using Plots, LaTeXStrings, Graphs, GraphMakie

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
Vc = [5.0, 0.0] # fixed node voltages (boundary conditions)
If = zeros(number_of_free_nodes) # KCL rule at free nodes

target_nodes = [3, 5] # nodes to be clamped
target_values = [3.0, 1.0] # target voltages

# Training loop
CostList = Float64[]
g_hist = []
p_hist = []
for step in 1:5000
    C, P_hist = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α=5e-4, η=1e-3)
    push!(CostList, C)
    push!(g_hist, [b[3] for b in branches])
    push!(p_hist, P_hist)
end

plot(1:length(CostList), CostList, xlabel="Training step", ylabel="Cost", title="Training Resistor Network", label=L"C=\frac{1}{2}\sum_T (P_T^F - P_T)^2")
savefig("Code & Algorithm/figures/Training_Resistor_Network_Cost.pdf")

Gmat = hcat(g_hist...)'  # epoch × edge
plot(Gmat, xlabel="Iteration", ylabel=L"g_{ij}", legend=false, lw=1.5)
savefig("Code & Algorithm/figures/Training_Resistor_Network_Conductance.pdf")

plot(hcat(p_hist...)', xlabel="Iteration", ylabel="Power Dissipation", lw=1.5, label=L"P=\frac{1}{2}\sum_{ij} g_{ij}(\Delta V_{ij})^2")
savefig("Code & Algorithm/figures/Training_Resistor_Network_Power.pdf")