includet("Scripts.jl")
using Revise
using Plots, LaTeXStrings, Graphs, GraphMakie

########
# y=ax #
########
number_of_free_nodes = 1

# 1. Free Phase 
branches = [
    (1, 2, 2.0),
    (2, 3, 3.0)
]

free_nodes  = [2]
fixed_nodes = [1, 3]
Vc = [2.0, 3.0] # fixed node voltages (boundary conditions)
If = zeros(number_of_free_nodes) # KCL rule at free nodes

target_nodes = [2] # nodes to be clamped
target_values = [2.5] # target voltages

##########
# y=ax+b #
##########

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

target_nodes = [2] # nodes to be clamped
target_values = [2.5] # target voltages

# Training loop
CostList = Float64[]
g_hist = []
p_hist = []
gnew = [0,0]
for step in 1:1000

    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α=5*10e-3, η=1e-2)    
    if findall(x->x<0, gnew) != []
        @warn "Conductance went negative!"
        #break
    end

    push!(CostList, C)
    push!(g_hist, [b[3] for b in branches])
    push!(p_hist, P_hist)
end

summ = gnew[1]+gnew[2]
Vc[1]*gnew[1]/summ+Vc[2]*gnew[2]/summ

plot(1:length(CostList), CostList, xlabel="Training step", ylabel="Cost", title="Training Resistor Network", label=L"C=\frac{1}{2}\sum_T (P_T^F - P_T)^2")
savefig("Code & Algorithm/figures/Training_Resistor_Network_Cost.pdf")

Gmat = hcat(g_hist...)'  # epoch × edge
plot(Gmat, xlabel="Iteration", ylabel=L"g_{ij}", legend=false, lw=1.5)
savefig("Code & Algorithm/figures/Training_Resistor_Network_Conductance.pdf")

plot(hcat(p_hist...)', xlabel="Iteration", ylabel="Power Dissipation", lw=1.5, label=L"P=\frac{1}{2}\sum_{ij} g_{ij}(\Delta V_{ij})^2")
savefig("Code & Algorithm/figures/Training_Resistor_Network_Power.pdf")