using Revise
include("Scripts.jl")

number_of_free_nodes = 1

# 1. Free Phase
branches = [
    (1, 2, 3.0),
    (2, 3, 2.0)
]
free_nodes  = [2]
fixed_nodes = [1, 3]
Gff, Gfc = build_blocks(branches, free_nodes, fixed_nodes)

Vc = [2.0, 3.0]
If = zeros(number_of_free_nodes)
Vf = solve_free(Gff, Gfc, Vc, If)

Full_Voltage = full_voltage(free_nodes, fixed_nodes, Vf, Vc)

# 2. Clamped Phase

η = 1e-1
target_nodes = [2] # nodes to be clamped
target_values = [7] # target voltages
VtC   = [Full_Voltage[n] + η*(pt - Full_Voltage[n]) for (n,pt) in zip(target_nodes, target_values)]

free2  = [n for n in free_nodes if !(n in target_nodes)] # remaining free nodes after clamping
fixed2 = vcat(fixed_nodes, target_nodes) # set target nodes as temporary fixed nodes (add targets to fixed nodes)
Vc2    = vcat(Vc, VtC) # add fixed node voltages

Gff2, Gfc2 = build_blocks(branches, free2, fixed2) 
If2 = zeros(length(free2))
Vf2 = solve_free(Gff2, Gfc2, Vc2, If2)

Vclamped = full_voltage(free2, fixed2, Vf2, Vc2)

ΔVF = branch_dV(branches, Full_Voltage)
ΔVC = branch_dV(branches, Vclamped)

α = 1e-1
kmin = 1e-6

# since update is iterative, we need to run every time by exclamation mark (!) indicating in-place modification
# exclamation mark (!): Modifies original in-place, more memory efficient, but changes the input
for n in eachindex(branches) 
    i, j, g = branches[n] 
    Δg = (α/(2η)) * ((ΔVF[n]^2) - (ΔVC[n]^2)) # Equation (4)
    gnew = max(g + Δg, kmin) # gnew can't be negative
    branches[n] = (i, j, gnew)
end

branches
sum([(Full_Voltage[n] - pt)^2 for (n, pt) in zip(target_nodes, target_values)])
C = 0.5 * sum((Full_Voltage[n] - pt)^2 for (n, pt) in zip(target_nodes, target_values))