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

using Revise, Plots
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

plot(CList , xlabel="Epoch", ylabel="Cost", title="Training Curve", legend=false)


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

plot(MainCList[1] , xlabel="Epoch", ylabel="Cost", title="Training Curve", 
marker=:o, markersize=10, label="Epochs=$(epoch_size_list[1])", legend=:topright)
plot!(MainCList[2], marker=:star, markersize=8, label="Epochs=$(epoch_size_list[2])", legend=:topright)
plot!(MainCList[3], marker=:diamond, markersize=6, label="Epochs=$(epoch_size_list[3])", legend=:topright)
plot!(MainCList[4], marker=:hexagon, markersize=4, label="Epochs=$(epoch_size_list[4])", legend=:topright)
plot!(MainCList[5], marker=:cross, markersize=2, label="Epochs=$(epoch_size_list[5])", legend=:topright)