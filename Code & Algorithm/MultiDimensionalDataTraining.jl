include("Scripts.jl")
using Revise, Statistics
using Plots, LaTeXStrings, Graphs, GraphMakie

number_of_free_nodes = 1
training_size = 100

##################################
##### DATA GENERATION (y=ax) #####
##################################

#= x = range(0.1, stop=10, length=training_size)
y = sort(0.3 .* x) 

# 1. Free Phase
branches = [
    (1, 2, 7.0),
    (2, 3, 9.0)
]

free_nodes  = [2]
fixed_nodes = [1, 3] =#

####################################
##### DATA GENERATION (y=ax+b) #####
####################################

# Noise 
σ = 2.5 # Standart deviation of noise
noise_vector = randn(training_size) .* σ 
std(noise_vector)
noise_vector = noise_vector .- mean(noise_vector)
mean(noise_vector) # Noise with zero mean

x = range(start = 0, stop = 20, length = training_size)
y = sort(0.3 .* x) .+ 9 .+ noise_vector

branches = [
    (1, 2, 4.0),
    (2, 3, 7.0),
    (3, 4, 3.0),
    (1, 4, 10.0),
    (3, 4, 5.0)
]
free_nodes  = [2]
fixed_nodes = [1, 3, 4] 

If = zeros(number_of_free_nodes) # KCL rule at free nodes

target_nodes = [2] # nodes to be clamped (only one node here, meaning target nodes is clamped to one voltage)

gnew_list = []
P_list = []
iteration_size = 5000
for step in 1:iteration_size
    rand_position = rand(eachindex(x))
    target_values = [y[rand_position]] # target (output) voltages
    
    Vc = [x[rand_position], 0.0, 7.0] # [V1, V3, V4]
    # Vc = [x[rand_position], 0.0] # [V1, V3]
    # SORU: Her adımda bir önceki gnew'i kullanıyor muyuz? EVET!
    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α=1e-1, η=1e-5) 
    push!(gnew_list, gnew)
    push!(P_list, P_hist)
end

ylabel = latexstring("Conductance  \$ [ \\Omega^{-1} ] \$")
p1 = plot(map(g->g[1], gnew_list), label=L"g_{12}", title="Training Size=$(training_size)", xlabel="Iteration", ylabel=ylabel)
plot!(map(g->g[2], gnew_list), label=L"g_{23}")

# Learned conductances
g12, g23 = branches[1][3], branches[2][3]
pred_y = [xi * g12 / (g12 + g23) for xi in x]
println("Slope learned: ", g12 / (g12 + g23))

title = latexstring("Learning Rate: \$ \\alpha = 10^{-5} \$, Noise Std Dev: \$ \\sigma = 1 \$")
p2 = plot(x, pred_y, label="Predicted", xlabel=L"x", ylabel=L"y", title=title)
plot!(x, y, label=latexstring("\$ y = ax + b + \\xi(\\sigma) \$"), lw=2)

p3 = plot(map(p->p[1], P_list), label="Power Dissipation", title="Power Dissipation over Iterations", xlabel="Iteration", ylabel="Power")

plot(p1, p2, p3, layout=(3,1), size=(800,800))