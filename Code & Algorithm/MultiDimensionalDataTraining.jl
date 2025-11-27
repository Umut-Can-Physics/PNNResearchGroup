include("Scripts.jl")
using Revise, Statistics
using Plots, LaTeXStrings, Graphs, GraphMakie

training_size = 100

##################################
##### DATA GENERATION (y=ax) #####
##################################
#= number_of_free_nodes = 1
bmax_area = max_area
# Noise
σ = 0.3 # Standart deviation of noise
noise_vector = randn(training_size) .* σ 
std(noise_vector)
noise_vector = noise_vector .- mean(noise_vector)
mean(noise_vector) # Noise with zero mean

x = range(0.1, stop=10, length=training_size)
y = sort(0.8 .* x) .+ noise_vector

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

bias = 0.3

number_of_free_nodes = 3

# Data Generation with noise Configuration
σ = 0.4 # Standart deviation of noise
noise_vector = randn(training_size) .* σ 
std(noise_vector)
noise_vector = noise_vector .- mean(noise_vector)
mean(noise_vector) # Noise with zero mean
x = range(start = 0, stop = 20, length = training_size)
y = sort(0.5 .* x) .+ noise_vector .+ bias

branches = [
    (2, 1, 4.0),
    (2, 3, 7.0),
    (2, 4, 1.0),
    (2, 5, 6.0),
    (1, 5, 2.0),
    (1, 4, 4.0),
    (3, 5, 3.0),
    (3, 4, 5.0)
]
free_nodes = [2, 3, 4]
fixed_nodes = [1, 5] 

bias_voltage = 1.0 # V4 = bias voltage

If = zeros(number_of_free_nodes) # KCL rule at free nodes

target_nodes = [2] # nodes to be clamped (only one node here, meaning target nodes is clamped to one voltage)

gnew_list = []
P_list = []
iteration_size = 500
α=1e-2 # learning rate
η=1e-2 # clamping strength
for step in 1:iteration_size
    rand_position = rand(eachindex(x))
    target_value = [y[rand_position]] # target (output) voltages
    input_value = x[rand_position] # input voltages
    Vc = [input_value, bias_voltage] # [V1 (input), V3(grounded), V4(bias)]
    # SORU: Her adımda bir önceki gnew'i kullanıyor muyuz? EVET!
    C, P_hist, gnew = train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_value; α, η) 
    push!(gnew_list, gnew)
    push!(P_list, P_hist)
end

plot(map(P_list->P_list[end], P_list))

# plot all conductances
ylabel = latexstring("Conductance  \$ [ \\Omega^{-1} ] \$")
p1 = plot(title="Training Size=$(training_size)", xlabel="Iteration", ylabel=ylabel, legend=false)
for i in 1:length(branches)
    plot!(map(g->g[i], gnew_list))
end
p1

pred_y = []
for xi in x
    Vc = [xi, bias_voltage]
    solved = solve_free(branches, free_nodes, fixed_nodes, Vc, If) # it use final conductances after training
    push!(pred_y, solved[1]) # output voltage at node 2
end

plot_style = Dict(
    :title => latexstring("Learning Rate: \$ \\alpha = $(α) \$ , Noise Std Dev: \$ \\sigma = $(σ) \$\n Clamping strength: \$ \\eta = $(η) \$"),
    :xlabel => L"x",
    :ylabel => L"y",
    :legendfontsize => 8,
    :linewidth => 5,
    :markersize => 4
)

p2 = scatter(x, y, label=latexstring("\$ y = ax + b + \\xi(\\sigma) \$"); plot_style...)
plot!(x, pred_y, label="Predicted"; plot_style...)
savefig(p2, "Code & Algorithm/figures/Trained Data Linear with Bias.pdf")

#p_main = plot(p1, p2, layout=(1,2), size=(900,400))
#savefig(p_main, "Code & Algorithm/figures/Conductances and Trained Data.pdf")