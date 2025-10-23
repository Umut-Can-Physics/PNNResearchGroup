using LinearAlgebra
using Revise

# FREE PHASE SOLUTION

"""
    build_blocks(nnodes, branches, free_nodes, fixed_nodes)

Return (Gff, Gfc) block matrices for nodal analysis.

- `branches`: list of tuples (i, j, g)
- `free_nodes`: vector of free node indices
- `fixed_nodes`: vector of fixed node indices
"""
function build_blocks(branches, free_nodes, fixed_nodes)

    #@warn "Note that connection between fixed nodes are ignored!"

    nfree = length(free_nodes)
    nfixed = length(fixed_nodes) 

    # Map node index -> position in block
    free_pos = Dict(free_nodes[i] => i for i in 1:nfree) # row positions of nodes (free) in ascending order
    fixed_pos = Dict(fixed_nodes[i] => i for i in 1:nfixed) # column positions of nodes (fixed) in ascending order

    # Block matrix (G) initialization
    Gff = zeros(Float64, nfree, nfree)
    Gfc = zeros(Float64, nfree, nfixed)

    for (i, j, g) in branches
        # Case 1: check both nodes are free
        if haskey(free_pos, i) && haskey(free_pos, j)
            pi, pj = free_pos[i], free_pos[j]
            Gff[pi, pi] += g # diagonal entry of Gff sub-block matrix 
            Gff[pj, pj] += g  
            Gff[pi, pj] = -g # off-diagonal
            Gff[pj, pi] = -g 
        # Case 2: check free–fixed connections
        elseif haskey(free_pos, i) && haskey(fixed_pos, j) # free i connected fixed j by pipe
            pi, pj = free_pos[i], fixed_pos[j] # position in the G matrix
            Gff[pi, pi] += g
            Gfc[pi, pj] = -g
        elseif haskey(free_pos, j) && haskey(fixed_pos, i) # free j connected fixed i by pipe
            pj, pi = free_pos[j], fixed_pos[i] # position in the G matrix
            Gff[pj, pj] += g 
            Gfc[pj, pi] = -g 
        # Case 3: both nodes are fixed (IGNORED)
        end
    end

    return Gff, Gfc
end

"""
    solve_free(Gff, Gfc, Vc, If)

Solve for free node voltages: Vf = Gff / (If - Gfc * Vc)
- `Gff`: block matrix for free-free nodes
- `Gfc`: block matrix for free-fixed nodes
- `Vc`: vector of fixed node voltages
- `If`: vector of total current at free nodes
"""
function solve_free(Gff, Gfc, Vc, If)
    return Gff \ (If - Gfc * Vc) # Gff^-1 * (If - Gfc * Vc)
end

# LOCAL LEARNING ALGORITHM

"""
    full_voltage(free_nodes, fixed_nodes, Vf, Vc)
Combine free and fixed node voltages into a dictionary mapping node index to voltage.

Returns a Dict{Int,Float64}

- `free_nodes`: vector of free node indices
- `fixed_nodes`: vector of fixed node indices
- `Vf`: vector of free node voltages
- `Vc`: vector of fixed node voltages
"""
function full_voltage(free_nodes, fixed_nodes, Vf, Vc)
    V = Dict{Int,Float64}()
    for (n,v) in zip(free_nodes, Vf);  V[n] = v; end
    for (n,v) in zip(fixed_nodes, Vc); V[n] = v; end
    return V
end

"""
    branch_dV(branches, V)
Compute voltage differences across branches.

Returns a vector of voltage differences corresponding to each branch in `branches`.

- `branches`: list of tuples (i, j, g)
- `V`: Dict{Int,Float64} mapping node index to voltage
"""
branch_dV(branches, V) = [V[i] - V[j] for (i,j,_) in branches] # Local voltage differences across branches (Main idea of the algorithm)

"""
    solve_clamped(branches, free_nodes, fixed_nodes, Vc, Vf, target_nodes, target_values, η)
Perform the clamped phase solution.

- `branches`: list of tuples (i, j, g)
- `free_nodes`: vector of free node indices
- `fixed_nodes`: vector of fixed node indices
- `Vc`: vector of fixed node voltages
- `Vf`: vector of free node voltages from the free phase
- `target_nodes`: vector of target node indices to be clamped
- `target_values`: vector of target voltages for the target nodes
- `η`: clamping strength
"""
function solve_clamped(branches, free_nodes, fixed_nodes, Vc, Vf, target_nodes, target_values, η)
    Vfree = full_voltage(free_nodes, fixed_nodes, Vf, Vc) # Voltages in free phase
    VtC   = [Vfree[n] + η*(pt - Vfree[n]) for (n,pt) in zip(target_nodes, target_values)] # Equation (3)
    for (n,pt) in zip(target_nodes, target_values)
        #println("Target node $n: free voltage = $(Vfree[n]), target voltage = $pt, clamped voltage = ", Vfree[n] ,"+", η ,"*", (pt ,"-", Vfree[n]),"=", Vfree[n] + η*(pt - Vfree[n]))
    end
  
    free2  = [n for n in free_nodes if !(n in target_nodes)] # remaining free nodes after clamping
    fixed2 = vcat(fixed_nodes, target_nodes) # set target nodes as temporary fixed nodes (add targets to fixed nodes)
    Vc2    = vcat(Vc, VtC) # add fixed node voltages

    # solve for new free node voltages
    Gff2, Gfc2 = build_blocks(branches, free2, fixed2) # build new block matrices
    If2 = zeros(length(free2))
    Vf2 = solve_free(Gff2, Gfc2, Vc2, If2)

    return full_voltage(free2, fixed2, Vf2, Vc2) # solved voltages in clamped phase
end
"""
    update_conductances!(branches, ΔVF, ΔVC; α=5e-4, η=1e-3, kmin=1e-6)
Update branch conductances in-place based on voltage differences.
- `branches`: list of tuples (i, j, g)
- `ΔVF`: vector of voltage differences in free phase
- `ΔVC`: vector of voltage differences in clamped phase
- `α`: learning rate (default 5e-4)
- `η`: clamping strength (default 1e-3)
- `kmin`: minimum conductance value to prevent negative conductances (default 1e-6)
"""
function update_conductances!(branches, ΔVF, ΔVC; α=5e-4, η=1e-3, kmin=1e-6)
    for n in eachindex(branches) 
        i, j, g = branches[n] 
        Δg = (α/(2η)) * ((ΔVF[n]^2) - (ΔVC[n]^2)) # Equation (4)
        gnew = g + Δg # gnew can't be negative
        branches[n] = (i, j, gnew)
    end
    #println("Updated conductances: ", branches)
end

"""
    train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α=5e-4, η=1e-3)
Perform a single training step consisting of free and clamped phases, conductance update, and cost calculation.
- `branches`: list of tuples (i, j, g)
- `free_nodes`: vector of free node indices
- `fixed_nodes`: vector of fixed node indices
- `Vc`: vector of fixed node voltages
- `If`: vector of total current at free nodes
- `target_nodes`: vector of target node indices to be clamped
- `target_values`: vector of target voltages for the target nodes
- `α`: learning rate (default 5e-4)
- `η`: clamping strength (default 1e-3)
"""
function train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α=5e-4, η=1e-3)

    #println("START OF THE STEP \n")

    # Free phase solution
    Gff, Gfc = build_blocks(branches, free_nodes, fixed_nodes)
    Vf = solve_free(Gff, Gfc, Vc, If)
    Vfree = full_voltage(free_nodes, fixed_nodes, Vf, Vc)

    #println("Full voltages at free phase: ", Vfree)

    # Clamped phase solution
    Vclamped = solve_clamped(branches, free_nodes, fixed_nodes, Vc, Vf, target_nodes, target_values, η)

    println("Full voltages at clamped phase: ", Vclamped)

    #differences (approximately equals back-propagation)
    ΔVF = branch_dV(branches, Vfree) # voltage difference in free phase
    ΔVC = branch_dV(branches, Vclamped) # voltage difference in clamped phase

    #println("Voltage differences in free phase: ", ΔVF)
    #println("Voltage differences in clamped phase: ", ΔVC)

    # Update the conductances
    update_conductances!(branches, ΔVF, ΔVC; α=α, η=η)

    # Cost function
    C = 0.5 * sum((Vfree[n] - pt)^2 for (n, pt) in zip(target_nodes, target_values)) # Equation (1)
    #println("Voltages of the target nodes: ", [Vfree[n] for n in target_nodes], " | Cost: ", C)

    P_hist = [0.5 * sum(b[3]*(Δ^2) for (b,Δ) in zip(branches, ΔVF))] # Power dissipation 

    gnew = [b[3] for b in branches]

    #println("END OF THE STEP\n====================\n")

    return C, P_hist, gnew
end

# In Progress: Visualization with GraphMakie
function plot_network(branches; title="Conductance network")
    nodes = unique(vcat([b[1] for b in branches], [b[2] for b in branches]))
    G = SimpleGraph(length(nodes))
    for (i,j,_) in branches; add_edge!(G, i, j); end
    weights = [b[3] for b in branches]
    fig, ax, plt = graphplot(G, node_labels=1:nv(G), edge_width=3 .* normalize(weights))
    ax.title = title
    display(fig)
end