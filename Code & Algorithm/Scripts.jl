using Revise, LinearAlgebra, Statistics

# FREE PHASE SOLUTION

"""
    build_blocks(nnodes, branches, free_nodes, fixed_nodes)

Return (Gff, Gfc) block matrices for nodal analysis.

- `branches`: list of tuples (i, j, g)
- `free_nodes`: vector of free node indices
- `fixed_nodes`: vector of fixed node indices
"""
function build_blocks(branches, free_nodes, fixed_nodes)

    nfree = length(free_nodes)
    nfixed = length(fixed_nodes) 

    # Map node index -> position in block
    free_pos = Dict(free_nodes[i] => i for i in 1:nfree) # row positions of nodes (free) in ascending order
    fixed_pos = Dict(fixed_nodes[i] => i for i in 1:nfixed) # column positions of nodes (fixed) in ascending order

    # Block matrix (G) initialization
    Gff = zeros(Float64, nfree, nfree)
    Gfc = zeros(Float64, nfree, nfixed)
    Gcc = zeros(Float64, nfixed, nfixed)

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
        # Case 3: both nodes are fixed
        elseif haskey(fixed_pos, i) && haskey(fixed_pos, j)
            pi, pj = fixed_pos[i], fixed_pos[j]
            Gcc[pi, pi] += g
            Gcc[pj, pj] += g
            Gcc[pi, pj] = -g
            Gcc[pj, pi] = -g
        end
    end

    return Gff, Gfc, free_pos
end

"""
    solve_free(Gff, Gfc, Vc, If)

Solve for free node voltages: Vf = Gff / (If - Gfc * Vc)
- `Gff`: block matrix for free-free nodes
- `Gfc`: block matrix for free-fixed nodes
- `Vc`: vector of fixed node voltages
- `If`: vector of total current at free nodes
"""
function solve_free(branches, free_nodes, fixed_nodes, Vc, If)
    Gff, Gfc, free_pos = build_blocks(branches, free_nodes, fixed_nodes) 
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
# Sadece free ve fixed node'ları alıyorum, çünkü hidden node'ların voltaj farkları zaten free node'ların içinde.
"""
    branch_dV(branches, V)
Compute voltage differences across branches.

Returns a vector of voltage differences corresponding to each branch in `branches`.

- `branches`: list of tuples (i, j, g)
- `V`: Dict{Int,Float64} mapping node index to voltage
"""
# Clean version of function branch_dV2
function branch_dV3(branches, V::AbstractDict)
    out = Vector{valtype(V)}(undef, length(branches))
    @inbounds for (k, (i, j, _)) in enumerate(branches)
        vi = get(V, i, nothing); vi === nothing && error("No voltage for node $i")
        vj = get(V, j, nothing); vj === nothing && error("No voltage for node $j")
        out[k] = vi - vj
    end
    out
end

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

    Vfree = full_voltage(free_nodes, fixed_nodes, Vf, Vc) # Get voltages in free phase

    VtC   = [Vfree[n] + η*(pt - Vfree[n]) for (n,pt) in zip(target_nodes, target_values)] # Equation (3)

    @debug "Clamped voltages for target nodes" only(VtC)
  
    free2  = [n for n in free_nodes if !(n in target_nodes)] # remaining free nodes after clamping
    fixed2 = vcat(fixed_nodes, target_nodes) # set target nodes as temporary fixed nodes (add targets to fixed nodes)
    Vc2    = vcat(Vc, VtC) # add fixed node voltages

    # solve for new free node voltages
    # build new block matrices
    If2 = zeros(length(free2))
    Vf2 = solve_free(branches, free2, fixed2, Vc2, If2)

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
        #gnew = g + Δg # gnew can't be negative
        gnew = max(kmin, g + Δg)   # <-- clamp
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
function train_step!(branches, free_nodes, fixed_nodes, Vc, If, target_nodes, target_values; α, η)

    # Free phase solution
    Vf = solve_free(branches, free_nodes, fixed_nodes, Vc, If)

    Vfree = full_voltage(free_nodes, fixed_nodes, Vf, Vc)

    v_out = Vfree
    #[only(target_nodes)]
    @debug "solved target voltage in free phase" v_out
    
    # Clamped phase solution
    # Solve clamped return full voltages in free phase
    Vclamped = solve_clamped(branches, free_nodes, fixed_nodes, Vc, Vf, target_nodes, target_values, η)

    v_out_clamped = Vclamped
    #[only(target_nodes)]
    @debug "Solved target voltage in clamped phase" v_out_clamped

    #differences (approximately equals back-propagation)
    ΔVF = branch_dV3(branches, Vfree) # voltage difference in free phase
    ΔVC = branch_dV3(branches, Vclamped) # voltage difference in clamped phase

    #println("Voltage differences in free phase: ", ΔVF)
    #println("Voltage differences in clamped phase: ", ΔVC) 

    # Update the conductances
    update_conductances!(branches, ΔVF, ΔVC; α, η)

    # Cost function
    C = 0.5 * sum((Vfree[n] - pt)^2 for (n, pt) in zip(target_nodes, target_values)) # Equation (1)
    #println("Voltages of the target nodes: ", [Vfree[n] for n in target_nodes], " | Cost: ", C)

    P_hist = [0.5 * sum(b[3]*(Δ^2) for (b,Δ) in zip(branches, ΔVF))] # Power dissipation 

    #println("END OF THE STEP\n====================\n")

    gnew = [b[3] for b in branches] 

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

"""
Generate synthetic training data with optional noise and bias.
- `slope`: Slope of the linear function
- `σ`: Dtandard deviation of Gaussian noise
- `bias`: Bias term added to the output 
- `training_size`: Number of training samples
"""
function generate_training_data(slope, σ, bias, training_size)
    noise_vector = randn(training_size) .* σ 
    noise_vector = noise_vector .- mean(noise_vector)
    x = range(start = 0, stop = 20, length = training_size)
    y = sort(slope .* x) .+ noise_vector .+ bias
    #println("Standart Deviation of Data = ",std(noise_vector),"\n","Mean of the Noise = ", mean(noise_vector))
    return x, y
end

function find_factors(n)
    factors = []
    for i in 1:n
        if n % i == 0
            push!(factors, i)
        end
    end
    return factors
end

"""
Generate tensor-based hyperplane regression data.
Each sample is a tensor X[:,:,k] of size (H, W).
Outputs follow:
    y[k] = sum(A .* X[:,:,k]) + noise

Arguments:
- H, W           : tensor dimensions
- training_size  : number of samples
- σ              : noise standard deviation

Returns:
- X  : H × W × training_size tensor
- y  : output vector (training_size)
- A  : true weight tensor (H × W)
"""
function generate_tensor_hyperplane_data(H, W, training_size, InputRange; σ=0.1, bias=0.0)

    # true underlying weight tensor (same for all samples)
    # choose weights such that summation is around order of 1 (normalizeation of the weights)
    A = abs.(randn(H, W))
    A ./= sum(A)
    # Each pixel has a coefficient

    # input tensor
    X = abs.(rand(InputRange, H, W, training_size)) .+ bias
    
    # outputs
    y = zeros(training_size)

    # noise
    noise = σ .* randn(training_size)
    noise = noise .- mean(noise)

    for k in 1:training_size
        y[k] = sum(A .* X[:,:,k])
    end

    y = y .+ noise

    return X, y, A, noise
end

"""
Generate a uniform triangular mesh using TriangleMesh.jl and plot it.
"""
function generate_uniform_mesh(area_max)
    # uniform mesh (https://mathworld.wolfram.com/VoronoiDiagram.html)
    poly = polygon_Lshape()
    mesh = create_mesh(poly, info_str="triangular mesh", voronoi=true, delaunay=true, add_switches = "qa$(area_max)")

    pts   = mesh.point      # 2 × n_point
    cells = mesh.cell       # 3 × n_cell  (indices of vertices)

    # Coordinates (note: rows, not columns)
    x = pts[1, :]           # all x's
    y = pts[2, :]           # all y's

    P = plot(legend=false, aspect_ratio = :equal)

    # iterate over columns of `cells`, each is a triangle (i, j, k)
    for tri in eachcol(cells)
        i, j, k = tri
        plot!([x[i], x[j], x[k], x[i]],
            [y[i], y[j], y[k], y[i]],
            color = :gray)
    end

    # label the points according to their coordinates
    #for (idx, (xi, yi)) in enumerate(zip(x, y))
    #    annotate!(xi, yi, text("$(idx)", :left, 15))
    #end

    return P, x, y, cells, mesh
end

"""
Select a random node from a triangular mesh
- `all_nodes`: vector of all node indices
"""
randomly_pick_nodes(all_nodes) = all_nodes[rand(1:length(all_nodes))]

"""
Map a flattened input vector x_vec to fixed_nodes voltages.
- fixed_nodes: vector of node indices used as inputs
- x_vec      : vector of input values, same length as fixed_nodes
Returns Dict{Int,Float64} with only fixed node voltages.
"""
function input_to_voltage_dict(fixed_nodes, x_vec)
    @assert length(fixed_nodes) == length(x_vec)
    V = Dict{Int,Float64}()
    for (k, node) in enumerate(fixed_nodes)
        V[node] = x_vec[k]
    end
    return V
end

function pick_nonadjacent_inputs(all_nodes, neighbors, output_node, num_inputs)
    candidates = setdiff(all_nodes, [output_node])
    candidates = collect(candidates)
    shuffle!(candidates)

    input_nodes = Int[]

    for v in candidates
        # v, mevcut input'lardan hiçbiriyle komşu olmasın
        if all(u -> !(v in neighbors[u]), input_nodes)
            push!(input_nodes, v)
            length(input_nodes) == num_inputs && break
        end
    end

    if length(input_nodes) < num_inputs
        error("Input nodes selection failed: ")
    end

    return input_nodes
end


function generate_triangulate_mesh(area_max)
    # Define a simple bounding box (0,0) to (1,1) for the mesh
    # Triangulate requires points and segments to define the domain
    nodes = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]'
    segs = Int32[1 2; 2 3; 3 4; 4 1]'
    
    tin = TriangulateIO()
    tin.pointlist = nodes
    tin.segmentlist = segs
    
    # "p" for PSLG, "q" for quality, "a" for area constraint, "Q" for quiet
    switches = "pqea$(area_max)Q"
    (tout, vorout) = triangulate(switches, tin)
    
    # Extracting x, y for plotting
    x = tout.pointlist[1, :]
    y = tout.pointlist[2, :]
    
    # Create a base plot
    P_network = plot(aspect_ratio=:equal)
    
    return P_network, x, y, tout
end

function extract_branches_from_mesh(mesh, x, num_of_inputs; gmax=2.0)
    # num_of_inputs = d + 2  (features + bias + ground)

    # 1) Unique edges from triangles
    unique_edges = Set{Tuple{Int, Int}}()
    for col in 1:size(mesh.trianglelist, 2)
        t = mesh.trianglelist[:, col]
        for (a, b) in [(t[1], t[2]), (t[2], t[3]), (t[3], t[1])]
            push!(unique_edges, a < b ? (a, b) : (b, a))
        end
    end

    branches = Vector{Tuple{Int,Int,Float64}}()
    for (i, j) in unique_edges
        g = rand() * gmax
        push!(branches, (Int(i), Int(j), g))
    end

    all_nodes = collect(1:length(x))

    # 2) Neighbor map
    neighbors = Dict(n => Set{Int}() for n in all_nodes)
    for (i, j, _) in branches
        push!(neighbors[i], j)
        push!(neighbors[j], i)
    end

    # 3) Pick output node (keep as you had, but you may want smarter choice later)
    output_node = rand(all_nodes)

    # 4) Pick input nodes (size = num_of_inputs)
    input_nodes = pick_nonadjacent_inputs(all_nodes, neighbors, output_node, num_of_inputs)

    # 5) Filter out input-input edges (optional but you were doing it)
    filtered_branches = Tuple{Int,Int,Float64}[]
    for (i, j, g) in branches
        if (i in input_nodes) && (j in input_nodes)
            continue
        end
        push!(filtered_branches, (i, j, g))
    end
    branches = filtered_branches

    # 6) Split input_nodes into (feature_nodes, bias_node, ground_node)
    # degree for each node
    deg(n) = length(neighbors[n])

    # choose ground as smallest degree among input_nodes
    ground_node = argmin(n -> deg(n), input_nodes)

    # choose bias as largest degree among remaining
    remaining = setdiff(input_nodes, [ground_node])
    bias_node = argmax(n -> deg(n), remaining)

    feature_nodes = collect(setdiff(input_nodes, [ground_node, bias_node]))

    @info "Total nodes: $(length(all_nodes))"
    @info "Output node: $output_node"
    @info "Feature nodes: $(length(feature_nodes))"
    @info "Bias node: $bias_node"
    @info "Ground node: $ground_node"

    return branches, feature_nodes, bias_node, ground_node, output_node, all_nodes
end
