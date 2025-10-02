using LinearAlgebra

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
    free_pos = Dict(free_nodes[i] => i for i in 1:nfree)
    fixed_pos = Dict(fixed_nodes[i] => i for i in 1:nfixed)

    Gff = zeros(Float64, nfree, nfree)
    Gfc = zeros(Float64, nfree, nfixed)

    for (i, j, g) in branches
        # Case 1: both free
        if haskey(free_pos, i) && haskey(free_pos, j)
            pi, pj = free_pos[i], free_pos[j]
            Gff[pi, pi] += g
            Gff[pj, pj] += g
            Gff[pi, pj] -= g
            Gff[pj, pi] -= g
        # Case 2: free–fixed
        elseif haskey(free_pos, i) && haskey(fixed_pos, j)
            pi, pj = free_pos[i], fixed_pos[j]
            Gff[pi, pi] += g
            Gfc[pi, pj] -= g
        elseif haskey(free_pos, j) && haskey(fixed_pos, i)
            pj, pi = free_pos[j], fixed_pos[i]
            Gff[pj, pj] += g
            Gfc[pj, pi] -= g
        end
    end

    return Gff, Gfc
end

"""
    solve_free(Gff, Gfc, Vc, If)

Solve for free node voltages:
    Vf = Gff / (If - Gfc * Vc)
"""
function solve_free(Gff, Gfc, Vc, If)
    return Gff \ (If - Gfc * Vc)
end