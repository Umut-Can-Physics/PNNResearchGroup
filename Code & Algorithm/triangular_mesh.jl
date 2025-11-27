include("Scripts.jl")
using TriangleMesh
using Plots

# Note that: Hidden node has at least two branches connected to it, and output node has at least one branch connected to it.

################################
# MESH GENERATING AND PLOTTING #
################################

area_max = 0.01
p, x, y , cells, mesh = generate_uniform_mesh(area_max)
scatter(p, x, y, label="Nodes", markersize=1)