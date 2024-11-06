using CairoMakie

# Constants
const Nd = 2e16    
const μn0 = 1350    
const vsat = 1.8e7  
const q = 1.6e-19   


E = 10 .^ range(0, 6, length=1000)  # From 10^0 to 10^6 V/cm


vd = @. (μn0 * E) / sqrt(1 + (μn0 * E / vsat)^2)

# Calculate current density J = qnv
J = abs.(q * Nd * vd)

# Create the plot
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1],
    xscale=log10,
    yscale=log10,
    xlabel="Electric Field (V/cm)",
    ylabel="Current Density (A/cm²)",
    title="Electron Drift Current Density vs Electric Field")


lines!(ax, E, J, linewidth=2)


ax.xgridstyle = :dash
ax.ygridstyle = :dash


display(fig)

save("drift_current_density.png", fig)