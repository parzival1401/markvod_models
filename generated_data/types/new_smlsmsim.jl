using SMLMSim
using CairoMakie

# Set up a minimal simulation with just two particles
params = DiffusionSMLMParams(
    box_size = 1.0,       # 1 μm box for close interactions
    diff_monomer = 0.1,   # μm²/s
    diff_dimer = 0.05,    # μm²/s
    k_off = 0.5,          # s⁻¹ (moderate dimer stability)
    r_react = 0.05,       # μm (large reaction radius for demonstration)
    d_dimer = 0.07,       # μm (dimer separation)
    dt = 0.01,            # s
    t_max = 5.0,          # s
    boundary = "reflecting",  # Reflecting boundaries
    camera_framerate = 10.0   # fps
)

# Create two particles with specific initial positions
particle1 = DiffusingEmitter2D{Float64}(
    0.2, 0.2,       # Position in lower-left quadrant
    1000.0,         # Photons
    0.0,            # Initial timestamp
    1,              # Initial frame
    1,              # Dataset
    1,              # track_id
    :monomer,       # Initial state
    nothing         # No partner initially
)

particle2 = DiffusingEmitter2D{Float64}(
    0.8, 0.8,       # Position in upper-right quadrant
    1000.0,         # Photons
    0.0,            # Initial timestamp
    1,              # Initial frame
    1,              # Dataset
    2,              # track_id
    :monomer,       # Initial state
    nothing         # No partner initially
)

# Run simulation with custom starting positions
smld = simulate(params; starting_conditions=[particle1, particle2])

track_smlds = get_tracks(smld)

# Convert to the format needed for plotting
trajectories = []
for track_smld in track_smlds
    # Get ID from first emitter
    id = track_smld.emitters[1].track_id

    # Sort by timestamp
    sort!(track_smld.emitters, by = e -> e.timestamp)

    # Extract coordinates and state
    times = [e.timestamp for e in track_smld.emitters]
    x = [e.x for e in track_smld.emitters]
    y = [e.y for e in track_smld.emitters]
    states = [e.state for e in track_smld.emitters]

    push!(trajectories, (id=id, times=times, x=x, y=y, states=states))
end

# Visualize interaction dynamics
fig = Figure(size=(700, 600))

ax = Axis(fig[1, 1],
    title="Two Particles in 1μm Box (Reflecting Boundaries)",
    xlabel="x (μm)",
    ylabel="y (μm)",
    aspect=DataAspect()
)

# Plot trajectories with state-dependent coloring
for (i, traj) in enumerate(trajectories)
    # Create segments with colors based on state
    segments_x = []
    segments_y = []
    colors = []

    for j in 1:(length(traj.times)-1)
        push!(segments_x, [traj.x[j], traj.x[j+1]])
        push!(segments_y, [traj.y[j], traj.y[j+1]])
        push!(colors, traj.states[j] == :monomer ? :blue : :red)
    end

    # Plot each segment with appropriate color
    for j in 1:length(segments_x)
        lines!(ax, segments_x[j], segments_y[j],
               color=colors[j], linewidth=2,
               label=j==1 ? "Particle $(traj.id)" : nothing)
    end

    # Mark starting position
    scatter!(ax, [traj.x[1]], [traj.y[1]],
            color=:black, marker=:circle, markersize=10)

    # Mark ending position
    scatter!(ax, [traj.x[end]], [traj.y[end]],
            color=:black, marker=:star, markersize=12)
end

# Show box boundaries
box = [0 0; 1 0; 1 1; 0 1; 0 0]
lines!(ax, box[:, 1], box[:, 2], color=:black, linewidth=2)

# Add legend for state colors
legend_elements = [
    LineElement(color=:blue, linewidth=3),
    LineElement(color=:red, linewidth=3),
    MarkerElement(color=:black, marker=:circle, markersize=8),
    MarkerElement(color=:black, marker=:star, markersize=10)
]
legend_labels = ["Monomer", "Dimer", "Start", "End"]

Legend(fig[1, 2], legend_elements, legend_labels, "States")

# Set axis limits with some padding
limits!(ax, -0.05, 1.05, -0.05, 1.05)

fig

