using Random
using Pkg
Pkg.activate(".")

# Load the types and functions
include("types/types_with_data.jl")

println("=" ^ 60)
println("EXAMPLE DATA DEMONSTRATIONS")
println("=" ^ 60)


# Set parameters
n_steps = 100
dt = 0.0016
D = 0.01
σ = 0.005
k_states = [0.5, 0.5]  # [k_on, k_off] parameters
# ========================================
# EXAMPLE 2: CUSTOM TYPE APPROACH
# ========================================

println("\n🔹 EXAMPLE 2: Custom Type Approach")
println("-" ^ 40)

println("🔄 Converting matrix data to custom simulation type...")

# Convert matrices to custom types

sim = run_simulation(k_states[1],k_states[2],10; D=D, dt=dt, σ=σ)
sim_noisy = add_noise(sim, σ)

println("🔄 Generating animations...")

# Generate animation for clean simulation
clean_animation_file = animate_particles(sim, "clean_particle_animation.mp4")
println("✅ Clean simulation animation saved as '$(clean_animation_file)'")

# Generate animation for noisy simulation  
noisy_animation_file = animate_particles(sim_noisy, "noisy_particle_animation.mp4")
println("✅ Noisy simulation animation saved as '$(noisy_animation_file)'")


# ========================================
# EXAMPLE 3: FORWARD ALGORITHM WITH ORIGINAL PARAMETERS
# ========================================

println("\n🔹 EXAMPLE 3: Forward Algorithm with Original K Parameters")
println("-" ^ 40)

# Run forward algorithm with the original k_states used in simulation
println("🔄 Running forward algorithm with original k_states [$(k_states[1]), $(k_states[2])]...")

# Clean data forward algorithm (using larger dt for numerical stability)
alpha_clean, loglikelihood_clean = forward_algorithm(sim; dt=0.01, sigma=0.01)
println("   Clean data results:")
println("     • Log-likelihood: $(round(loglikelihood_clean, digits=2))")
println("     • Alpha matrix size: $(size(alpha_clean))")

# Noisy data forward algorithm  
alpha_noisy, loglikelihood_noisy = forward_algorithm(sim_noisy; dt=0.01, sigma=0.01)
println("   Noisy data results:")
println("     • Log-likelihood: $(round(loglikelihood_noisy, digits=2))")
println("     • Alpha matrix size: $(size(alpha_noisy))")

# Extract e1 and e2 density vectors from alpha matrices
e1_vec_original_clean = alpha_clean[1, :]  # Free state densities
e2_vec_original_clean = alpha_clean[2, :]  # Bound state densities

e1_vec_original_noisy = alpha_noisy[1, :]  # Free state densities
e2_vec_original_noisy = alpha_noisy[2, :]  # Bound state densities

println("   Density vector analysis:")
println("     • Clean e1 range: [$(round(minimum(e1_vec_original_clean), digits=6)), $(round(maximum(e1_vec_original_clean), digits=6))]")
println("     • Clean e2 range: [$(round(minimum(e2_vec_original_clean), digits=6)), $(round(maximum(e2_vec_original_clean), digits=6))]")
println("     • Noisy e1 range: [$(round(minimum(e1_vec_original_noisy), digits=6)), $(round(maximum(e1_vec_original_noisy), digits=6))]")
println("     • Noisy e2 range: [$(round(minimum(e2_vec_original_noisy), digits=6)), $(round(maximum(e2_vec_original_noisy), digits=6))]")

# ========================================
# EXAMPLE 4: PLOTTING STATE DENSITIES AND POSITIONS
# ========================================

println("\n🔹 EXAMPLE 4: Plotting State Densities and Position Analysis")
println("-" ^ 40)

using CairoMakie

# Get position matrices for analysis
p1_clean, p2_clean = get_position_matrices(sim)
p1_noisy, p2_noisy = get_position_matrices(sim_noisy)

# Calculate inter-particle distances
distances_clean = [sqrt((p1_clean[1,i] - p2_clean[1,i])^2 + (p1_clean[2,i] - p2_clean[2,i])^2) for i in 1:size(p1_clean,2)]
distances_noisy = [sqrt((p1_noisy[1,i] - p2_noisy[1,i])^2 + (p1_noisy[2,i] - p2_noisy[2,i])^2) for i in 1:size(p1_noisy,2)]

# ========================================
# CLEAN DATA FIGURE
# ========================================

# Create figure for clean data
fig_clean = Figure(size=(1000, 800))

# Plot 1: E1 and E2 vectors together for clean data
ax_clean_densities = Axis(fig_clean[1, 1], 
                         title="Clean Data: State Densities (k=[0.5, 0.5])", 
                         xlabel="Time Step", 
                         ylabel="Density Value")

lines!(ax_clean_densities, 1:length(e1_vec_original_clean), e1_vec_original_clean, 
       color=:blue, linewidth=2, label="e1 (Free State)")
lines!(ax_clean_densities, 1:length(e2_vec_original_clean), e2_vec_original_clean, 
       color=:red, linewidth=2, label="e2 (Bound State)")

axislegend(ax_clean_densities, position=:rt)

# Plot 2: Inter-particle distance for clean data
ax_clean_positions = Axis(fig_clean[2, 1], 
                         title="Clean Data: Inter-Particle Distance", 
                         xlabel="Time Step", 
                         ylabel="Distance")

lines!(ax_clean_positions, 1:length(distances_clean), distances_clean, 
       color=:green, linewidth=2, label="Distance")
hlines!(ax_clean_positions, [sim.d_dimer], color=:black, linestyle=:dash, linewidth=1, label="d_dimer = $(sim.d_dimer)")

axislegend(ax_clean_positions, position=:rt)

# Save clean data figure
save("clean_data_analysis.png", fig_clean)
println("✅ Clean data analysis plot saved as 'clean_data_analysis.png'")

# Display clean data figure
display(fig_clean)

# ========================================
# NOISY DATA FIGURE
# ========================================

# Create figure for noisy data
fig_noisy = Figure(size=(1000, 800))

# Plot 1: E1 and E2 vectors together for noisy data
ax_noisy_densities = Axis(fig_noisy[1, 1], 
                         title="Noisy Data: State Densities (k=[0.5, 0.5])", 
                         xlabel="Time Step", 
                         ylabel="Density Value")

lines!(ax_noisy_densities, 1:length(e1_vec_original_noisy), e1_vec_original_noisy, 
       color=:blue, linewidth=2, label="e1 (Free State)")
lines!(ax_noisy_densities, 1:length(e2_vec_original_noisy), e2_vec_original_noisy, 
       color=:red, linewidth=2, label="e2 (Bound State)")

axislegend(ax_noisy_densities, position=:rt)

# Plot 2: Inter-particle distance for noisy data
ax_noisy_positions = Axis(fig_noisy[2, 1], 
                         title="Noisy Data: Inter-Particle Distance", 
                         xlabel="Time Step", 
                         ylabel="Distance")

lines!(ax_noisy_positions, 1:length(distances_noisy), distances_noisy, 
       color=:green, linewidth=2, label="Distance")
hlines!(ax_noisy_positions, [sim_noisy.d_dimer], color=:black, linestyle=:dash, linewidth=1, label="d_dimer = $(sim_noisy.d_dimer)")

axislegend(ax_noisy_positions, position=:rt)

# Save noisy data figure
save("noisy_data_analysis.png", fig_noisy)
println("✅ Noisy data analysis plot saved as 'noisy_data_analysis.png'")

# Display noisy data figure
display(fig_noisy)


