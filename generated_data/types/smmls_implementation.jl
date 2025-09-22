using SMLMSim
using CairoMakie
using Random
using Distributions
using Optim

# Include the custom types from types_with_data.jl
include("types_with_data.jl")

# ========================================
# SMM SIMULATION WITH CUSTOM TYPES
# ========================================

"""
Run SMM simulation using SMLMSim and convert to custom simulation type
"""
function run_smm_simulation(;density=0.01, t_max=50, box_size=10, k_off=0.5, r_react=4, 
                           save=false, boundary="reflecting", d_dimer=0.05, σ=0.01)
    
    # Run the original SMLMSim simulation
    state_history, args = SMLMSim.InteractionDiffusion.smoluchowski(
        density=density,
        t_max=t_max, 
        box_size=box_size,
        k_off=k_off,
        r_react=r_react,
        boundary=boundary
    )
    
    # Extract particle trajectories
    p1x, p1y, p1s, p2x, p2y, p2s, time_stamps = extract_particle_trajectories_with_time(state_history)
    
    # Convert to custom types
    sim = convert_to_custom_simulation(p1x, p1y, p2x, p2y, time_stamps, k_off, r_react, 
                                     d_dimer=d_dimer, σ=σ)
    
    if save
        SMLMSim.gen_movie(state_history, args; filename="smm_simulation.mp4")
        println(" SMM simulation animation saved as 'smm_simulation.mp4'")
    end
    
    return sim, state_history, args
end

"""
Extract particle trajectories with time information from SMM simulation
"""
function extract_particle_trajectories_with_time(state_history)
    num_frames = size(state_history.frames, 1)
    particle1_x = zeros(num_frames)
    particle1_y = zeros(num_frames)
    particle1_state = zeros(Int, num_frames)
    particle2_x = zeros(num_frames)
    particle2_y = zeros(num_frames)
    particle2_state = zeros(Int, num_frames)
    
    # Extract time stamps (assuming uniform dt)
    time_stamps = zeros(num_frames)
    if hasfield(typeof(state_history), :dt)
        dt = state_history.dt
    else
        # Estimate dt from total time and frames
        dt = state_history.args.t_max / num_frames
    end
    
    for frame_idx in eachindex(state_history.frames)
        current_frame = state_history.frames[frame_idx]
        particle1_x[frame_idx] = current_frame.molecules[1].x
        particle1_y[frame_idx] = current_frame.molecules[1].y
        particle2_x[frame_idx] = current_frame.molecules[2].x
        particle2_y[frame_idx] = current_frame.molecules[2].y
        particle2_state[frame_idx] = current_frame.molecules[2].state
        particle1_state[frame_idx] = current_frame.molecules[1].state
        time_stamps[frame_idx] = (frame_idx - 1) * dt
    end

    return particle1_x, particle1_y, particle1_state, particle2_x, particle2_y, particle2_state, time_stamps
end

"""
Convert SMM simulation data to custom simulation type
"""
function convert_to_custom_simulation(p1x, p1y, p2x, p2y, time_stamps, k_off, r_react; 
                                    d_dimer=0.05, σ=0.01, D=0.01)
    n_steps = length(p1x)
    
    # Create Particles vectors
    particle_1 = [Particles(p1x[i], p1y[i], time_stamps[i]) for i in 1:n_steps]
    particle_2 = [Particles(p2x[i], p2y[i], time_stamps[i]) for i in 1:n_steps]
    
    # Estimate states based on distances and d_dimer threshold
    states = estimate_states_from_positions(p1x, p1y, p2x, p2y, d_dimer)
    
    # Set k_states: r_react can be related to k_on, k_off is given
    k_on = r_react * 0.1  # Simple conversion, adjust as needed
    k_states = [k_on, k_off]
    
    # Calculate dt from time stamps
    dt = length(time_stamps) > 1 ? time_stamps[2] - time_stamps[1] : 0.01
    
    return simulation(particle_1, particle_2, k_states, σ, D, dt, d_dimer)
end

"""
Estimate states from particle positions using distance threshold
"""
function estimate_states_from_positions(p1x, p1y, p2x, p2y, d_dimer)
    n_steps = length(p1x)
    states = zeros(Int, n_steps-1)  # States for transitions
    
    for i in 1:(n_steps-1)
        dist = sqrt((p1x[i] - p2x[i])^2 + (p1y[i] - p2y[i])^2)
        states[i] = dist <= d_dimer ? 2 : 1  # 2 = bound, 1 = free
    end
    
    return states
end

# ========================================
# PREDICTION AND ANALYSIS FUNCTIONS FROM TYPES_WITH_DATA
# ========================================

"""
Run forward algorithm on SMM simulation data
"""
function predict_states_smm(sim::simulation; dt=0.01, sigma=0.1)
    return forward_algorithm(sim; dt=dt, sigma=sigma)
end

"""
Optimize parameters using SMM simulation data
"""
function optimize_parameters_smm(sim::simulation; initial_params=[0.1, 0.1], dt=0.01, sigma=0.1)
    return optimize_k_parameters_custom(sim; initial_params=initial_params, dt=dt, sigma=sigma)
end

"""
Analyze binding/unbinding events in SMM simulation
"""
function analyze_smm_binding_events(sim::simulation)
    # Use existing functions from types_with_data.jl
    free_segments = extract_free_segments(sim)
    bound_segments = extract_bound_segments(sim)
    state_analysis = analyze_simulation_states(sim)
    
    return (
        free_segments = free_segments,
        bound_segments = bound_segments,
        estimated_states = state_analysis,
        binding_events = length(bound_segments),
        unbinding_events = length(free_segments),
        total_time = sim.particle_1[end].t - sim.particle_1[1].t
    )
end

"""
Create comprehensive plots for SMM simulation analysis
"""
function plot_smm_analysis(sim::simulation; dt=0.01, sigma=0.1, save_plots=true)
    println("= Analyzing SMM simulation...")
    
    # Run forward algorithm
    alpha, loglik = forward_algorithm(sim; dt=dt, sigma=sigma)
    e1_vec = alpha[1, :]
    e2_vec = alpha[2, :]
    
    # Get particle positions and distances
    p1, p2 = get_position_matrices(sim)
    distances = [sqrt((p1[1,i] - p2[1,i])^2 + (p1[2,i] - p2[2,i])^2) for i in 1:size(p1,2)]
    
    # Analyze binding events
    analysis = analyze_smm_binding_events(sim)
    
    # Create comprehensive plot
    fig = Figure(size=(1200, 1000))
    
    # Plot 1: State densities
    ax1 = Axis(fig[1, 1], 
               title="SMM Simulation: State Densities", 
               xlabel="Time Step", 
               ylabel="Probability")
    
    lines!(ax1, 1:length(e1_vec), e1_vec, color=:blue, linewidth=2, label="e1 (Free State)")
    lines!(ax1, 1:length(e2_vec), e2_vec, color=:red, linewidth=2, label="e2 (Bound State)")
    axislegend(ax1, position=:rt)
    
    # Plot 2: Inter-particle distances
    ax2 = Axis(fig[2, 1], 
               title="SMM Simulation: Inter-Particle Distance", 
               xlabel="Time Step", 
               ylabel="Distance")
    
    lines!(ax2, 1:length(distances), distances, color=:green, linewidth=2, label="Distance")
    hlines!(ax2, [sim.d_dimer], color=:black, linestyle=:dash, linewidth=1, 
            label="d_dimer = $(sim.d_dimer)")
    axislegend(ax2, position=:rt)
    
    # Plot 3: Particle trajectories
    ax3 = Axis(fig[3, 1], 
               title="SMM Simulation: Particle Trajectories", 
               xlabel="X Position", 
               ylabel="Y Position")
    
    lines!(ax3, p1[1,:], p1[2,:], color=:blue, linewidth=2, label="Particle 1")
    lines!(ax3, p2[1,:], p2[2,:], color=:red, linewidth=2, label="Particle 2")
    axislegend(ax3, position=:rt)
    
    # Plot 4: Binding events histogram
    ax4 = Axis(fig[4, 1], 
               title="SMM Simulation: State Distribution", 
               xlabel="State", 
               ylabel="Count")
    
    state_counts = [sum(e1_vec .> 0.5), sum(e2_vec .> 0.5)]
    barplot!(ax4, [1, 2], state_counts, color=[:blue, :red])
    ax4.xticks = ([1, 2], ["Free", "Bound"])
    
    if save_plots
        save("smm_simulation_analysis.png", fig)
        println(" SMM analysis plot saved as 'smm_simulation_analysis.png'")
    end
    
    # Print analysis summary
    println("\n=σ SMM Simulation Analysis Summary:")
    println("   Total simulation time: $(round(analysis.total_time, digits=2))")
    println("   Binding events: $(analysis.binding_events)")
    println("    Unbinding events: $(analysis.unbinding_events)")
    println("    Log-likelihood: $(round(loglik, digits=2))")
    println("    k_states: [$(round(sim.k_states[1], digits=3)), $(round(sim.k_states[2], digits=3))]")
    println("    d_dimer: $(sim.d_dimer)")
    
    display(fig)
    return fig, analysis
end

# ========================================
# EXAMPLE USAGE FUNCTION
# ========================================

"""
Complete example demonstrating SMM simulation with custom types integration
"""
function example_smm_with_custom_types()
    println("=" ^ 60)
    println("SMM SIMULATION WITH CUSTOM TYPES INTEGRATION")
    println("=" ^ 60)
    
    Random.seed!(1234)
    
    # Run SMM simulation and convert to custom types
    println("\n=9 Running SMM simulation...")
    sim, state_history, args = run_smm_simulation(
        density=0.01, 
        t_max=50, 
        box_size=15, 
        k_off=0.5, 
        r_react=4,
        save=true,
        d_dimer=0.05,
        σ=0.01
    )
    
    println(" SMM simulation completed and converted to custom types")
    println("    Simulation length: $(length(sim.particle_1)) time steps")
    println("    Time range: [$(round(sim.particle_1[1].t, digits=2)), $(round(sim.particle_1[end].t, digits=2))]")
    
    # Create analysis plots
    fig, analysis = plot_smm_analysis(sim; dt=0.01, sigma=0.1)
    
    # Run parameter optimization
    println("\n=9 Running parameter optimization...")
    opt_result = optimize_parameters_smm(sim; initial_params=[0.1, 0.1], dt=0.01, sigma=0.1)
    
    println("   Optimization Results:")
    println("      Original k_on:  $(round(sim.k_states[1], digits=4))")
    println("      Original k_off: $(round(sim.k_states[2], digits=4))")
    println("      Optimized k_on:  $(round(opt_result.k_on, digits=4))")
    println("      Optimized k_off: $(round(opt_result.k_off, digits=4))")
    println("      Log-likelihood: $(round(opt_result.loglikelihood, digits=2))")
    println("      Converged: $(Optim.converged(opt_result.optimization_result))")
    
    println("\n SMM simulation with custom types integration completed!")
    
    return sim, opt_result, fig, analysis
end

println("=� SMM-Custom Types integration loaded successfully!")
println("   Use example_smm_with_custom_types() to run complete example")