using SMLMSim
using GLMakie
using Random
using Distributions
using Optim

# ========================================
# CUSTOM TYPES DEFINITION
# ========================================

"""
Particle structure containing position and time information
"""
struct Particles
    x::Float64
    y::Float64
    t::Float64
end

"""
Simulation structure containing particle trajectories and parameters
"""
struct simulation
    particle_1::Vector{Particles}
    particle_2::Vector{Particles}
    k_states::Vector{Float64}
    sigma::Float64
    D::Float64
    dt::Float64
    d_dimer::Float64
end

# ========================================
# FORWARD ALGORITHM AND OPTIMIZATION
# ========================================

"""
Forward algorithm implementation for HMM state estimation
"""
function forward_algorithm(sim::simulation; dt=0.01, sigma=0.1)
    n_steps = length(sim.particle_1) - 1
    
    # Initialize alpha matrix (2 states × n_steps)
    alpha = zeros(2, n_steps + 1)
    
    # Initial state probabilities
    alpha[1, 1] = 0.7  # Free state
    alpha[2, 1] = 0.3  # Bound state
    
    # Transition matrix
    k_on, k_off = sim.k_states
    P = [1 - k_on*dt   k_on*dt;
         k_off*dt      1 - k_off*dt]
    
    # Forward pass
    for t in 2:(n_steps + 1)
        # Get positions at time t-1
        x1, y1 = sim.particle_1[t].x, sim.particle_1[t].y
        x2, y2 = sim.particle_2[t].x, sim.particle_2[t].y
        
        # Calculate distance
        distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        
        # Emission probabilities
        # State 1 (free): higher probability for large distances
        # State 2 (bound): higher probability for small distances
        emit_free = exp(-0.5 * (distance - 0.2)^2 / sigma^2)
        emit_bound = exp(-0.5 * (distance - sim.d_dimer)^2 / sigma^2)
        
        # Normalize emissions
        emit_total = emit_free + emit_bound
        emit_free /= emit_total
        emit_bound /= emit_total
        
        # Forward step
        for j in 1:2
            alpha[j, t] = 0.0
            for i in 1:2
                alpha[j, t] += alpha[i, t-1] * P[i, j]
            end
            if j == 1
                alpha[j, t] *= emit_free
            else
                alpha[j, t] *= emit_bound
            end
        end
        
        # Normalize to prevent underflow
        alpha_sum = sum(alpha[:, t])
        if alpha_sum > 0
            alpha[:, t] ./= alpha_sum
        end
    end
    
    # Calculate log-likelihood
    loglik = sum(log.(sum(alpha[:, t]) + 1e-10) for t in 2:(n_steps + 1))
    
    return alpha, loglik
end

"""
Extract free segments from simulation
"""
function extract_free_segments(sim::simulation; threshold=0.15)
    n_steps = length(sim.particle_1)
    segments = []
    current_segment = []
    
    for i in 1:n_steps
        x1, y1 = sim.particle_1[i].x, sim.particle_1[i].y
        x2, y2 = sim.particle_2[i].x, sim.particle_2[i].y
        distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        
        if distance > threshold
            push!(current_segment, i)
        else
            if !isempty(current_segment)
                push!(segments, current_segment)
                current_segment = []
            end
        end
    end
    
    if !isempty(current_segment)
        push!(segments, current_segment)
    end
    
    return segments
end

"""
Extract bound segments from simulation
"""
function extract_bound_segments(sim::simulation; threshold=0.15)
    n_steps = length(sim.particle_1)
    segments = []
    current_segment = []
    
    for i in 1:n_steps
        x1, y1 = sim.particle_1[i].x, sim.particle_1[i].y
        x2, y2 = sim.particle_2[i].x, sim.particle_2[i].y
        distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        
        if distance <= threshold
            push!(current_segment, i)
        else
            if !isempty(current_segment)
                push!(segments, current_segment)
                current_segment = []
            end
        end
    end
    
    if !isempty(current_segment)
        push!(segments, current_segment)
    end
    
    return segments
end

"""
Analyze simulation states
"""
function analyze_simulation_states(sim::simulation)
    alpha, loglik = forward_algorithm(sim)
    
    # Most likely states
    estimated_states = [argmax(alpha[:, t]) for t in 1:size(alpha, 2)]
    
    return (
        alpha = alpha,
        loglikelihood = loglik,
        estimated_states = estimated_states,
        free_probability = alpha[1, :],
        bound_probability = alpha[2, :]
    )
end

"""
Get position matrices from simulation
"""
function get_position_matrices(sim::simulation)
    n_steps = length(sim.particle_1)
    
    p1 = zeros(2, n_steps)
    p2 = zeros(2, n_steps)
    
    for i in 1:n_steps
        p1[1, i] = sim.particle_1[i].x
        p1[2, i] = sim.particle_1[i].y
        p2[1, i] = sim.particle_2[i].x
        p2[2, i] = sim.particle_2[i].y
    end
    
    return p1, p2
end

"""
Optimization result structure
"""
struct OptimizationResult
    k_on::Float64
    k_off::Float64
    loglikelihood::Float64
    optimization_result::Optim.OptimizationResults
end

"""
Optimize k parameters using custom simulation data
"""
function optimize_k_parameters_custom(sim::simulation; initial_params=[0.1, 0.1], dt=0.01, sigma=0.1)
    
    function objective(params)
        k_on, k_off = abs.(params)  # Ensure positive rates
        
        # Create temporary simulation with new parameters
        temp_sim = simulation(sim.particle_1, sim.particle_2, [k_on, k_off], 
                            sim.sigma, sim.D, sim.dt, sim.d_dimer)
        
        # Calculate negative log-likelihood
        _, loglik = forward_algorithm(temp_sim; dt=dt, sigma=sigma)
        return -loglik
    end
    
    # Optimize
    result = optimize(objective, initial_params, BFGS())
    
    # Extract optimized parameters
    opt_params = abs.(Optim.minimizer(result))
    k_on_opt, k_off_opt = opt_params
    
    # Calculate final log-likelihood
    final_sim = simulation(sim.particle_1, sim.particle_2, [k_on_opt, k_off_opt], 
                          sim.sigma, sim.D, sim.dt, sim.d_dimer)
    _, final_loglik = forward_algorithm(final_sim; dt=dt, sigma=sigma)
    
    return OptimizationResult(k_on_opt, k_off_opt, final_loglik, result)
end

# ========================================
# SMLMSIM INTEGRATION WITH DIFFUSION DYNAMICS
# ========================================

"""
Create simulation using SMLMSim's diffusion dynamics with DiffusionSMLMParams
Usage: sim = run_simulation_smlms(k_off=0.3, k_on=0.1, ...)
"""
function run_simulation_smlms(;k_off=0.3, d_dimer=0.08, t_max=10.0, dt=0.01,
                             sigma=0.01, D=0.1, box_size=1.0, r_react=0.05)
    
    println("🔬 Creating SMLMSim diffusion simulation...")
    println("   Parameters: k_off=$(k_off), r_react=$(r_react), d_dimer=$(d_dimer)")
    println("   Diffusion: D=$(D), box_size=$(box_size), t_max=$(t_max)")
    
    # Step 1: Set up SMLMSim diffusion parameters
    println("   Setting up SMLMSim DiffusionSMLMParams...")

        # Create SMLMSim parameters structure
        params = SMLMSim.DiffusionSMLMParams(
            box_size = box_size,
            diff_monomer = D,        # Monomer diffusion coefficient
            diff_dimer = D * 0.5,    # Dimer diffuses slower
            k_off = k_off,
            r_react = r_react,       # Reaction radius (relates to k_on)
            d_dimer = d_dimer,
            dt = dt,
            t_max = t_max,
            boundary = "reflecting",
            camera_framerate = 1.0/dt
        )
        
        println("   ✅ DiffusionSMLMParams created successfully")
        
        # Step 2: Create initial particles
        println("   Creating initial DiffusingEmitter2D particles...")
        particle1 = SMLMSim.DiffusingEmitter2D{Float64}(
            box_size * 0.3, box_size * 0.3,  # Position in lower-left area
            1000.0,         # Photons
            0.0,            # Initial timestamp
            1,              # Initial frame
            1,              # Dataset
            1,              # track_id
            :monomer,       # Initial state
            nothing         # No partner initially
        )
        
        particle2 = SMLMSim.DiffusingEmitter2D{Float64}(
            box_size * 0.7, box_size * 0.7,  # Position in upper-right area
            1000.0,         # Photons
            0.0,            # Initial timestamp
            1,              # Initial frame
            1,              # Dataset
            2,              # track_id
            :monomer,       # Initial state
            nothing         # No partner initially
        )
        
        println("   ✅ Initial particles created")
        println("      P1: ($(round(particle1.x, digits=3)), $(round(particle1.y, digits=3)))")
        println("      P2: ($(round(particle2.x, digits=3)), $(round(particle2.y, digits=3)))")
        
        # Step 3: Run SMLMSim diffusion simulation
        println("   Running SMLMSim diffusion simulation...")
        smld = SMLMSim.simulate(params; starting_conditions=[particle1, particle2])
        
        # Step 4: Extract tracks
        track_smlds = SMLMSim.get_tracks(smld)
        
        if length(track_smlds) < 2
            error("Expected 2 tracks but got $(length(track_smlds))")
        end
        
        # Step 5: Convert to trajectories
        println("   Converting SMLMSim tracks to custom simulation format...")
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
        
        # Sort trajectories by ID to ensure consistent ordering
        sort!(trajectories, by = t -> t.id)
        
        # Step 6: Convert to custom simulation type
        println("   Converting to custom simulation type...")
        
        # Find the common time grid (interpolate to regular dt if needed)
        n_steps = Int(ceil(t_max / dt))
        time_stamps = collect(0:dt:(n_steps-1)*dt)
        
        # Interpolate particle trajectories to regular time grid
        function interpolate_trajectory(traj_times, traj_values, target_times)
            interpolated = zeros(length(target_times))
            for (i, t) in enumerate(target_times)
                if t <= traj_times[1]
                    interpolated[i] = traj_values[1]
                elseif t >= traj_times[end]
                    interpolated[i] = traj_values[end]
                else
                    # Linear interpolation
                    idx = findfirst(tt -> tt >= t, traj_times)
                    if idx > 1
                        t1, t2 = traj_times[idx-1], traj_times[idx]
                        v1, v2 = traj_values[idx-1], traj_values[idx]
                        interpolated[i] = v1 + (v2 - v1) * (t - t1) / (t2 - t1)
                    else
                        interpolated[i] = traj_values[idx]
                    end
                end
            end
            return interpolated
        end
        
        # Get particle 1 and 2 trajectories
        traj1 = trajectories[1]
        traj2 = trajectories[2]
        
        # Interpolate positions
        p1x = interpolate_trajectory(traj1.times, traj1.x, time_stamps)
        p1y = interpolate_trajectory(traj1.times, traj1.y, time_stamps)
        p2x = interpolate_trajectory(traj2.times, traj2.x, time_stamps)
        p2y = interpolate_trajectory(traj2.times, traj2.y, time_stamps)
        
        # Create particle vectors
        particle_1 = [Particles(p1x[i], p1y[i], time_stamps[i]) for i in 1:n_steps]
        particle_2 = [Particles(p2x[i], p2y[i], time_stamps[i]) for i in 1:n_steps]
        
        # Calculate effective k_on from r_react (SMLMSim internal relationship)
        # This is an approximation - the actual relationship depends on SMLMSim's implementation
        effective_k_on = r_react * 10.0  # Scaling factor, may need adjustment
        
        k_states = [effective_k_on, k_off]
        sim = simulation(particle_1, particle_2, k_states, sigma, D, dt, d_dimer)
        
        println("   ✅ SMLMSim diffusion simulation completed")
        println("      Total time steps: $(n_steps)")
        println("      Duration: $(round(t_max, digits=2)) s")
        println("      Effective k_on: $(round(effective_k_on, digits=4))")
        println("      k_off: $(round(k_off, digits=4))")
        
        # Step 7: Analysis of binding events from SMLMSim states
        binding_events = 0
        for traj in trajectories
            for i in 2:length(traj.states)
                if traj.states[i-1] == :monomer && traj.states[i] == :dimer
                    binding_events += 1
                end
            end
        end
        
        println("      Binding events observed: $(binding_events)")
        
        return sim
        
    
end

"""
Simple fallback simulation (original approach)
"""
function run_simulation_simple(;k_off=0.3, k_on=0.1, d_dimer=0.08, t_max=10.0, dt=0.01,
                              sigma=0.01, D=0.1, box_size=1.0)
    
    println("🧪 Creating simple 2-particle simulation...")
    
    n_steps = Int(ceil(t_max / dt))
    time_stamps = collect(0:dt:(n_steps-1)*dt)
    
    # Initialize positions
    p1x = zeros(n_steps)
    p1y = zeros(n_steps)
    p2x = zeros(n_steps)
    p2y = zeros(n_steps)
    
    # Set initial positions
    p1x[1] = box_size * 0.3
    p1y[1] = box_size * 0.3
    p2x[1] = box_size * 0.7
    p2y[1] = box_size * 0.7
    
    # Simulate dynamics
    bound = sqrt((p1x[1] - p2x[1])^2 + (p1y[1] - p2y[1])^2) <= d_dimer
    
    for i in 2:n_steps
        # Diffusion step
        diffusion_step = sqrt(2 * D * dt)
        
        p1x[i] = p1x[i-1] + randn() * diffusion_step
        p1y[i] = p1y[i-1] + randn() * diffusion_step
        p2x[i] = p2x[i-1] + randn() * diffusion_step
        p2y[i] = p2y[i-1] + randn() * diffusion_step
        
        # Apply boundary conditions (reflecting)
        p1x[i] = max(0, min(box_size, p1x[i]))
        p1y[i] = max(0, min(box_size, p1y[i]))
        p2x[i] = max(0, min(box_size, p2x[i]))
        p2y[i] = max(0, min(box_size, p2y[i]))
        
        # Distance and binding logic
        dist = sqrt((p1x[i] - p2x[i])^2 + (p1y[i] - p2y[i])^2)
        
        if !bound && dist < d_dimer && rand() < k_on * dt
            bound = true
        elseif bound && rand() < k_off * dt
            bound = false
        end
        
        # If bound, keep particles close
        if bound
            p2x[i] = p1x[i] + randn() * d_dimer * 0.1
            p2y[i] = p1y[i] + randn() * d_dimer * 0.1
            # Re-apply boundaries
            p2x[i] = max(0, min(box_size, p2x[i]))
            p2y[i] = max(0, min(box_size, p2y[i]))
        end
    end
    
    # Convert to custom simulation type
    particle_1 = [Particles(p1x[i], p1y[i], time_stamps[i]) for i in 1:n_steps]
    particle_2 = [Particles(p2x[i], p2y[i], time_stamps[i]) for i in 1:n_steps]
    
    k_states = [k_on, k_off]
    sim = simulation(particle_1, particle_2, k_states, sigma, D, dt, d_dimer)
    
    println("   ✅ Simple simulation completed")
    return sim
end

"""
Convert SMLMSim 2-particle data to custom simulation type with dynamics
"""
function smlmsim_to_custom_simulation(smld; t_max=10.0, dt=0.01, k_on=0.1, k_off=0.3, 
                                     sigma=0.01, D=0.1, d_dimer=0.08)
    println("🔄 Converting SMLMSim to custom simulation with dynamics...")
    
    if length(smld.x) != 2
        error("Need exactly 2 particles for conversion")
    end
    
    # Extract initial positions
    x1_init, y1_init = smld.x[1], smld.y[1]
    x2_init, y2_init = smld.x[2], smld.y[2]
    
    println("   Initial positions from SMLMSim:")
    println("     Particle 1: ($(round(x1_init, digits=3)), $(round(y1_init, digits=3)))")
    println("     Particle 2: ($(round(x2_init, digits=3)), $(round(y2_init, digits=3)))")
    
    # Run dynamics simulation starting from SMLMSim positions
    n_steps = Int(ceil(t_max / dt))
    time_stamps = collect(0:dt:(n_steps-1)*dt)
    
    # Initialize arrays
    p1x = zeros(n_steps)
    p1y = zeros(n_steps)
    p2x = zeros(n_steps)
    p2y = zeros(n_steps)
    
    # Set initial positions from SMLMSim
    p1x[1] = x1_init
    p1y[1] = y1_init
    p2x[1] = x2_init
    p2y[1] = y2_init
    
    # Simulate dynamics
    bound = sqrt((p1x[1] - p2x[1])^2 + (p1y[1] - p2y[1])^2) <= d_dimer
    
    for i in 2:n_steps
        # Diffusion step
        diffusion_step = sqrt(2 * D * dt)
        
        p1x[i] = p1x[i-1] + randn() * diffusion_step
        p1y[i] = p1y[i-1] + randn() * diffusion_step
        p2x[i] = p2x[i-1] + randn() * diffusion_step
        p2y[i] = p2y[i-1] + randn() * diffusion_step
        
        # Distance and binding logic
        dist = sqrt((p1x[i] - p2x[i])^2 + (p1y[i] - p2y[i])^2)
        
        if !bound && dist < d_dimer && rand() < k_on * dt
            bound = true
        elseif bound && rand() < k_off * dt
            bound = false
        end
        
        # If bound, keep particles close
        if bound
            p2x[i] = p1x[i] + randn() * d_dimer * 0.1
            p2y[i] = p1y[i] + randn() * d_dimer * 0.1
        end
    end
    
    # Convert to custom simulation type
    particle_1 = [Particles(p1x[i], p1y[i], time_stamps[i]) for i in 1:n_steps]
    particle_2 = [Particles(p2x[i], p2y[i], time_stamps[i]) for i in 1:n_steps]
    k_states = [k_on, k_off]
    
    sim = simulation(particle_1, particle_2, k_states, sigma, D, dt, d_dimer)
    
    println("   ✅ Conversion completed")
    println("      Total time steps: $(n_steps)")
    println("      Duration: $(round(t_max, digits=2)) s")
    
    return sim
end

"""
Create pure custom simulation (without SMLMSim)
"""
function create_custom_simulation(;x1_init=0.5, y1_init=0.5, x2_init=1.0, y2_init=0.5,
                                 t_max=10.0, dt=0.01, k_on=0.1, k_off=0.3, 
                                 sigma=0.01, D=0.1, d_dimer=0.08)
    println("🧪 Creating pure custom simulation...")
    
    # Run dynamics simulation
    n_steps = Int(ceil(t_max / dt))
    time_stamps = collect(0:dt:(n_steps-1)*dt)
    
    # Initialize arrays
    p1x = zeros(n_steps)
    p1y = zeros(n_steps)
    p2x = zeros(n_steps)
    p2y = zeros(n_steps)
    
    # Set initial positions
    p1x[1] = x1_init
    p1y[1] = y1_init
    p2x[1] = x2_init
    p2y[1] = y2_init
    
    # Simulate dynamics
    bound = sqrt((p1x[1] - p2x[1])^2 + (p1y[1] - p2y[1])^2) <= d_dimer
    
    for i in 2:n_steps
        # Diffusion step
        diffusion_step = sqrt(2 * D * dt)
        
        p1x[i] = p1x[i-1] + randn() * diffusion_step
        p1y[i] = p1y[i-1] + randn() * diffusion_step
        p2x[i] = p2x[i-1] + randn() * diffusion_step
        p2y[i] = p2y[i-1] + randn() * diffusion_step
        
        # Distance and binding logic
        dist = sqrt((p1x[i] - p2x[i])^2 + (p1y[i] - p2y[i])^2)
        
        if !bound && dist < d_dimer && rand() < k_on * dt
            bound = true
        elseif bound && rand() < k_off * dt
            bound = false
        end
        
        # If bound, keep particles close
        if bound
            p2x[i] = p1x[i] + randn() * d_dimer * 0.1
            p2y[i] = p1y[i] + randn() * d_dimer * 0.1
        end
    end
    
    # Convert to custom simulation type
    particle_1 = [Particles(p1x[i], p1y[i], time_stamps[i]) for i in 1:n_steps]
    particle_2 = [Particles(p2x[i], p2y[i], time_stamps[i]) for i in 1:n_steps]
    k_states = [k_on, k_off]
    
    sim = simulation(particle_1, particle_2, k_states, sigma, D, dt, d_dimer)
    
    println("   ✅ Custom simulation completed")
    println("      Total time steps: $(n_steps)")
    println("      Duration: $(round(t_max, digits=2)) s")
    
    return sim
end

# ========================================
# ANIMATION FUNCTIONS
# ========================================

"""
Animate custom simulation (works with both SMLMSim-derived and pure custom)
"""
function animate_simulation(sim::simulation, filename="simulation_animation.mp4"; 
                           fps=30, duration=nothing, show_trail=true, trail_length=50)
    println("🎬 Creating simulation animation...")
    
    # Extract particle data
    n_steps = length(sim.particle_1)
    p1_x = [p.x for p in sim.particle_1]
    p1_y = [p.y for p in sim.particle_1]
    p2_x = [p.x for p in sim.particle_2]
    p2_y = [p.y for p in sim.particle_2]
    times = [p.t for p in sim.particle_1]
    
    # Calculate distances and binding states
    distances = [sqrt((p1_x[i] - p2_x[i])^2 + (p1_y[i] - p2_y[i])^2) for i in 1:n_steps]
    bound_states = distances .<= sim.d_dimer
    
    # Calculate frame parameters
    if duration === nothing
        duration = times[end] - times[1]
    end
    n_frames = Int(round(fps * duration))
    frame_indices = round.(Int, range(1, n_steps, length=n_frames))
    
    # Calculate plot bounds with padding
    all_x = vcat(p1_x, p2_x)
    all_y = vcat(p1_y, p2_y)
    x_range = (minimum(all_x) - 0.1, maximum(all_x) + 0.1)
    y_range = (minimum(all_y) - 0.1, maximum(all_y) + 0.1)
    
    println("   Animation parameters:")
    println("      Frames: $(n_frames)")
    println("      Duration: $(round(duration, digits=2)) s")
    println("      FPS: $(fps)")
    println("      Total time steps: $(n_steps)")
    
    # Create figure
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], 
              title="Two-Particle Interaction Simulation",
              xlabel="X Position (μm)",
              ylabel="Y Position (μm)",
              aspect=DataAspect())
    
    # Set axis limits
    xlims!(ax, x_range...)
    ylims!(ax, y_range...)
    
    # Initialize observables for animation
    p1_pos = Observable(Point2f(p1_x[1], p1_y[1]))
    p2_pos = Observable(Point2f(p2_x[1], p2_y[1]))
    is_bound = Observable(bound_states[1])
    current_time = Observable(times[1])
    
    # Particle colors based on binding state
    p1_color = @lift($is_bound ? :red : :blue)
    p2_color = @lift($is_bound ? :red : :cyan)
    
    # Plot particles
    scatter!(ax, p1_pos, color=p1_color, markersize=20, label="Particle 1")
    scatter!(ax, p2_pos, color=p2_color, markersize=20, label="Particle 2")
    
    # Connection line when bound
    line_points = @lift([$p1_pos, $p2_pos])
    line_color = @lift($is_bound ? :red : :transparent)
    lines!(ax, line_points, color=line_color, linewidth=3, alpha=0.7)
    
    # Binding threshold circle around particle 1
    circle_points = @lift([Point2f($p1_pos[1] + sim.d_dimer * cos(θ), 
                                   $p1_pos[2] + sim.d_dimer * sin(θ)) for θ in 0:0.1:2π])
    lines!(ax, circle_points, color=:gray, linestyle=:dash, alpha=0.5, linewidth=1)
    
    # Trail visualization
    if show_trail
        trail_p1 = Observable(Point2f[])
        trail_p2 = Observable(Point2f[])
        lines!(ax, trail_p1, color=(:blue, 0.3), linewidth=2, label="P1 Trail")
        lines!(ax, trail_p2, color=(:cyan, 0.3), linewidth=2, label="P2 Trail")
    end
    
    # Time and state information
    time_text = @lift("Time: $(round($current_time, digits=3)) s")
    state_text = @lift($is_bound ? "State: BOUND" : "State: FREE")
    distance_text = @lift("Distance: $(round(sqrt(($p1_pos[1] - $p2_pos[1])^2 + 
                                                  ($p1_pos[2] - $p2_pos[2])^2), digits=4)) μm")
    
    text!(ax, x_range[1] + 0.05, y_range[2] - 0.05, text=time_text, 
          fontsize=14, color=:black)
    text!(ax, x_range[1] + 0.05, y_range[2] - 0.15, text=state_text, 
          fontsize=14, color=@lift($is_bound ? :red : :blue))
    text!(ax, x_range[1] + 0.05, y_range[2] - 0.25, text=distance_text, 
          fontsize=12, color=:gray)
    
    # Parameters text
    params_text = "k_on = $(sim.k_states[1]) s⁻¹, k_off = $(sim.k_states[2]) s⁻¹, d_dimer = $(sim.d_dimer) μm"
    text!(ax, x_range[1] + 0.05, y_range[1] + 0.05, text=params_text, 
          fontsize=10, color=:gray)
    
    # Legend
    axislegend(ax, position=:rt)
    
    # Record animation
    println("   🎥 Recording animation...")
    record(fig, filename, frame_indices; framerate=fps) do frame_idx
        # Update positions
        p1_pos[] = Point2f(p1_x[frame_idx], p1_y[frame_idx])
        p2_pos[] = Point2f(p2_x[frame_idx], p2_y[frame_idx])
        is_bound[] = bound_states[frame_idx]
        current_time[] = times[frame_idx]
        
        # Update trails
        if show_trail
            start_idx = max(1, frame_idx - trail_length)
            trail_p1[] = [Point2f(p1_x[i], p1_y[i]) for i in start_idx:frame_idx]
            trail_p2[] = [Point2f(p2_x[i], p2_y[i]) for i in start_idx:frame_idx]
        end
    end
    
    println("   ✅ Animation saved as: $(filename)")
    println("      File size: $(round(stat(filename).size / 1024^2, digits=2)) MB")
    
    return filename
end

# ========================================
# COMPREHENSIVE ANALYSIS FUNCTION
# ========================================

"""
Complete analysis of simulation including forward algorithm and optimization
"""
function analyze_simulation_complete(sim::simulation; dt=0.01, sigma=0.1, 
                                   optimize_params=true, create_animation=true)
    println("📊 Complete simulation analysis...")
    
    # 1. Forward algorithm
    println("\n🔍 Running forward algorithm...")
    alpha, loglik = forward_algorithm(sim; dt=dt, sigma=sigma)
    e1_vec = alpha[1, :]
    e2_vec = alpha[2, :]
    
    println("   Log-likelihood: $(round(loglik, digits=2))")
    
    # 2. Extract segments
    println("\n📏 Extracting binding segments...")
    free_segments = extract_free_segments(sim)
    bound_segments = extract_bound_segments(sim)
    
    println("   Free segments: $(length(free_segments))")
    println("   Bound segments: $(length(bound_segments))")
    
    # 3. Parameter optimization
    opt_result = nothing
    if optimize_params
        println("\n⚙️  Optimizing parameters...")
        opt_result = optimize_k_parameters_custom(sim; dt=dt, sigma=sigma)
        
        println("   Original k_on:  $(round(sim.k_states[1], digits=4))")
        println("   Original k_off: $(round(sim.k_states[2], digits=4))")
        println("   Optimized k_on:  $(round(opt_result.k_on, digits=4))")
        println("   Optimized k_off: $(round(opt_result.k_off, digits=4))")
        println("   Improvement: $(round(opt_result.loglikelihood - loglik, digits=2))")
    end
    
    # 4. Create animation
    animation_file = nothing
    if create_animation
        println("\n🎬 Creating animation...")
        animation_file = animate_simulation(sim, "complete_analysis.mp4", fps=25)
    end
    
    # 5. Summary
    println("\n📈 Analysis Summary:")
    println("   Simulation length: $(length(sim.particle_1)) steps")
    println("   Total time: $(round(sim.particle_1[end].t, digits=2)) s")
    println("   Binding events: $(length(bound_segments))")
    println("   Free periods: $(length(free_segments))")
    if opt_result !== nothing
        println("   Optimization converged: $(Optim.converged(opt_result.optimization_result))")
    end
    
    return (
        alpha = alpha,
        loglikelihood = loglik,
        free_segments = free_segments,
        bound_segments = bound_segments,
        optimization_result = opt_result,
        animation_file = animation_file
    )
end

# ========================================
# EXAMPLE FUNCTIONS
# ========================================

"""
Example 1: Using SMLMSim for initial positions
"""
function example_smlmsim_workflow()
    println("=" ^ 60)
    println("EXAMPLE 1: SMLMSim + Custom Dynamics Workflow")
    println("=" ^ 60)
    
    Random.seed!(1234)
    
    # Step 1: Create SMLMSim 2-particle setup
    smld = smlmsim_two_particles(rho=0.5, separation=0.08, xsize=3.0, ysize=3.0)
    
    # Step 2: Convert to custom simulation with dynamics
    sim = smlmsim_to_custom_simulation(smld; 
                                      t_max=8.0, 
                                      k_on=0.2, 
                                      k_off=0.3, 
                                      d_dimer=0.1)
    
    # Step 3: Complete analysis
    analysis = analyze_simulation_complete(sim; 
                                         optimize_params=true, 
                                         create_animation=true)
    
    println("\n✅ SMLMSim workflow completed!")
    return sim, analysis
end

"""
Example 2: Pure custom simulation
"""
function example_custom_workflow()
    println("=" ^ 60)
    println("EXAMPLE 2: Pure Custom Simulation Workflow")
    println("=" ^ 60)
    
    Random.seed!(1234)
    
    # Step 1: Create custom simulation
    sim = create_custom_simulation(x1_init=0.5, y1_init=0.5, 
                                  x2_init=1.2, y2_init=0.8,
                                  t_max=10.0, 
                                  k_on=0.15, 
                                  k_off=0.25, 
                                  d_dimer=0.12)
    
    # Step 2: Complete analysis  
    analysis = analyze_simulation_complete(sim; 
                                         optimize_params=true, 
                                         create_animation=true)
    
    println("\n✅ Custom workflow completed!")
    return sim, analysis
end

"""
Example 3: Comparison between SMLMSim and custom initialization
"""
function example_comparison()
    println("=" ^ 60)
    println("EXAMPLE 3: SMLMSim vs Custom Comparison")
    println("=" ^ 60)
    
    Random.seed!(1234)
    
    # SMLMSim version
    println("\n🔬 SMLMSim version:")
    smld = smlmsim_two_particles()
    sim_smlm = smlmsim_to_custom_simulation(smld; t_max=6.0)
    
    # Custom version with same parameters
    println("\n🧪 Custom version:")
    sim_custom = create_custom_simulation(x1_init=smld.x[1], y1_init=smld.y[1],
                                         x2_init=smld.x[2], y2_init=smld.y[2],
                                         t_max=6.0, 
                                         k_on=sim_smlm.k_states[1], 
                                         k_off=sim_smlm.k_states[2])
    
    # Analyze both
    println("\n📊 Analysis comparison:")
    analysis_smlm = analyze_simulation_complete(sim_smlm; create_animation=false)
    analysis_custom = analyze_simulation_complete(sim_custom; create_animation=false)
    
    # Create comparison animation
    animate_simulation(sim_smlm, "smlmsim_version.mp4", fps=25)
    animate_simulation(sim_custom, "custom_version.mp4", fps=25)
    
    println("\n🎯 Comparison Summary:")
    println("   SMLMSim likelihood: $(round(analysis_smlm.loglikelihood, digits=2))")
    println("   Custom likelihood:  $(round(analysis_custom.loglikelihood, digits=2))")
    
    return sim_smlm, sim_custom, analysis_smlm, analysis_custom
end

println("🎯 Final Implementation Module Loaded Successfully!")
println("=" ^ 60)
println("Available workflows:")
println("  📍 example_smlmsim_workflow()   - SMLMSim + dynamics")
println("  🧪 example_custom_workflow()    - Pure custom simulation")  
println("  📊 example_comparison()         - Compare both approaches")
println()
println("Core functions:")
println("  🔬 smlmsim_two_particles()      - Create 2-particle SMLMSim setup")
println("  🔄 smlmsim_to_custom_simulation() - Convert SMLMSim to custom with dynamics")
println("  🧪 create_custom_simulation()   - Create pure custom simulation")
println("  📊 analyze_simulation_complete() - Full analysis with optimization")
println("  🎬 animate_simulation()         - Create animation")
println("=" ^ 60)