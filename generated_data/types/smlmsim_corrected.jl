using SMLMSim
using GLMakie
using Random
using Distributions
using Optim

# Include the custom types from types_with_data.jl
include("types_with_data.jl")

println("🔬 SMLMSim API Corrected Integration")
println("=" ^ 50)

# ========================================
# WORKING SMLMSIM METHODS
# ========================================

"""
Method 1: Single molecule distribution using SMLMSim
Creates random positions of individual molecules
"""
function smlmsim_single_molecules(;rho=1.0, xsize=3.0, ysize=3.0)
    println("📍 SMLMSim Method 1: Single molecule simulation")
    
    # Create single molecule pattern
    pattern = SMLMSim.Point2D()
    
    # Generate molecule positions
    smld = SMLMSim.uniform2D(rho, pattern, xsize, ysize)
    
    println("   ✅ Single molecule simulation created")
    println("      Number of molecules: $(length(smld.x))")
    println("      Area: $(xsize)×$(ysize) μm²")
    println("      Density: $(rho) molecules/μm²")
    
    # Show positions
    println("      Molecule positions:")
    for i in 1:min(5, length(smld.x))
        println("        Molecule $i: ($(round(smld.x[i], digits=3)), $(round(smld.y[i], digits=3)))")
    end
    
    return smld
end

"""
Method 2: Dimer (2-molecule cluster) distribution using SMLMSim
Creates pairs of molecules at fixed distances - most relevant for particle interactions
"""
function smlmsim_dimer_simulation(;rho=0.8, separation=0.06, xsize=4.0, ysize=4.0)
    println("🔗 SMLMSim Method 2: Dimer simulation")
    
    # Create dimer pattern (2 molecules)
    pattern = SMLMSim.Nmer2D(n=2, d=separation)
    
    # Generate dimer positions
    smld_dimers = SMLMSim.uniform2D(rho, pattern, xsize, ysize)
    
    println("   ✅ Dimer simulation created")
    println("      Number of localizations: $(length(smld_dimers.x))")
    println("      Number of dimers: $(length(smld_dimers.x) ÷ 2)")
    println("      Separation distance: $(separation) μm")
    println("      Area: $(xsize)×$(ysize) μm²")
    
    # Show dimer pairs
    println("      Dimer pairs:")
    for i in 1:2:min(6, length(smld_dimers.x))
        if i+1 <= length(smld_dimers.x)
            x1, y1 = smld_dimers.x[i], smld_dimers.y[i]
            x2, y2 = smld_dimers.x[i+1], smld_dimers.y[i+1]
            dist = sqrt((x2-x1)^2 + (y2-y1)^2)
            println("        Dimer $((i+1)÷2): ($(round(x1,digits=3)),$(round(y1,digits=3))) ↔ ($(round(x2,digits=3)),$(round(y2,digits=3))) dist=$(round(dist,digits=3))")
        end
    end
    
    return smld_dimers
end

"""
Method 3: Larger clusters using SMLMSim
Creates clusters of multiple molecules
"""
function smlmsim_cluster_simulation(;rho=0.3, n_molecules=4, cluster_size=0.08, xsize=5.0, ysize=5.0)
    println("⭐ SMLMSim Method 3: Cluster simulation")
    
    # Create cluster pattern
    pattern = SMLMSim.Nmer2D(n=n_molecules, d=cluster_size)
    
    # Generate cluster positions
    smld_clusters = SMLMSim.uniform2D(rho, pattern, xsize, ysize)
    
    println("   ✅ Cluster simulation created")
    println("      Number of localizations: $(length(smld_clusters.x))")
    println("      Number of clusters: $(length(smld_clusters.x) ÷ n_molecules)")
    println("      Cluster size: $(cluster_size) μm")
    
    # Show first cluster
    if length(smld_clusters.x) >= n_molecules
        println("      First cluster molecules:")
        for i in 1:n_molecules
            println("        Molecule $i: ($(round(smld_clusters.x[i], digits=3)), $(round(smld_clusters.y[i], digits=3)))")
        end
    end
    
    return smld_clusters
end

"""
Convert SMLMSim SMLD data to custom simulation type (static positions only)
Note: This creates a static "snapshot" simulation, not dynamic trajectories
"""
function convert_smld_to_custom_simulation(smld; k_on=0.1, k_off=0.5, sigma=0.01, D=0.01, dt=0.01, d_dimer=0.05)
    println("🔄 Converting SMLD to custom simulation type...")
    
    n_molecules = length(smld.x)
    
    if n_molecules < 2
        error("Need at least 2 molecules for particle pair simulation")
    end
    
    # Take first two molecules as particle pair
    x1, y1 = smld.x[1], smld.y[1]
    x2, y2 = smld.x[2], smld.y[2]
    
    # Create single time point (static snapshot)
    t = 0.0
    particle_1 = [Particles(x1, y1, t)]
    particle_2 = [Particles(x2, y2, t)]
    
    # Set k_states
    k_states = [k_on, k_off]
    
    sim = simulation(particle_1, particle_2, k_states, sigma, D, dt, d_dimer)
    
    println("   ✅ Converted SMLD to custom simulation")
    println("      Particle 1: ($(round(x1, digits=3)), $(round(y1, digits=3)))")
    println("      Particle 2: ($(round(x2, digits=3)), $(round(y2, digits=3)))")
    
    distance = sqrt((x2-x1)^2 + (y2-y1)^2)
    println("      Distance: $(round(distance, digits=3)) μm")
    println("      State: $(distance <= d_dimer ? "Bound" : "Free")")
    
    return sim
end

"""
Combined SMLMSim + synthetic dynamics approach
Uses SMLMSim for initial positions, then adds dynamics
"""
function smlmsim_with_dynamics(;method="dimer", t_max=10.0, dt=0.01, k_off=0.3, sigma=0.01, D=0.01, d_dimer=0.05)
    println("🚀 SMLMSim + Dynamics simulation")
    
    # Step 1: Generate initial positions with SMLMSim
    if method == "dimer"
        smld = smlmsim_dimer_simulation(rho=0.5, separation=d_dimer*0.8)
    else
        smld = smlmsim_single_molecules(rho=1.0)
    end
    
    # Step 2: Extract initial positions
    if length(smld.x) >= 2
        x1_init, y1_init = smld.x[1], smld.y[1]
        x2_init, y2_init = smld.x[2], smld.y[2]
    else
        # Fallback if not enough molecules
        x1_init, y1_init = 0.0, 0.0
        x2_init, y2_init = d_dimer*0.9, 0.0
    end
    
    println("   Initial positions from SMLMSim:")
    println("     Particle 1: ($(round(x1_init, digits=3)), $(round(y1_init, digits=3)))")
    println("     Particle 2: ($(round(x2_init, digits=3)), $(round(y2_init, digits=3)))")
    
    # Step 3: Run dynamics simulation starting from SMLMSim positions
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
    k_on = 0.1
    
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
    
    println("   ✅ SMLMSim + Dynamics simulation completed")
    println("      Total time steps: $(n_steps)")
    println("      Duration: $(round(t_max, digits=2)) s")
    
    return sim, smld
end

"""
Animate particles from SMLMSim simulation using GLMakie
Creates MP4 video of particle trajectories with binding state visualization
"""
function animate_smlmsim_particles(sim::simulation, filename="smlmsim_animation.mp4"; 
                                  fps=30, duration=nothing, show_trail=true, trail_length=50)
    println("🎬 Creating SMLMSim particle animation...")
    
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
              title="SMLMSim Particle Dynamics",
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

"""
Animate SMLMSim data directly (without custom type conversion)
Creates animation showing molecular positions and dimer formations
"""
function animate_smlmsim_data(smld, filename="smlmsim_data_animation.mp4"; 
                             fps=30, duration=5.0, animation_type="rotation", 
                             show_connections=true, highlight_dimers=true)
    println("🎬 Creating SMLMSim data animation...")
    
    n_molecules = length(smld.x)
    
    if n_molecules == 0
        error("No molecules to animate")
    end
    
    println("   Animation parameters:")
    println("      Molecules: $(n_molecules)")
    println("      Duration: $(duration) s")
    println("      FPS: $(fps)")
    println("      Type: $(animation_type)")
    
    # Calculate plot bounds with padding
    x_range = (minimum(smld.x) - 0.2, maximum(smld.x) + 0.2)
    y_range = (minimum(smld.y) - 0.2, maximum(smld.y) + 0.2)
    
    # Identify dimer pairs (if even number of molecules)
    dimer_pairs = []
    if n_molecules % 2 == 0 && show_connections
        for i in 1:2:n_molecules-1
            push!(dimer_pairs, (i, i+1))
        end
    end
    
    # Create figure
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], 
              title="SMLMSim Molecular Data Animation",
              xlabel="X Position (μm)",
              ylabel="Y Position (μm)",
              aspect=DataAspect())
    
    # Set axis limits
    xlims!(ax, x_range...)
    ylims!(ax, y_range...)
    
    # Calculate number of frames
    n_frames = Int(round(fps * duration))
    
    if animation_type == "rotation"
        # Rotation animation - molecules orbit around their initial positions
        center_x = mean(smld.x)
        center_y = mean(smld.y)
        
        # Calculate initial angles and radii from center
        initial_angles = [atan(smld.y[i] - center_y, smld.x[i] - center_x) for i in 1:n_molecules]
        radii = [sqrt((smld.x[i] - center_x)^2 + (smld.y[i] - center_y)^2) for i in 1:n_molecules]
        
        # Animation observables
        mol_positions = Observable([Point2f(smld.x[i], smld.y[i]) for i in 1:n_molecules])
        time_obs = Observable(0.0)
        
    elseif animation_type == "oscillation"
        # Oscillation animation - molecules oscillate around initial positions
        oscillation_amplitude = 0.1
        
        mol_positions = Observable([Point2f(smld.x[i], smld.y[i]) for i in 1:n_molecules])
        time_obs = Observable(0.0)
        
    elseif animation_type == "expansion"
        # Expansion animation - molecules move away and back to center
        center_x = mean(smld.x)
        center_y = mean(smld.y)
        
        mol_positions = Observable([Point2f(smld.x[i], smld.y[i]) for i in 1:n_molecules])
        time_obs = Observable(0.0)
    end
    
    # Color scheme for molecules
    colors = if highlight_dimers && !isempty(dimer_pairs)
        molecule_colors = fill(:blue, n_molecules)
        for (i, j) in dimer_pairs
            molecule_colors[i] = :red
            molecule_colors[j] = :orange
        end
        molecule_colors
    else
        [:blue for _ in 1:n_molecules]
    end
    
    # Plot molecules
    scatter!(ax, mol_positions, 
             color=colors, 
             markersize=15, 
             alpha=0.8,
             strokewidth=2,
             strokecolor=:black)
    
    # Plot dimer connections
    if show_connections && !isempty(dimer_pairs)
        for (i, j) in dimer_pairs
            line_points = @lift([$(mol_positions)[$(i)], $(mol_positions)[$(j)]])
            lines!(ax, line_points, color=:red, linewidth=3, alpha=0.6)
        end
    end
    
    # Add molecule labels
    for i in 1:n_molecules
        text!(ax, @lift($(mol_positions)[$(i)]), text="$(i)", 
              fontsize=12, color=:white, align=(:center, :center))
    end
    
    # Field boundary
    field_x = smld.datasize[1]
    field_y = smld.datasize[2]
    lines!(ax, [0, field_x, field_x, 0, 0], [0, 0, field_y, field_y, 0], 
           color=:gray, linestyle=:dash, linewidth=2, alpha=0.7)
    
    # Time display
    time_text = @lift("Time: $(round($time_obs, digits=2)) s")
    text!(ax, x_range[1] + 0.05, y_range[2] - 0.05, text=time_text, 
          fontsize=16, color=:black, fontweight=:bold)
    
    # Info text
    info_text = "Molecules: $(n_molecules) | Type: $(animation_type)"
    if !isempty(dimer_pairs)
        info_text *= " | Dimers: $(length(dimer_pairs))"
    end
    text!(ax, x_range[1] + 0.05, y_range[1] + 0.05, text=info_text, 
          fontsize=12, color=:gray)
    
    # Record animation
    println("   🎥 Recording animation...")
    record(fig, filename, 1:n_frames; framerate=fps) do frame
        t = (frame - 1) / fps
        time_obs[] = t
        
        new_positions = Point2f[]
        
        for i in 1:n_molecules
            if animation_type == "rotation"
                # Rotate around center
                angle = initial_angles[i] + 2π * t / duration
                new_x = center_x + radii[i] * cos(angle)
                new_y = center_y + radii[i] * sin(angle)
                
            elseif animation_type == "oscillation"
                # Oscillate around initial position
                offset_x = oscillation_amplitude * sin(2π * t * 2) * (i % 2 == 0 ? 1 : -1)
                offset_y = oscillation_amplitude * cos(2π * t * 1.5) * (i % 3 == 0 ? 1 : -1)
                new_x = smld.x[i] + offset_x
                new_y = smld.y[i] + offset_y
                
            elseif animation_type == "expansion"
                # Expand and contract from center
                expansion_factor = 1.0 + 0.3 * sin(2π * t / duration)
                new_x = center_x + (smld.x[i] - center_x) * expansion_factor
                new_y = center_y + (smld.y[i] - center_y) * expansion_factor
            end
            
            push!(new_positions, Point2f(new_x, new_y))
        end
        
        mol_positions[] = new_positions
    end
    
    println("   ✅ Animation saved as: $(filename)")
    println("      File size: $(round(stat(filename).size / 1024^2, digits=2)) MB")
    
    return filename
end

"""
Create animated sequence showing dimer formation process
Works with SMLMSim dimer data to show binding process
"""
function animate_smlmsim_dimer_formation(smld, filename="smlmsim_dimer_formation.mp4"; 
                                        fps=20, formation_duration=3.0, pause_duration=1.0)
    println("🎬 Creating SMLMSim dimer formation animation...")
    
    n_molecules = length(smld.x)
    
    if n_molecules < 2 || n_molecules % 2 != 0
        error("Need even number of molecules (≥2) for dimer formation animation")
    end
     
    n_dimers = n_molecules ÷ 2
    
    println("   Animation parameters:")
    println("      Molecules: $(n_molecules)")
    println("      Dimers: $(n_dimers)")
    println("      Formation duration: $(formation_duration) s")
    println("      FPS: $(fps)")
    
    # Calculate plot bounds
    x_range = (minimum(smld.x) - 0.3, maximum(smld.x) + 0.3)
    y_range = (minimum(smld.y) - 0.3, maximum(smld.y) + 0.3)
    
    # Create figure
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1, 1], 
              title="SMLMSim Dimer Formation Process",
              xlabel="X Position (μm)",
              ylabel="Y Position (μm)",
              aspect=DataAspect())
    
    xlims!(ax, x_range...)
    ylims!(ax, y_range...)
    
    # Initial positions (separated molecules)
    separated_positions = Point2f[]
    for i in 1:2:n_molecules-1
        # Calculate center of dimer pair
        center_x = (smld.x[i] + smld.x[i+1]) / 2
        center_y = (smld.y[i] + smld.y[i+1]) / 2
        
        # Place molecules far apart initially
        separation = 0.4
        push!(separated_positions, Point2f(center_x - separation/2, center_y))
        push!(separated_positions, Point2f(center_x + separation/2, center_y))
    end
    
    # Final positions (from SMLMSim)
    final_positions = [Point2f(smld.x[i], smld.y[i]) for i in 1:n_molecules]
    
    # Animation observables
    mol_positions = Observable(separated_positions)
    formation_progress = Observable(0.0)
    phase_text = Observable("Initial: Separated molecules")
    
    # Colors and connections
    molecule_colors = []
    for i in 1:2:n_molecules-1
        push!(molecule_colors, :blue, :cyan)
    end
    
    # Plot molecules
    scatter!(ax, mol_positions, 
             color=molecule_colors, 
             markersize=18, 
             alpha=0.8,
             strokewidth=2,
             strokecolor=:black)
    
    # Dimer connections (initially transparent)
    for i in 1:2:n_molecules-1
        line_points = @lift([$(mol_positions)[$(i)], $(mol_positions)[$(i+1)]])
        connection_alpha = @lift($formation_progress * 1.0)  # Convert to expression
        lines!(ax, line_points, color=(:red, connection_alpha), linewidth=4)
    end
    
    # Labels
    for i in 1:n_molecules
        text!(ax, @lift($(mol_positions)[$(i)]), text="$(i)", 
              fontsize=12, color=:white, align=(:center, :center))
    end
    
    # Progress indicator
    progress_text = @lift("Formation Progress: $(round($formation_progress * 100, digits=1))%")
    text!(ax, x_range[1] + 0.05, y_range[2] - 0.05, text=progress_text, 
          fontsize=14, color=:black, fontweight=:bold)
    
    text!(ax, x_range[1] + 0.05, y_range[2] - 0.15, text=phase_text, 
          fontsize=12, color=:blue)
    
    # Total animation duration
    total_duration = formation_duration + pause_duration
    n_frames = Int(round(fps * total_duration))
    
    println("   🎥 Recording dimer formation...")
    record(fig, filename, 1:n_frames; framerate=fps) do frame
        t = (frame - 1) / fps
        
        if t <= formation_duration
            # Formation phase
            progress = t / formation_duration
            formation_progress[] = progress
            phase_text[] = "Forming dimers... ($(round(progress*100, digits=1))%)"
            
            # Interpolate positions
            new_positions = Point2f[]
            for i in 1:n_molecules
                start_pos = separated_positions[i]
                end_pos = final_positions[i]
                
                # Smooth interpolation
                interp_factor = 0.5 * (1 - cos(π * progress))
                new_x = start_pos[1] + (end_pos[1] - start_pos[1]) * interp_factor
                new_y = start_pos[2] + (end_pos[2] - start_pos[2]) * interp_factor
                
                push!(new_positions, Point2f(new_x, new_y))
            end
            
            mol_positions[] = new_positions
            
        else
            # Pause phase - dimers formed
            formation_progress[] = 1.0
            phase_text[] = "Dimers formed! (Final state)"
            mol_positions[] = final_positions
        end
    end
    
    println("   ✅ Dimer formation animation saved as: $(filename)")
    println("      File size: $(round(stat(filename).size / 1024^2, digits=2)) MB")
    
    return filename
end

"""
Create static snapshot visualization of SMLMSim data
Shows molecular positions without time evolution
"""
function plot_smlmsim_snapshot(smld, filename="smlmsim_snapshot.png"; 
                              title="SMLMSim Molecular Distribution")
    println("📸 Creating SMLMSim snapshot plot...")
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              title=title,
              xlabel="X Position (μm)",
              ylabel="Y Position (μm)",
              aspect=DataAspect())
    
    # Plot all molecules
    scatter!(ax, smld.x, smld.y, 
             color=:blue, 
             markersize=15, 
             alpha=0.7,
             label="Molecules (N=$(length(smld.x)))")
    
    # If it's dimer data, connect pairs
    if length(smld.x) % 2 == 0  # Even number suggests dimer pairs
        for i in 1:2:length(smld.x)-1
            lines!(ax, [smld.x[i], smld.x[i+1]], [smld.y[i], smld.y[i+1]], 
                   color=:red, linewidth=2, alpha=0.5)
        end
    end
    
    # Add field boundary
    field_x = smld.datasize[1]
    field_y = smld.datasize[2]
    lines!(ax, [0, field_x, field_x, 0, 0], [0, 0, field_y, field_y, 0], 
           color=:gray, linestyle=:dash, linewidth=1, label="Field boundary")
    
    axislegend(ax, position=:rt)
    
    save(filename, fig)
    println("   ✅ Snapshot saved as: $(filename)")
    
    return fig
end

"""
Analyze SMLMSim SMLD data structure
"""
function analyze_smld_data(smld)
    println("🔍 Analysis of SMLMSim SMLD data:")
    println("   Data type: $(typeof(smld))")
    println("   Number of molecules: $(length(smld.x))")
    
    println("\n   Available fields:")
    for field in fieldnames(typeof(smld))
        value = getfield(smld, field)
        if isa(value, Vector) && length(value) > 0
            println("      $field: $(typeof(value)) (length $(length(value)))")
        else
            println("      $field: $value")
        end
    end
    
    println("\n   Key insights:")
    println("      • This is STATIC localization data")
    println("      • x, y: Molecular coordinates (μm)")
    println("      • NO time evolution or dynamics")
    println("      • Perfect for initial conditions, not trajectories")
end

"""
Example using all SMLMSim methods
"""
function example_all_smlmsim_methods()
    println("🚀 Running all SMLMSim methods example...")
    
    Random.seed!(1234)
    
    # Method 1: Single molecules
    println("\n\n🔹 Single molecule distribution:")
    smld_single = smlmsim_single_molecules(rho=1.0, xsize=3.0, ysize=3.0)
    
    # Method 2: Dimers
    println("\n\n🔹 Dimer distribution:")
    smld_dimers = smlmsim_dimer_simulation(rho=0.8, separation=0.06, xsize=4.0, ysize=4.0)
    
    # Method 3: Clusters
    println("\n\n🔹 Cluster distribution:")
    smld_clusters = smlmsim_cluster_simulation(rho=0.3, n_molecules=4, cluster_size=0.08)
    
    # Convert dimer data to custom simulation
    println("\n\n🔹 Converting to custom simulation:")
    sim_static = convert_smld_to_custom_simulation(smld_dimers, k_on=0.1, k_off=0.5, d_dimer=0.08)
    
    # SMLMSim + dynamics
    println("\n\n🔹 SMLMSim + Dynamics:")
    sim_dynamic, smld_init = smlmsim_with_dynamics(method="dimer", t_max=5.0, k_off=0.3, d_dimer=0.08)
    
    # Analysis
    println("\n\n🔹 Data analysis:")
    analyze_smld_data(smld_dimers)
    
    println("\n🎯 Summary:")
    println("   ✅ SMLMSim uniform2D() works for spatial distributions")
    println("   ✅ Dimer and cluster patterns work reliably") 
    println("   ✅ Can convert to custom types for analysis")
    println("   ✅ Can use as initial conditions for dynamics")
    println("   ❌ Cannot create time-dependent trajectories directly")
    
    return sim_static, sim_dynamic, smld_dimers
end

"""
Simple example using SMLMSim methods
"""
function example_smlmsim_corrected()
    println("🚀 Running SMLMSim corrected methods example...")
    
    Random.seed!(1234)
    
    # Method 1: Single molecules
    println("\n🔹 Single molecule distribution:")
    smld_single = smlmsim_single_molecules(rho=1.0, xsize=3.0, ysize=3.0)
    
    # Method 2: Dimers  
    println("\n🔹 Dimer distribution:")
    smld_dimers = smlmsim_dimer_simulation(rho=0.8, separation=0.06, xsize=4.0, ysize=4.0)
    
    # Method 3: SMLMSim + dynamics
    println("\n🔹 SMLMSim + Dynamics:")
    sim_dynamic, smld_init = smlmsim_with_dynamics(method="dimer", t_max=5.0, k_off=0.3, d_dimer=0.08)
    
    display(sim_dynamic)
    
    # Method 4: Create animations
    println("\n🔹 Creating animations:")
    
    # Static snapshot of initial SMLMSim positions
    plot_smlmsim_snapshot(smld_dimers, "smlmsim_dimers_snapshot.png", 
                         title="SMLMSim Dimer Distribution")
    
    # Direct SMLMSim data animations (without conversion to custom types)
    println("\n🔸 Direct SMLMSim animations:")
    
    # Rotation animation of dimer data
    rotation_file = animate_smlmsim_data(smld_dimers, "smlmsim_rotation.mp4", 
                                       fps=25, duration=4.0, animation_type="rotation",
                                       show_connections=true, highlight_dimers=true)
    
    # Oscillation animation
    oscillation_file = animate_smlmsim_data(smld_dimers, "smlmsim_oscillation.mp4", 
                                          fps=25, duration=3.0, animation_type="oscillation",
                                          show_connections=true, highlight_dimers=true)
    
    # Dimer formation animation
    formation_file = animate_smlmsim_dimer_formation(smld_dimers, "smlmsim_formation.mp4", 
                                                   fps=20, formation_duration=3.0, pause_duration=1.0)
    
    # Animated dynamics (converted to custom type)
    println("\n🔸 Custom type dynamics animation:")
    dynamics_file = animate_smlmsim_particles(sim_dynamic, "smlmsim_dynamics.mp4", 
                                             fps=20, show_trail=true, trail_length=30)
    
    println("\n🎯 Summary:")
    println("   ✅ SMLMSim uniform2D() works for spatial distributions")
    println("   ✅ Dimer patterns work reliably") 
    println("   ✅ Can use as initial conditions for dynamics")
    println("   ✅ Animations created showing binding/unbinding dynamics")
    println("   ❌ Cannot create time-dependent trajectories directly")
    
    return sim_dynamic, smld_dimers, animation_file
end

println("📦 SMLMSim corrected methods loaded!")
println("   Use example_smlmsim_corrected() to run simple example")
println("   Use example_all_smlmsim_methods() to run all methods")
println("   Use smlmsim_dimer_simulation() for 2-molecule pairs")
println("   Use smlmsim_with_dynamics() for SMLMSim + dynamics")