


function constrained_diffusion(; initial_p1 = nothing,D = 0.01,r = 0.2,box = 1.0, dt = 0.016,steps = 500)
    # Initialize particle 1 position
    if isnothing(initial_p1)
        p1_pos = rand(2) .* (box - 2r) .+ r
    else
        p1_pos = copy(initial_p1)
        for i in 1:2
            p1_pos[i] = clamp(p1_pos[i], r, box - r)
        end
    end
    
    # Initial angle and position for particle 2
    initial_angle = 2π * rand()
    p2_pos = p1_pos + r .* [cos(initial_angle), sin(initial_angle)]
    
    # Velocity for particle 1
    p1_vel = randn(2) .* sqrt(2D/dt)
    
    # Arrays to store positions
    p1_positions = zeros(2, steps)
    p2_positions = zeros(2, steps)
    p1_positions[:, 1] = p1_pos
    p2_positions[:, 1] = p2_pos
    
    
    # Run simulation without animation
    for frame in 2:steps
        p1_vel .= randn(2) .* sqrt(2D/dt)
        p1_pos .+= p1_vel .* dt
        
        for i in 1:2
            p1_pos[i] = clamp(p1_pos[i], 0, box)
        end
        
        new_angle = 2π * rand()
        p2_pos .= p1_pos + r .* [cos(new_angle), sin(new_angle)]
        
        for i in 1:2
            if p2_pos[i] <= 0 || p2_pos[i] >= box
                retry_count = 0
                while (p2_pos[i] <= 0 || p2_pos[i] >= box) && retry_count < 10
                    new_angle = 2π * rand()
                    p2_pos .= p1_pos + r .* [cos(new_angle), sin(new_angle)]
                    retry_count += 1
                end
                
                if p2_pos[i] <= 0 || p2_pos[i] >= box
                    if p1_pos[i] < box/2
                        p1_pos[i] = r + 0.05
                    else
                        p1_pos[i] = box - r - 0.05
                    end
                    p2_pos .= p1_pos + r .* [cos(new_angle), sin(new_angle)]
                end
            end
        end
        
        p1_positions[:, frame] = p1_pos
        p2_positions[:, frame] = p2_pos
    end
    
    
    return p1_positions, p2_positions
end

# Custom type integration - Include these when working with simulation types
# include("types/types_with_data.jl")

# Function to create constrained diffusion simulation using custom types
function constrained_diffusion_simulation(; initial_p1 = nothing, k_on = 0.5, k_off = 0.3,
                                        D = 0.01, r = 0.2, box = 1.0, dt = 0.016, steps = 500, σ = 0.0)
    
    # Generate constrained diffusion trajectories
    p1_matrix, p2_matrix = constrained_diffusion(initial_p1=initial_p1, D=D, r=r, box=box, dt=dt, steps=steps)
    
    # Create dummy states (all bound state since it's constrained diffusion)
    states = fill(2, steps-1)  # State 2 = bound state
    
    # Create simulation object
    k_states = [k_on, k_off]
    sim = simulation(p1_matrix, p2_matrix, states, k_states, D, dt=dt, σ=σ)
    
    return sim
end

# Function to extract constrained segments from existing simulation
function extract_bound_segments(sim::simulation, threshold_distance = nothing)
    # Use sim.d_dimer if threshold_distance not provided
    threshold = isnothing(threshold_distance) ? sim.d_dimer : threshold_distance
    
    n_steps = length(sim.particle_1)
    bound_segments = []
    current_segment_start = nothing
    
    for i in 1:n_steps
        dist = sqrt((sim.particle_1[i].x - sim.particle_2[i].x)^2 + 
                   (sim.particle_1[i].y - sim.particle_2[i].y)^2)
        
        if dist <= threshold  # Particles are bound
            if isnothing(current_segment_start)
                current_segment_start = i
            end
        else  # Particles are free
            if !isnothing(current_segment_start)
                # End of bound segment
                push!(bound_segments, (start=current_segment_start, stop=i-1))
                current_segment_start = nothing
            end
        end
    end
    
    # Handle case where simulation ends while bound
    if !isnothing(current_segment_start)
        push!(bound_segments, (start=current_segment_start, stop=n_steps))
    end
    
    return bound_segments
end

# Function to analyze dimer properties from simulation
function analyze_dimer_properties(sim::simulation)
    bound_segments = extract_bound_segments(sim)
    
    # Calculate average binding duration
    binding_durations = []
    for segment in bound_segments
        duration = sim.particle_1[segment.stop].t - sim.particle_1[segment.start].t
        push!(binding_durations, duration)
    end
    
    avg_binding_time = isempty(binding_durations) ? 0.0 : mean(binding_durations)
    
    # Calculate binding frequency
    total_time = sim.particle_1[end].t - sim.particle_1[1].t
    binding_frequency = length(bound_segments) / total_time
    
    return (
        bound_segments = bound_segments,
        binding_durations = binding_durations,
        avg_binding_time = avg_binding_time,
        binding_frequency = binding_frequency,
        num_binding_events = length(bound_segments)
    )
end

# Function to create a new simulation with only bound segments
function create_bound_only_simulation(sim::simulation, segment_index::Int = 1)
    bound_segments = extract_bound_segments(sim)
    
    if segment_index > length(bound_segments)
        error("Segment index $segment_index exceeds number of bound segments ($(length(bound_segments)))")
    end
    
    segment = bound_segments[segment_index]
    
    # Extract particles for this segment
    segment_particles_1 = sim.particle_1[segment.start:segment.stop]
    segment_particles_2 = sim.particle_2[segment.start:segment.stop]
    
    # Create new simulation with only this bound segment
    states_segment = fill(2, length(segment_particles_1)-1)  # All bound state
    
    return simulation(segment_particles_1, segment_particles_2, 
                     sim.k_states, sim.σ, sim.D, sim.dt, sim.d_dimer)
end
