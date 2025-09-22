

function simulate_diffusion(;initial_positions = nothing,D = 0.01,box = 1.0,dt = 0.016,steps = 2000)
    # Initialize positions
    if initial_positions isa Tuple && length(initial_positions) == 2
        p1_pos = copy(initial_positions[1])
        p2_pos = copy(initial_positions[2])
    else
        p1_pos = rand(2) .* box
        p2_pos = rand(2) .* box
    end
    
    # Initialize velocities
    p1_vel = randn(2) .* sqrt(2D/dt)
    p2_vel = randn(2) .* sqrt(2D/dt)
    
    # Arrays to store positions
    p1_positions = zeros(2, steps)
    p2_positions = zeros(2, steps)
    p1_positions[:, 1] = p1_pos
    p2_positions[:, 1] = p2_pos
    
    
    # Run simulation without animation
    for frame in 2:steps
        # Update velocities and positions
        p1_vel .= randn(2) .* sqrt(2D/dt)
        p2_vel .= randn(2) .* sqrt(2D/dt)
        
        p1_pos .+= p1_vel .* dt
        p2_pos .+= p2_vel .* dt
        
        # Enforce boundaries
        for i in 1:2
            p1_pos[i] = clamp(p1_pos[i], 0, box)
            p2_pos[i] = clamp(p2_pos[i], 0, box)
        end
        
        # Store positions
        p1_positions[:, frame] = p1_pos
        p2_positions[:, frame] = p2_pos
    end

    
    return p1_positions, p2_positions
end

# Custom type integration - Include these when working with simulation types
# include("types/types_with_data.jl")

# Function to create free diffusion simulation using custom types
function free_diffusion_simulation(; initial_positions = nothing, k_on = 0.5, k_off = 0.3,
                                  D = 0.01, box = 1.0, dt = 0.016, steps = 2000, σ = 0.0)
    
    # Generate free diffusion trajectories
    p1_matrix, p2_matrix = simulate_diffusion(initial_positions=initial_positions, D=D, box=box, dt=dt, steps=steps)
    
    # Create dummy states (all free state since it's free diffusion)
    states = fill(1, steps-1)  # State 1 = free state
    
    # Create simulation object
    k_states = [k_on, k_off]
    sim = simulation(p1_matrix, p2_matrix, states, k_states, D, dt=dt, σ=σ)
    
    return sim
end

# Function to extract free segments from existing simulation
function extract_free_segments(sim::simulation, threshold_distance = nothing)
    # Use sim.d_dimer if threshold_distance not provided
    threshold = isnothing(threshold_distance) ? sim.d_dimer : threshold_distance
    
    n_steps = length(sim.particle_1)
    free_segments = []
    current_segment_start = nothing
    
    for i in 1:n_steps
        dist = sqrt((sim.particle_1[i].x - sim.particle_2[i].x)^2 + 
                   (sim.particle_1[i].y - sim.particle_2[i].y)^2)
        
        if dist > threshold  # Particles are free
            if isnothing(current_segment_start)
                current_segment_start = i
            end
        else  # Particles are bound
            if !isnothing(current_segment_start)
                # End of free segment
                push!(free_segments, (start=current_segment_start, stop=i-1))
                current_segment_start = nothing
            end
        end
    end
    
    # Handle case where simulation ends while free
    if !isnothing(current_segment_start)
        push!(free_segments, (start=current_segment_start, stop=n_steps))
    end
    
    return free_segments
end

# Function to analyze free diffusion properties from simulation
function analyze_free_diffusion_properties(sim::simulation)
    free_segments = extract_free_segments(sim)
    
    # Calculate average free duration
    free_durations = []
    for segment in free_segments
        duration = sim.particle_1[segment.stop].t - sim.particle_1[segment.start].t
        push!(free_durations, duration)
    end
    
    avg_free_time = isempty(free_durations) ? 0.0 : mean(free_durations)
    
    # Calculate average inter-particle distance during free periods
    avg_distances = []
    for segment in free_segments
        segment_distances = []
        for i in segment.start:segment.stop
            dist = sqrt((sim.particle_1[i].x - sim.particle_2[i].x)^2 + 
                       (sim.particle_1[i].y - sim.particle_2[i].y)^2)
            push!(segment_distances, dist)
        end
        if !isempty(segment_distances)
            push!(avg_distances, mean(segment_distances))
        end
    end
    
    overall_avg_distance = isempty(avg_distances) ? 0.0 : mean(avg_distances)
    
    # Calculate diffusion coefficient from mean squared displacement
    diffusion_coefs = []
    for segment in free_segments
        if segment.stop - segment.start > 10  # Need sufficient points
            msd_values = []
            for lag in 1:min(50, segment.stop - segment.start)
                displacements_1 = []
                displacements_2 = []
                
                for i in segment.start:(segment.stop-lag)
                    dx1 = sim.particle_1[i+lag].x - sim.particle_1[i].x
                    dy1 = sim.particle_1[i+lag].y - sim.particle_1[i].y
                    dx2 = sim.particle_2[i+lag].x - sim.particle_2[i].x
                    dy2 = sim.particle_2[i+lag].y - sim.particle_2[i].y
                    
                    push!(displacements_1, dx1^2 + dy1^2)
                    push!(displacements_2, dx2^2 + dy2^2)
                end
                
                msd = mean(vcat(displacements_1, displacements_2))
                push!(msd_values, msd)
            end
            
            # Linear fit to get diffusion coefficient (MSD = 4*D*t)
            if length(msd_values) > 5
                time_lags = sim.dt * (1:length(msd_values))
                # Simple linear regression slope
                slope = sum((time_lags .- mean(time_lags)) .* (msd_values .- mean(msd_values))) / 
                       sum((time_lags .- mean(time_lags)).^2)
                estimated_D = slope / 4  # MSD = 4*D*t for 2D
                push!(diffusion_coefs, estimated_D)
            end
        end
    end
    
    avg_diffusion_coef = isempty(diffusion_coefs) ? sim.D : mean(diffusion_coefs)
    
    return (
        free_segments = free_segments,
        free_durations = free_durations,
        avg_free_time = avg_free_time,
        avg_distance = overall_avg_distance,
        estimated_diffusion_coef = avg_diffusion_coef,
        num_free_events = length(free_segments)
    )
end

# Function to create a new simulation with only free segments
function create_free_only_simulation(sim::simulation, segment_index::Int = 1)
    free_segments = extract_free_segments(sim)
    
    if segment_index > length(free_segments)
        error("Segment index $segment_index exceeds number of free segments ($(length(free_segments)))")
    end
    
    segment = free_segments[segment_index]
    
    # Extract particles for this segment
    segment_particles_1 = sim.particle_1[segment.start:segment.stop]
    segment_particles_2 = sim.particle_2[segment.start:segment.stop]
    
    # Create new simulation with only this free segment
    states_segment = fill(1, length(segment_particles_1)-1)  # All free state
    
    return simulation(segment_particles_1, segment_particles_2, 
                     sim.k_states, sim.σ, sim.D, sim.dt, sim.d_dimer)
end

# Function to calculate mean squared displacement for the entire simulation
function calculate_msd(sim::simulation, max_lag::Int = 100)
    n_steps = length(sim.particle_1)
    max_lag = min(max_lag, n_steps - 1)
    
    msd_values = Float64[]
    time_lags = Float64[]
    
    for lag in 1:max_lag
        displacements_1 = []
        displacements_2 = []
        
        for i in 1:(n_steps-lag)
            dx1 = sim.particle_1[i+lag].x - sim.particle_1[i].x
            dy1 = sim.particle_1[i+lag].y - sim.particle_1[i].y
            dx2 = sim.particle_2[i+lag].x - sim.particle_2[i].x
            dy2 = sim.particle_2[i+lag].y - sim.particle_2[i].y
            
            push!(displacements_1, dx1^2 + dy1^2)
            push!(displacements_2, dx2^2 + dy2^2)
        end
        
        msd = mean(vcat(displacements_1, displacements_2))
        push!(msd_values, msd)
        push!(time_lags, lag * sim.dt)
    end
    
    return time_lags, msd_values
end
