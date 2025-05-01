using Distributions
using GLMakie
using Random
using Optim

# Define Particle type
struct Particle
    x::Float64
    y::Float64
    state::Int
end

# Define Trajectory type
struct Trajectory
    positions::Vector{Particle}
    times::Vector{Float64}
end

# Constructors for Trajectory
Trajectory() = Trajectory(Vector{Particle}(), Vector{Float64}())
function add_position!(trajectory::Trajectory, particle::Particle, time::Float64)
    push!(trajectory.positions, particle)
    push!(trajectory.times, time)
    return trajectory
end

#==============================================================================
# State Generation Functions
==============================================================================#
function simulate_states(transition_rate_12::Float64, transition_rate_21::Float64,  desired_state_changes::Int=10, time_step::Float64=0.1;  max_steps::Int=100000)
    probability_12 = transition_rate_12 * time_step
    probability_21 = transition_rate_21 * time_step
    
    states = Int[]
    current_state = 1
    push!(states, current_state)
    
    changes_count = 0
    steps = 0
    
    while changes_count < desired_state_changes && steps < max_steps
        steps += 1
        random_value = rand()
        
        next_state = current_state
        if current_state == 1
            if random_value < probability_12
                next_state = 2
            end
        else
            if random_value < probability_21
                next_state = 1
            end
        end
        
        if next_state != current_state
            changes_count += 1
        end
        
        current_state = next_state
        push!(states, current_state)
    end
    
    return states, steps
end

# Additional state function that returns time information
function simulate_states(transition_rate_12::Float64, transition_rate_21::Float64, 
                        desired_state_changes::Int, time_step::Float64, 
                        return_times::Bool; max_steps::Int=100000)
    if !return_times
        return simulate_states(transition_rate_12, transition_rate_21, desired_state_changes, time_step, max_steps=max_steps)
    end
    
    probability_12 = transition_rate_12 * time_step
    probability_21 = transition_rate_21 * time_step
    
    states = Int[]
    times = Float64[]
    current_state = 1
    current_time = 0.0
    
    push!(states, current_state)
    push!(times, current_time)
    
    changes_count = 0
    steps = 0
    
    while changes_count < desired_state_changes && steps < max_steps
        steps += 1
        current_time += time_step
        random_value = rand()
        
        next_state = current_state
        if current_state == 1
            if random_value < probability_12
                next_state = 2
                changes_count += 1
            end
        else
            if random_value < probability_21
                next_state = 1
                changes_count += 1
            end
        end
        
        current_state = next_state
        push!(states, current_state)
        push!(times, current_time)
    end
    
    return states, times, steps
end

#==============================================================================
# State Transition Analysis Functions
==============================================================================#
function get_state_transitions(states::Vector{Int})
    transitions = []
    for i in 2:length(states)
        if states[i] != states[i - 1]
            time_step = (i - 1) 
            from_state = states[i - 1]
            to_state = states[i]
            push!(transitions, (time_step, from_state, to_state))
        end
    end
    return transitions
end

# Overloaded method for state transitions with time information
function get_state_transitions(states::Vector{Int}, times::Vector{Float64})
    transitions = []
    for i in 2:length(states)
        if states[i] != states[i-1]
            transition_time = times[i]
            from_state = states[i-1]
            to_state = states[i]
            push!(transitions, (transition_time, from_state, to_state))
        end
    end
    return transitions
end

function time_sequence(states::Vector{Int})
    if isempty(states)
        return []
    end
    
    state_time_sequence = []
    current_state = states[1]
    current_count = 1
    
    for i in 2:length(states)
        if states[i] == current_state
            current_count += 1
        else
            push!(state_time_sequence, current_state => current_count)
            current_state = states[i]
            current_count = 1
        end
    end
    
    push!(state_time_sequence, current_state => current_count)
    
    return state_time_sequence
end

function time_sequence_with_split(states::Vector{Int})
    if isempty(states)
        return []
    end
    
    durations = []
    current_state = states[1]
    current_count = 1
    
    for i in 2:length(states)
        if states[i] == current_state
            current_count += 1
        else
            push!(durations, current_state => current_count)
            current_state = states[i]
            current_count = 1
        end
    end
    push!(durations, current_state => current_count)
    
    sections = []
    current_section = []
    
    if !isempty(durations)
        push!(current_section, durations[1])
    end
    
    i = 2
    while i <= length(durations)
        state, duration = durations[i].first, durations[i].second
        
        if state == 1 && duration > 1
            half = div(duration, 2)
            push!(current_section, 1 => half)
            
            if length(current_section) == 3
                push!(sections, current_section)
                current_section = []
            end
            
            push!(current_section, 1 => (duration - half))
        else
            push!(current_section, state => duration)
        end
        
        if length(current_section) == 3
            push!(sections, current_section)
            current_section = []
        end
        
        i += 1
    end
    
    if !isempty(current_section)
        push!(sections, current_section)
    end
    
    return sections
end

#==============================================================================
# Utility Functions for Matrix & Particle Conversion
==============================================================================#
function reverse_columns_preserve_size(matrix::Matrix)
    num_rows, num_cols = size(matrix)
    result = zeros(eltype(matrix), num_rows, num_cols)
    
    for col in 1:num_cols
        result[:, col] = matrix[:, num_cols - col + 1]
    end
    
    return result
end

function shift(vector::Vector, shift_x::Float64, shift_y::Float64)
    return vector .+ [shift_x, shift_y]
end

function shift(matrix::Matrix{Float64}, shift_x::Float64, shift_y::Float64)
    result = copy(matrix)
    result[1, :] .+= shift_x
    result[2, :] .+= shift_y
    return result
end

# Convert 2D matrix representation to Particle arrays
function convert_to_particles(p1_positions::Matrix{Float64}, p2_positions::Matrix{Float64}, states::Vector{Int})
    n_steps = min(size(p1_positions, 2), size(p2_positions, 2))
    
    # Default state if not enough states are provided
    default_state = isempty(states) ? 1 : states[end]
    
    # Create particles
    p1_particles = Vector{Particle}(undef, n_steps)
    p2_particles = Vector{Particle}(undef, n_steps)
    
    for i in 1:n_steps
        # Get state (use the last state if we run out)
        current_state = i <= length(states) ? states[i] : default_state
        
        # Create particles
        p1_particles[i] = Particle(p1_positions[1, i], p1_positions[2, i], current_state)
        p2_particles[i] = Particle(p2_positions[1, i], p2_positions[2, i], current_state)
    end
    
    return p1_particles, p2_particles
end

# Convert Particle arrays to 2D matrix representation
function convert_from_particles(p1_particles::Vector{Particle}, p2_particles::Vector{Particle})
    n_steps = min(length(p1_particles), length(p2_particles))
    
    # Create matrices
    p1_positions = zeros(2, n_steps)
    p2_positions = zeros(2, n_steps)
    states = zeros(Int, n_steps)
    
    for i in 1:n_steps
        p1_positions[1, i] = p1_particles[i].x
        p1_positions[2, i] = p1_particles[i].y
        p2_positions[1, i] = p2_particles[i].x
        p2_positions[2, i] = p2_particles[i].y
        states[i] = p1_particles[i].state
    end
    
    return p1_positions, p2_positions, states
end

function path_correction(particle_path::Matrix{Float64}, new_destination::Vector{Float64})
    if size(particle_path, 1) != 2
        error("Particle path must be a 2×n matrix")
    end
    
    if length(new_destination) != 2
        error("New destination must be a 2-element vector")
    end
    
    n = size(particle_path, 2)
    
    start_point = particle_path[:, 1]
    original_end = particle_path[:, end]
    
    original_displacement = original_end - start_point
    desired_displacement = new_destination - start_point
    
    new_path = copy(particle_path)
    
    for i in 2:n
        t = (i - 1) / (n - 1)
        
        expected_position = start_point + original_displacement * t
        diffusion_component = particle_path[:, i] - expected_position
        new_expected_position = start_point + desired_displacement * t
        
        new_path[:, i] = new_expected_position + diffusion_component
    end
    
    return new_path
end

function path_correction!(particle_path::Matrix{Float64}, new_destination::Vector{Float64})
    if size(particle_path, 1) != 2
        error("Particle path must be a 2×n matrix")
    end
    
    if length(new_destination) != 2
        error("New destination must be a 2-element vector")
    end
    
    n = size(particle_path, 2)
    
    start_point = particle_path[:, 1]
    original_end = particle_path[:, end]
    
    original_displacement = original_end - start_point
    desired_displacement = new_destination - start_point
    
    for i in 2:n
        t = (i - 1) / (n - 1)
        
        expected_position = start_point + original_displacement * t
        diffusion_component = particle_path[:, i] - expected_position
        new_expected_position = start_point + desired_displacement * t
        
        particle_path[:, i] = new_expected_position + diffusion_component
    end
    
    return nothing
end

#==============================================================================
# Diffusion Simulation Functions - Matrix Version
==============================================================================#
function simulate_diffusion(;initial_positions=nothing, diffusion_coefficient::Float64=0.01, 
                           box_size::Float64=1.0, time_step::Float64=0.016, steps::Int=2000)
    if initial_positions isa Tuple && length(initial_positions) == 2
        particle1_position = copy(initial_positions[1])
        particle2_position = copy(initial_positions[2])
    else
        particle1_position = rand(2) .* box_size
        particle2_position = rand(2) .* box_size
    end
    
    particle1_velocity = randn(2) .* sqrt(2*diffusion_coefficient/time_step)
    particle2_velocity = randn(2) .* sqrt(2*diffusion_coefficient/time_step)
    
    particle1_positions = zeros(2, steps)
    particle2_positions = zeros(2, steps)
    particle1_positions[:, 1] = particle1_position
    particle2_positions[:, 1] = particle2_position
    
    for frame in 2:steps
        particle1_velocity .= randn(2) .* sqrt(2*diffusion_coefficient/time_step)
        particle2_velocity .= randn(2) .* sqrt(2*diffusion_coefficient/time_step)
        
        particle1_position .+= particle1_velocity .* time_step
        particle2_position .+= particle2_velocity .* time_step
        
        for i in 1:2
            particle1_position[i] = clamp(particle1_position[i], 0, box_size)
            particle2_position[i] = clamp(particle2_position[i], 0, box_size)
        end
        
        particle1_positions[:, frame] = particle1_position
        particle2_positions[:, frame] = particle2_position
    end
    
    return particle1_positions, particle2_positions
end

function constrained_diffusion(; initial_p1=nothing, diffusion_coefficient::Float64=0.01, 
                             radius::Float64=0.2, box_size::Float64=1.0, 
                             time_step::Float64=0.016, steps::Int=500)
    if isnothing(initial_p1)
        particle1_position = rand(2) .* (box_size - 2*radius) .+ radius
    else
        particle1_position = copy(initial_p1)
        for i in 1:2
            particle1_position[i] = clamp(particle1_position[i], radius, box_size - radius)
        end
    end
    
    initial_angle = 2π * rand()
    particle2_position = particle1_position + radius .* [cos(initial_angle), sin(initial_angle)]
    
    particle1_velocity = randn(2) .* sqrt(2*diffusion_coefficient/time_step)
    
    particle1_positions = zeros(2, steps)
    particle2_positions = zeros(2, steps)
    particle1_positions[:, 1] = particle1_position
    particle2_positions[:, 1] = particle2_position
    
    for frame in 2:steps
        particle1_velocity .= randn(2) .* sqrt(2*diffusion_coefficient/time_step)
        particle1_position .+= particle1_velocity .* time_step
        
        for i in 1:2
            particle1_position[i] = clamp(particle1_position[i], 0, box_size)
        end
        
        new_angle = 2π * rand()
        particle2_position .= particle1_position + radius .* [cos(new_angle), sin(new_angle)]
        
        for i in 1:2
            if particle2_position[i] <= 0 || particle2_position[i] >= box_size
                retry_count = 0
                while (particle2_position[i] <= 0 || particle2_position[i] >= box_size) && retry_count < 10
                    new_angle = 2π * rand()
                    particle2_position .= particle1_position + radius .* [cos(new_angle), sin(new_angle)]
                    retry_count += 1
                end
                
                if particle2_position[i] <= 0 || particle2_position[i] >= box_size
                    if particle1_position[i] < box_size/2
                        particle1_position[i] = radius + 0.05
                    else
                        particle1_position[i] = box_size - radius - 0.05
                    end
                    particle2_position .= particle1_position + radius .* [cos(new_angle), sin(new_angle)]
                end
            end
        end
        
        particle1_positions[:, frame] = particle1_position
        particle2_positions[:, frame] = particle2_position
    end
    
    return particle1_positions, particle2_positions
end

#==============================================================================
# Diffusion Simulation Functions - Particle Version
==============================================================================#
function simulate_diffusion(initial_particle1::Particle, initial_particle2::Particle;
                           diffusion_coefficient::Float64=0.01, box_size::Float64=1.0,
                           time_step::Float64=0.01, steps::Int=100)
    # Initialize trajectories
    particle1_trajectory = Trajectory()
    particle2_trajectory = Trajectory()
    
    add_position!(particle1_trajectory, initial_particle1, 0.0)
    add_position!(particle2_trajectory, initial_particle2, 0.0)
    
    current_p1 = initial_particle1
    current_p2 = initial_particle2
    current_time = 0.0
    
    for step in 1:steps
        current_time += time_step
        
        # Generate random displacements
        dx1 = randn() * sqrt(2 * diffusion_coefficient * time_step)
        dy1 = randn() * sqrt(2 * diffusion_coefficient * time_step)
        dx2 = randn() * sqrt(2 * diffusion_coefficient * time_step)
        dy2 = randn() * sqrt(2 * diffusion_coefficient * time_step)
        
        # Update positions
        new_x1 = clamp(current_p1.x + dx1, 0, box_size)
        new_y1 = clamp(current_p1.y + dy1, 0, box_size)
        new_x2 = clamp(current_p2.x + dx2, 0, box_size)
        new_y2 = clamp(current_p2.y + dy2, 0, box_size)
        
        # Create new particles with updated positions
        current_p1 = Particle(new_x1, new_y1, current_p1.state)
        current_p2 = Particle(new_x2, new_y2, current_p2.state)
        
        # Add to trajectories
        add_position!(particle1_trajectory, current_p1, current_time)
        add_position!(particle2_trajectory, current_p2, current_time)
    end
    
    return particle1_trajectory, particle2_trajectory
end

function constrained_diffusion(initial_particle1::Particle, initial_particle2::Particle;
                             diffusion_coefficient::Float64=0.01, bond_length::Float64=0.2,
                             box_size::Float64=1.0, time_step::Float64=0.01, steps::Int=100)
    # Initialize trajectories
    particle1_trajectory = Trajectory()
    particle2_trajectory = Trajectory()
    
    add_position!(particle1_trajectory, initial_particle1, 0.0)
    add_position!(particle2_trajectory, initial_particle2, 0.0)
    
    current_p1 = initial_particle1
    current_p2 = initial_particle2
    current_time = 0.0
    
    for step in 1:steps
        current_time += time_step
        
        # Generate random displacement for particle 1
        dx1 = randn() * sqrt(2 * diffusion_coefficient * time_step)
        dy1 = randn() * sqrt(2 * diffusion_coefficient * time_step)
        
        # Update position of particle 1
        new_x1 = clamp(current_p1.x + dx1, bond_length, box_size - bond_length)
        new_y1 = clamp(current_p1.y + dy1, bond_length, box_size - bond_length)
        
        # Generate random angle for particle 2 position
        angle = 2π * rand()
        
        # Calculate new position for particle 2 at fixed distance
        new_x2 = new_x1 + bond_length * cos(angle)
        new_y2 = new_y1 + bond_length * sin(angle)
        
        # Handle boundary conditions for particle 2
        retry_count = 0
        while ((new_x2 < 0 || new_x2 > box_size || new_y2 < 0 || new_y2 > box_size) && retry_count < 10)
            angle = 2π * rand()
            new_x2 = new_x1 + bond_length * cos(angle)
            new_y2 = new_y1 + bond_length * sin(angle)
            retry_count += 1
        end
        
        # If we still can't place particle 2 within boundaries, adjust particle 1
        if new_x2 < 0 || new_x2 > box_size || new_y2 < 0 || new_y2 > box_size
            if new_x1 < box_size/2
                new_x1 = bond_length + 0.05
            else
                new_x1 = box_size - bond_length - 0.05
            end
            
            if new_y1 < box_size/2
                new_y1 = bond_length + 0.05
            else
                new_y1 = box_size - bond_length - 0.05
            end
            
            angle = 2π * rand()
            new_x2 = new_x1 + bond_length * cos(angle)
            new_y2 = new_y1 + bond_length * sin(angle)
        end
        
        # Enforce boundary for safety
        new_x2 = clamp(new_x2, 0, box_size)
        new_y2 = clamp(new_y2, 0, box_size)
        
        # Create new particles with updated positions
        current_p1 = Particle(new_x1, new_y1, current_p1.state)
        current_p2 = Particle(new_x2, new_y2, current_p2.state)
        
        # Add to trajectories
        add_position!(particle1_trajectory, current_p1, current_time)
        add_position!(particle2_trajectory, current_p2, current_time)
    end
    
    return particle1_trajectory, particle2_trajectory
end

#==============================================================================
# Segment Functions
==============================================================================#
function modes_selection(section::Vector)
    size_section = length(section)
    if size_section == 3
        return :full
    elseif size_section == 2
        if section[1].first == 1 && section[2].first == 2
            return :left_half
        elseif section[1].first == 2 && section[2].first == 1
            return :right_half
        else
            error("Invalid section combination")
        end
    elseif size_section == 1
        return :middle
    else
        error("Invalid number of sections")
    end
end

function time_stamps(time_state::Vector)
    segments = []
    for i in 1:length(time_state)
        push!(segments, time_state[i].second)
    end
    return segments
end

# Matrix-based segments function
function segments(time_division::Vector, mode::Symbol; diffusion_coefficient::Float64=0.01, 
                 radius::Float64=0.2, box_size::Float64=1.0, time_step::Float64=0.016)
    if mode == :full && length(time_division) == 3
        particle1_middle, particle2_middle = constrained_diffusion(
            steps=time_division[2], diffusion_coefficient=diffusion_coefficient, 
            radius=radius, box_size=box_size, time_step=time_step)

        init_post_r = ([particle1_middle[1,end], particle1_middle[2,end]],
                      [particle2_middle[1,end], particle2_middle[2,end]])
        particle1_right, particle2_right = simulate_diffusion(
            initial_positions=init_post_r, steps=time_division[3], 
            diffusion_coefficient=diffusion_coefficient, box_size=box_size, time_step=time_step)

        init_post_l = ([particle1_middle[1,1], particle1_middle[2,1]], 
                      [particle2_middle[1,1], particle2_middle[2,1]])
        particle1_left, particle2_left = simulate_diffusion(
            initial_positions=init_post_l, steps=time_division[1], 
            diffusion_coefficient=diffusion_coefficient, box_size=box_size, time_step=time_step)
        particle1_left_reversed = reverse_columns_preserve_size(particle1_left)
        particle2_left_reversed = reverse_columns_preserve_size(particle2_left)

        particle1 = hcat(particle1_left_reversed, particle1_middle, particle1_right)
        particle2 = hcat(particle2_left_reversed, particle2_middle, particle2_right)

        return particle1, particle2

    elseif mode == :right_half && length(time_division) == 2
        particle1_middle, particle2_middle = constrained_diffusion(
            steps=time_division[1], diffusion_coefficient=diffusion_coefficient, 
            radius=radius, box_size=box_size, time_step=time_step)

        init_post_r = ([particle1_middle[1,end], particle1_middle[2,end]],  
                      [particle2_middle[1,end], particle2_middle[2,end]])
        particle1_right, particle2_right = simulate_diffusion(
            initial_positions=init_post_r, steps=time_division[2],  
            diffusion_coefficient=diffusion_coefficient, box_size=box_size, time_step=time_step)

        particle1 = hcat(particle1_middle, particle1_right)
        particle2 = hcat(particle2_middle, particle2_right)

        return particle1, particle2

    elseif mode == :middle && length(time_division) == 1
        particle1_middle, particle2_middle = constrained_diffusion(
            steps=time_division[1], diffusion_coefficient=diffusion_coefficient, 
            radius=radius, box_size=box_size, time_step=time_step)
        particle1 = hcat(particle1_middle)
        particle2 = hcat(particle2_middle)

        return particle1, particle2

    elseif mode == :left_half && length(time_division) == 2
        particle1_middle, particle2_middle = constrained_diffusion(
            steps=time_division[2], diffusion_coefficient=diffusion_coefficient, 
            radius=radius, box_size=box_size, time_step=time_step)

        init_post_l = ([particle1_middle[1,1], particle1_middle[2,1]],  
                     [particle2_middle[1,1], particle2_middle[2,1]])
        particle1_left, particle2_left = simulate_diffusion(
            initial_positions=init_post_l, steps=time_division[1], 
            diffusion_coefficient=diffusion_coefficient, box_size=box_size, time_step=time_step)
        particle1_left_reversed = reverse_columns_preserve_size(particle1_left)
        particle2_left_reversed = reverse_columns_preserve_size(particle2_left)

        particle1 = hcat(particle1_left_reversed, particle1_middle)
        particle2 = hcat(particle2_left_reversed, particle2_middle)
        return particle1, particle2

    else
        error("Invalid mode or time division")
    end
end

# Particle-based segments function
function segments(time_division::Vector, mode::Symbol, initial_p1::Particle, initial_p2::Particle;
                diffusion_coefficient::Float64=0.01, bond_length::Float64=0.2, 
                box_size::Float64=1.0, time_step::Float64=0.01)
    if mode == :full && length(time_division) == 3
        # Generate middle segment (constrained/dimer diffusion)
        p1_middle = Particle(initial_p1.x, initial_p1.y, 2)  # State 2 for dimer
        p2_middle = Particle(initial_p2.x, initial_p2.y, 2)
        p1_middle_traj, p2_middle_traj = constrained_diffusion(
            p1_middle, p2_middle,
            diffusion_coefficient=diffusion_coefficient, 
            bond_length=bond_length, box_size=box_size, 
            time_step=time_step, steps=time_division[2]
        )
        
        # Generate right segment (free diffusion)
        middle_end_p1 = p1_middle_traj.positions[end]
        middle_end_p2 = p2_middle_traj.positions[end]
        p1_right = Particle(middle_end_p1.x, middle_end_p1.y, 1)  # State 1 for free
        p2_right = Particle(middle_end_p2.x, middle_end_p2.y, 1)
        p1_right_traj, p2_right_traj = simulate_diffusion(
            p1_right, p2_right,
            diffusion_coefficient=diffusion_coefficient, 
            box_size=box_size, time_step=time_step, 
            steps=time_division[3]
        )
        
        # Generate left segment (free diffusion, to be reversed)
        middle_start_p1 = p1_middle_traj.positions[1]
        middle_start_p2 = p2_middle_traj.positions[1]
        p1_left = Particle(middle_start_p1.x, middle_start_p1.y, 1)  # State 1 for free
        p2_left = Particle(middle_start_p2.x, middle_start_p2.y, 1)
        p1_left_traj, p2_left_traj = simulate_diffusion(
            p1_left, p2_left,
            diffusion_coefficient=diffusion_coefficient, 
            box_size=box_size, time_step=time_step, 
            steps=time_division[1]
        )
        
        # Reverse left segments (to connect properly)
        p1_left_positions = reverse([p for p in p1_left_traj.positions])
        p2_left_positions = reverse([p for p in p2_left_traj.positions])
        left_times = reverse(copy(p1_left_traj.times))
        
        # Combine segments into single trajectories
        combined_p1_positions = vcat(
            p1_left_positions,
            p1_middle_traj.positions,
            p1_right_traj.positions
        )
        
        combined_p2_positions = vcat(
            p2_left_positions,
            p2_middle_traj.positions,
            p2_right_traj.positions
        )
        
        # Combine times (need to adjust to make continuous)
        middle_start_time = left_times[end] + time_step
        right_start_time = middle_start_time + time_step * length(p1_middle_traj.positions)
        
        middle_times = middle_start_time .+ (0:length(p1_middle_traj.positions)-1) .* time_step
        right_times = right_start_time .+ (0:length(p1_right_traj.positions)-1) .* time_step
        
        combined_times = vcat(left_times, middle_times, right_times)
        
        # Create new trajectories with combined data
        p1_combined = Trajectory(combined_p1_positions, combined_times)
        p2_combined = Trajectory(combined_p2_positions, combined_times)
        
        return p1_combined, p2_combined

    elseif mode == :right_half && length(time_division) == 2
        # Generate middle segment (constrained/dimer diffusion)
        p1_middle = Particle(initial_p1.x, initial_p1.y, 2)  # State 2 for dimer
        p2_middle = Particle(initial_p2.x, initial_p2.y, 2)
        p1_middle_traj, p2_middle_traj = constrained_diffusion(
            p1_middle, p2_middle,
            diffusion_coefficient=diffusion_coefficient, 
            bond_length=bond_length, box_size=box_size, 
            time_step=time_step, steps=time_division[1]
        )
        
        # Generate right segment (free diffusion)
        middle_end_p1 = p1_middle_traj.positions[end]
        middle_end_p2 = p2_middle_traj.positions[end]
        p1_right = Particle(middle_end_p1.x, middle_end_p1.y, 1)  # State 1 for free
        p2_right = Particle(middle_end_p2.x, middle_end_p2.y, 1)
        p1_right_traj, p2_right_traj = simulate_diffusion(
            p1_right, p2_right,
            diffusion_coefficient=diffusion_coefficient, 
            box_size=box_size, time_step=time_step, 
            steps=time_division[2]
        )
        
        # Combine segments
        combined_p1_positions = vcat(p1_middle_traj.positions, p1_right_traj.positions)
        combined_p2_positions = vcat(p2_middle_traj.positions, p2_right_traj.positions)
        
        # Combine times (need to adjust to make continuous)
        right_start_time = p1_middle_traj.times[end] + time_step
        right_times = right_start_time .+ (0:length(p1_right_traj.positions)-1) .* time_step
        
        combined_times = vcat(p1_middle_traj.times, right_times)
        
        # Create new trajectories with combined data
        p1_combined = Trajectory(combined_p1_positions, combined_times)
        p2_combined = Trajectory(combined_p2_positions, combined_times)
        
        return p1_combined, p2_combined

    elseif mode == :middle && length(time_division) == 1
        # Only middle segment (constrained/dimer diffusion)
        p1_middle = Particle(initial_p1.x, initial_p1.y, 2)  # State 2 for dimer
        p2_middle = Particle(initial_p2.x, initial_p2.y, 2)
        p1_middle_traj, p2_middle_traj = constrained_diffusion(
            p1_middle, p2_middle,
            diffusion_coefficient=diffusion_coefficient, 
            bond_length=bond_length, box_size=box_size, 
            time_step=time_step, steps=time_division[1]
        )
        
        return p1_middle_traj, p2_middle_traj

    elseif mode == :left_half && length(time_division) == 2
        # Generate middle segment (constrained/dimer diffusion)
        p1_middle = Particle(initial_p1.x, initial_p1.y, 2)  # State 2 for dimer
        p2_middle = Particle(initial_p2.x, initial_p2.y, 2)
        p1_middle_traj, p2_middle_traj = constrained_diffusion(
            p1_middle, p2_middle,
            diffusion_coefficient=diffusion_coefficient, 
            bond_length=bond_length, box_size=box_size, 
            time_step=time_step, steps=time_division[2]
        )
        
        # Generate left segment (free diffusion, to be reversed)
        middle_start_p1 = p1_middle_traj.positions[1]
        middle_start_p2 = p2_middle_traj.positions[1]
        p1_left = Particle(middle_start_p1.x, middle_start_p1.y, 1)  # State 1 for free
        p2_left = Particle(middle_start_p2.x, middle_start_p2.y, 1)
        p1_left_traj, p2_left_traj = simulate_diffusion(
            p1_left, p2_left,
            diffusion_coefficient=diffusion_coefficient, 
            box_size=box_size, time_step=time_step, 
            steps=time_division[1]
        )
        
        # Reverse left segments (to connect properly)
        p1_left_positions = reverse([p for p in p1_left_traj.positions])
        p2_left_positions = reverse([p for p in p2_left_traj.positions])
        left_times = reverse(copy(p1_left_traj.times))
        
        # Combine segments
        combined_p1_positions = vcat(p1_left_positions, p1_middle_traj.positions)
        combined_p2_positions = vcat(p2_left_positions, p2_middle_traj.positions)
        
        # Combine times (need to adjust to make continuous)
        middle_start_time = left_times[end] + time_step
        middle_times = middle_start_time .+ (0:length(p1_middle_traj.positions)-1) .* time_step
        
        combined_times = vcat(left_times, middle_times)
        
        # Create new trajectories with combined data
        p1_combined = Trajectory(combined_p1_positions, combined_times)
        p2_combined = Trajectory(combined_p2_positions, combined_times)
        
        return p1_combined, p2_combined

    else
        error("Invalid mode or time division")
    end
end

#==============================================================================
# Simulation Functions
==============================================================================#
# Matrix-based simulation
function simulation(transition_rate_12::Float64, transition_rate_21::Float64, changes::Int)
    states, steps = simulate_states(transition_rate_12, transition_rate_21, changes)
    time_in_state = time_sequence_with_split(states)
    
    particle1_trajectory = []
    particle2_trajectory = []

    for i in 1:length(time_in_state)
        section_time_stamps = time_stamps(time_in_state[i])
        mode = modes_selection(time_in_state[i])
        
        if i == 1
            particle1, particle2 = segments(section_time_stamps, mode, radius=0.01, box_size=1.0, time_step=0.016)
            particle1_trajectory = particle1
            particle2_trajectory = particle2
        else 
            temp_particle1, temp_particle2 = segments(section_time_stamps, mode, radius=0.01, box_size=1.0, time_step=0.016)
            
            shift_x = -temp_particle1[1,1] + particle1_trajectory[1,end]
            shift_y = -temp_particle1[2,1] + particle1_trajectory[2,end]
            
            shifted_particle1 = shift(temp_particle1, shift_x, shift_y)
            shifted_particle2 = shift(temp_particle2, shift_x, shift_y)
            
            segment_view = @view shifted_particle2[:,1:time_in_state[i][1].second]
            path_correction!(segment_view, [particle2_trajectory[1,end], particle2_trajectory[2,end]])
            
            particle1_trajectory = hcat(particle1_trajectory, shifted_particle1)
            particle2_trajectory = hcat(particle2_trajectory, shifted_particle2)
        end
    end
    
    return particle1_trajectory, particle2_trajectory
end

# Particle-based simulation
function simulation_with_particles(transition_rate_12::Float64, transition_rate_21::Float64, 
                                 desired_state_changes::Int; diffusion_coefficient::Float64=0.01, 
                                 bond_length::Float64=0.2, box_size::Float64=1.0, 
                                 time_step::Float64=0.01)
    
    # Generate states with time information
    states, times, steps = simulate_states(transition_rate_12, transition_rate_21, 
                                         desired_state_changes, time_step, true)
    
    # Get transitions
    transitions = get_state_transitions(states, times)
    
    # Initialize particles at random positions
    initial_x1 = rand() * box_size
    initial_y1 = rand() * box_size
    initial_state = states[1]
    
    # Initialize first particle
    p1 = Particle(initial_x1, initial_y1, initial_state)
    
    # Initialize second particle based on state
    local p2
    if initial_state == 1  # Free state
        initial_x2 = rand() * box_size
        initial_y2 = rand() * box_size
        p2 = Particle(initial_x2, initial_y2, initial_state)
    else  # Dimer state
        angle = 2π * rand()
        initial_x2 = initial_x1 + bond_length * cos(angle)
        initial_y2 = initial_y1 + bond_length * sin(angle)
        initial_x2 = clamp(initial_x2, 0, box_size)
        initial_y2 = clamp(initial_y2, 0, box_size)
        p2 = Particle(initial_x2, initial_y2, initial_state)
    end
    
    # Create trajectories
    p1_trajectory = Trajectory([p1], [0.0])
    p2_trajectory = Trajectory([p2], [0.0])
    
    # Simulate each segment between transitions
    for i in 1:length(transitions)
        start_time = i == 1 ? 0.0 : transitions[i-1][1]
        end_time = transitions[i][1]
        segment_duration = end_time - start_time
        segment_steps = max(1, round(Int, segment_duration / time_step))
        
        # Get current state
        current_state = i == 1 ? states[1] : states[findfirst(t -> t >= start_time, times)]
        
        # Start with the last positions but update state
        current_p1 = p1_trajectory.positions[end]
        current_p2 = p2_trajectory.positions[end]
        p1_state_updated = Particle(current_p1.x, current_p1.y, current_state)
        p2_state_updated = Particle(current_p2.x, current_p2.y, current_state)
        
        # Simulate appropriate diffusion
        local segment_p1_traj, segment_p2_traj
        if current_state == 1  # Free diffusion
            segment_p1_traj, segment_p2_traj = simulate_diffusion(
                p1_state_updated, p2_state_updated,
                diffusion_coefficient=diffusion_coefficient,
                box_size=box_size, time_step=time_step,
                steps=segment_steps
            )
        else  # Dimer diffusion
            segment_p1_traj, segment_p2_traj = constrained_diffusion(
                p1_state_updated, p2_state_updated,
                diffusion_coefficient=diffusion_coefficient,
                bond_length=bond_length, box_size=box_size,
                time_step=time_step, steps=segment_steps
            )
        end
        
        # Append segment to full trajectory (skip first point to avoid duplicates)
        for j in 2:length(segment_p1_traj.positions)
            segment_time = start_time + (j-1)*time_step
            add_position!(p1_trajectory, segment_p1_traj.positions[j], segment_time)
            add_position!(p2_trajectory, segment_p2_traj.positions[j], segment_time)
        end
    end
    
    # Simulate final segment
    if !isempty(transitions)
        final_start_time = transitions[end][1]
        final_end_time = times[end]
        final_duration = final_end_time - final_start_time
        final_steps = max(1, round(Int, final_duration / time_step))
        
        # Get final state
        final_state = states[end]
        
        # Start with the last positions but update state
        final_p1 = p1_trajectory.positions[end]
        final_p2 = p2_trajectory.positions[end]
        p1_state_updated = Particle(final_p1.x, final_p1.y, final_state)
        p2_state_updated = Particle(final_p2.x, final_p2.y, final_state)
        
        # Simulate appropriate diffusion for final segment
        local final_p1_traj, final_p2_traj
        if final_state == 1  # Free diffusion
            final_p1_traj, final_p2_traj = simulate_diffusion(
                p1_state_updated, p2_state_updated,
                diffusion_coefficient=diffusion_coefficient,
                box_size=box_size, time_step=time_step,
                steps=final_steps
            )
        else  # Dimer diffusion
            final_p1_traj, final_p2_traj = constrained_diffusion(
                p1_state_updated, p2_state_updated,
                diffusion_coefficient=diffusion_coefficient,
                bond_length=bond_length, box_size=box_size,
                time_step=time_step, steps=final_steps
            )
        end
        
        # Append final segment (skip first point to avoid duplicates)
        for j in 2:length(final_p1_traj.positions)
            final_time = final_start_time + (j-1)*time_step
            add_position!(p1_trajectory, final_p1_traj.positions[j], final_time)
            add_position!(p2_trajectory, final_p2_traj.positions[j], final_time)
        end
    end
    
    return p1_trajectory, p2_trajectory, states, times
end

#==============================================================================
# Animation Functions
==============================================================================#
# Matrix-based animation
function animate_particles(particle1_positions::Matrix, particle2_positions::Matrix, filename::String="particle_animation.mp4")
    n_steps = size(particle1_positions, 2)
    
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    all_x = vcat(particle1_positions[1,:], particle2_positions[1,:])
    all_y = vcat(particle1_positions[2,:], particle2_positions[2,:])
    
    limits!(ax, -1, 1, -1, 1)
    
    point1 = scatter!([particle1_positions[1,1]], [particle1_positions[2,1]], color=:blue, markersize=10)
    point2 = scatter!([particle2_positions[1,1]], [particle2_positions[2,1]], color=:red, markersize=10)
    
    record(fig, filename, 1:n_steps; framerate=10) do frame
        point1[1] = [particle1_positions[1,frame]]
        point1[2] = [particle1_positions[2,frame]]
        
        point2[1] = [particle2_positions[1,frame]]
        point2[2] = [particle2_positions[2,frame]]
    end
    
    return filename
end

# Particle-based animation
function animate_particles(p1_trajectory::Trajectory, p2_trajectory::Trajectory, filename::String="particle_state_animation.mp4")
    # Extract coordinates and states
    p1_x = [p.x for p in p1_trajectory.positions]
    p1_y = [p.y for p in p1_trajectory.positions]
    p2_x = [p.x for p in p2_trajectory.positions]
    p2_y = [p.y for p in p2_trajectory.positions]
    
    p1_states = [p.state for p in p1_trajectory.positions]
    
    # Ensure trajectories are the same length
    n_steps = min(length(p1_x), length(p2_x))
    
    # Create figure
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Calculate axis limits
    all_x = vcat(p1_x, p2_x)
    all_y = vcat(p1_y, p2_y)
    
    x_min, x_max = minimum(all_x), maximum(all_x)
    y_min, y_max = minimum(all_y), maximum(all_y)
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    limits!(ax, x_min - x_padding, x_max + x_padding, y_min - y_padding, y_max + y_padding)
    
    # Create initial points
    point1 = scatter!([p1_x[1]], [p1_y[1]], color=p1_states[1] == 1 ? :blue : :purple, markersize=10)
    point2 = scatter!([p2_x[1]], [p2_y[1]], color=p1_states[1] == 1 ? :red : :orange, markersize=10)
    
    # Add bond line for dimer state
    line = lines!([p1_x[1], p2_x[1]], [p1_y[1], p2_y[1]], 
                 color=p1_states[1] == 2 ? :green : :transparent, 
                 linewidth=2, visible=p1_states[1] == 2)
    
    # Add state indicator
    state_text = text!("State: $(p1_states[1])", 
                     position=(x_min, y_max + y_padding/2), 
                     align=(:left, :center), 
                     fontsize=20)
    
    # Create animation
    record(fig, filename, 1:n_steps; framerate=30) do frame
        # Update particle positions
        point1[1] = [p1_x[frame]]
        point1[2] = [p1_y[frame]]
        point2[1] = [p2_x[frame]]
        point2[2] = [p2_y[frame]]
        
        # Update colors based on state
        point1.color = p1_states[frame] == 1 ? :blue : :purple
        point2.color = p1_states[frame] == 1 ? :red : :orange
        
        # Update bond line
        if p1_states[frame] == 2  # Dimer state
            line[1] = [p1_x[frame], p2_x[frame]]
            line[2] = [p1_y[frame], p2_y[frame]]
            line.visible = true
            line.color = :green
        else
            line.visible = false
        end
        
        # Update state text
        state_text.text = "State: $(p1_states[frame])"
    end
    
    return filename
end

#==============================================================================
# Forward Algorithm & Analysis Functions - Matrix Version
==============================================================================#
function modified_bessel(angle_step::Float64, distance1::Float64, distance2::Float64, sigma::Float64)
    result = 0.0
    x = (distance1*distance2)/(sigma^2)

    for θ in 0:angle_step:2*pi
        result += exp(x * cos(θ))
    end

    result *= angle_step / (2*pi)
    return result 
end

function compute_free_density(particle1::Matrix{Float64}, particle2::Matrix{Float64}, index::Int; 
                             sigma::Float64=0.1, dt::Float64=0.01)
    delta_x_current = particle1[1, index] - particle2[1, index]
    delta_y_current = particle1[2, index] - particle2[2, index]
    
    delta_x_prev = particle1[1, index-1] - particle2[1, index-1]
    delta_y_prev = particle1[2, index-1] - particle2[2, index-1]
    
    distance_prev_square = (delta_x_prev^2) + (delta_y_prev^2)
    distance_prev = sqrt(distance_prev_square)
    distance_current_square = (delta_x_current^2) + (delta_y_current^2)
    distance_current = sqrt(distance_current_square)
    
    density = (distance_current/sigma^2) * (exp((-(distance_current_square) - (distance_prev_square))/sigma^2)) * 
              modified_bessel(dt, distance_current, distance_prev, sigma)
    
    if density == Inf || isnan(density) || density == 0
        density = 0
        for θ in 0:dt:2*pi
            density += exp(-((distance_current_square) + (distance_prev_square) - 
                          (2*distance_current*distance_prev*cos(θ)))/(2*sigma^2))
        end 
        density *= (distance_current/sigma^2)*dt
    end
    
    return density
end

function compute_dimer_density(particle1::Matrix{Float64}, particle2::Matrix{Float64}, index::Int, 
                              dimer_distance::Float64; sigma::Float64=0.1, dt::Float64=0.01)
    delta_x = particle1[1, index] - particle2[1, index]
    delta_y = particle1[2, index] - particle2[2, index]
    
    distance_square = (delta_x^2) + (delta_y^2)
    distance = sqrt(distance_square)
    
    density = (distance/sigma^2) * exp((-(dimer_distance^2) - (distance_square)/sigma^2)) * 
              modified_bessel(dt, dimer_distance, distance, sigma)
    
    if density == Inf || isnan(density) || density == 0
        density = 0
        for θ in 0:dt:2*pi
            density += exp(-((distance_square) + (dimer_distance^2) - 
                          (2*distance*dimer_distance*cos(θ)))/(2*sigma^2))
        end 
        density *= (distance/sigma^2)*dt
    end
    
    return density
end

function forward_algorithm(particle1::Matrix{Float64}, particle2::Matrix{Float64}, dimer_distance::Float64, 
                         params::Vector{Float64}; dt::Float64=0.01, sigma::Float64=0.1)
    N = min(size(particle1, 2), size(particle2, 2)) - 1
    
    alpha = zeros(2, N)
    scale = zeros(N)
    
    frame_index = min(2, N)
    
    emission_free_initial = compute_free_density(particle1, particle2, frame_index, sigma=sigma, dt=dt)
    emission_dimer_initial = compute_dimer_density(particle1, particle2, frame_index, dimer_distance, sigma=sigma, dt=dt)
    
    alpha[1, 1] = emission_free_initial
    alpha[2, 1] = emission_dimer_initial
    
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= max(scale[1], eps())
    
    rate_on = params[1]
    rate_off = params[2]
    transition_matrix = [1-(rate_on*dt) rate_on*dt; rate_off*dt 1-(rate_off*dt)]
    
    for t in 2:N
        frame_index = min(t + 2, size(particle1, 2))
        
        emission_free = compute_free_density(particle1, particle2, frame_index, sigma=sigma, dt=dt)
        emission_dimer = compute_dimer_density(particle1, particle2, frame_index, dimer_distance, sigma=sigma, dt=dt)
        
        alpha[1, t] = (alpha[1, t-1]*transition_matrix[1,1] + alpha[2, t-1]*transition_matrix[2,1]) * emission_free
        alpha[2, t] = (alpha[1, t-1]*transition_matrix[1,2] + alpha[2, t-1]*transition_matrix[2,2]) * emission_dimer
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= max(scale[t], eps())
    end
    
    loglikelihood = sum(log.(max.(scale, eps())))
    
    return alpha, loglikelihood
end

#==============================================================================
# Forward Algorithm & Analysis Functions - Particle Version
==============================================================================#
function compute_free_density(p1::Particle, p2::Particle, prev_p1::Particle, prev_p2::Particle;
                            sigma::Float64=0.1, dt::Float64=0.01)
    # Calculate current distance
    delta_x_current = p1.x - p2.x
    delta_y_current = p1.y - p2.y
    distance_current_square = delta_x_current^2 + delta_y_current^2
    distance_current = sqrt(distance_current_square)
    
    # Calculate previous distance
    delta_x_prev = prev_p1.x - prev_p2.x
    delta_y_prev = prev_p1.y - prev_p2.y
    distance_prev_square = delta_x_prev^2 + delta_y_prev^2
    distance_prev = sqrt(distance_prev_square)
    
    # Compute density
    density = (distance_current/sigma^2) * exp(-(distance_current_square + distance_prev_square)/sigma^2) *
              modified_bessel(dt, distance_current, distance_prev, sigma)
    
    # Handle numerical issues
    if density == Inf || isnan(density) || density == 0
        density = 0
        for θ in 0:dt:2*pi
            density += exp(-((distance_current_square) + (distance_prev_square) - 
                          (2*distance_current*distance_prev*cos(θ)))/(2*sigma^2))
        end
        density *= (distance_current/sigma^2)*dt
    end
    
    return density
end

function compute_dimer_density(p1::Particle, p2::Particle, prev_p1::Particle, prev_p2::Particle, 
                             bond_length::Float64; sigma::Float64=0.1, dt::Float64=0.01)
    # Calculate current distance
    delta_x_current = p1.x - p2.x
    delta_y_current = p1.y - p2.y
    distance_current_square = delta_x_current^2 + delta_y_current^2
    distance_current = sqrt(distance_current_square)
    
    # Compute density
    density = (distance_current/sigma^2) * exp(-(bond_length^2 + distance_current_square)/sigma^2) *
              modified_bessel(dt, bond_length, distance_current, sigma)
    
    # Handle numerical issues
    if density == Inf || isnan(density) || density == 0
        density = 0
        for θ in 0:dt:2*pi
            density += exp(-((distance_current_square) + (bond_length^2) - 
                          (2*distance_current*bond_length*cos(θ)))/(2*sigma^2))
        end
        density *= (distance_current/sigma^2)*dt
    end
    
    return density
end

function forward_algorithm(p1_trajectory::Trajectory, p2_trajectory::Trajectory, 
                         bond_length::Float64, params::Vector{Float64};
                         dt::Float64=0.01, sigma::Float64=0.1)
    # Ensure trajectories have the same length
    n_frames = min(length(p1_trajectory.positions), length(p2_trajectory.positions))
    
    # Initialize matrices
    alpha = zeros(2, n_frames - 1)  # We need at least 2 frames to compute transitions
    scale = zeros(n_frames - 1)
    
    # Get the first two particles for initialization
    p1_current = p1_trajectory.positions[2]
    p2_current = p2_trajectory.positions[2]
    p1_prev = p1_trajectory.positions[1]
    p2_prev = p2_trajectory.positions[1]
    
    # Compute initial emissions
    e1_initial = compute_free_density(p1_current, p2_current, p1_prev, p2_prev, sigma=sigma, dt=dt)
    e2_initial = compute_dimer_density(p1_current, p2_current, p1_prev, p2_prev, bond_length, sigma=sigma, dt=dt)
    
    # Initialize alpha values
    alpha[1, 1] = e1_initial
    alpha[2, 1] = e2_initial
    
    # Scale alpha values
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= max(scale[1], eps())
    
    # Extract transition rates
    rate_on = params[1]   # Transition rate from state 1 (free) to state 2 (dimer)
    rate_off = params[2]  # Transition rate from state 2 (dimer) to state 1 (free)
    
    # Define transition matrix
    transition_matrix = [
        1 - (rate_on * dt)  rate_on * dt;
        rate_off * dt       1 - (rate_off * dt)
    ]
    
    # Run forward algorithm
    for t in 2:(n_frames-1)
        # Get current and previous particles
        p1_current = p1_trajectory.positions[t+1]
        p2_current = p2_trajectory.positions[t+1]
        p1_prev = p1_trajectory.positions[t]
        p2_prev = p2_trajectory.positions[t]
        
        # Compute emissions
        e1 = compute_free_density(p1_current, p2_current, p1_prev, p2_prev, sigma=sigma, dt=dt)
        e2 = compute_dimer_density(p1_current, p2_current, p1_prev, p2_prev, bond_length, sigma=sigma, dt=dt)
        
        # Update alpha values
        alpha[1, t] = (alpha[1, t-1] * transition_matrix[1,1] + alpha[2, t-1] * transition_matrix[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1] * transition_matrix[1,2] + alpha[2, t-1] * transition_matrix[2,2]) * e2
        
        # Scale alpha values
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= max(scale[t], eps())
    end
    
    # Compute log-likelihood
    loglikelihood = sum(log.(max.(scale, eps())))
    
    return alpha, loglikelihood
end

#==============================================================================
# Optimization Functions
==============================================================================#
# Matrix-based optimization
function optimize_parameters(particle1::Matrix{Float64}, particle2::Matrix{Float64}, 
                           dimer_distance::Float64, initial_params::Vector{Float64})
    function objective(params)
        _, loglikelihood = forward_algorithm(particle1, particle2, dimer_distance, params)
        return -loglikelihood  # Minimize negative log-likelihood
    end
    
    # Run optimization
    result = optimize(objective, initial_params, NelderMead())
    
    # Extract optimized parameters
    optimal_params = Optim.minimizer(result)
    rate_on_optimized = optimal_params[1]
    rate_off_optimized = optimal_params[2]
    
    # Calculate final log-likelihood
    _, final_loglikelihood = forward_algorithm(particle1, particle2, dimer_distance, optimal_params)
    
    return rate_on_optimized, rate_off_optimized, final_loglikelihood
end

# Particle-based optimization
function optimize_parameters(p1_trajectory::Trajectory, p2_trajectory::Trajectory, 
                           bond_length::Float64, initial_params::Vector{Float64})
    function objective(params)
        _, loglikelihood = forward_algorithm(p1_trajectory, p2_trajectory, bond_length, params)
        return -loglikelihood  # Minimize negative log-likelihood
    end
    
    # Run optimization
    result = optimize(objective, initial_params, NelderMead())
    
    # Extract optimized parameters
    optimal_params = Optim.minimizer(result)
    rate_on_optimized = optimal_params[1]
    rate_off_optimized = optimal_params[2]
    
    # Calculate final log-likelihood
    _, final_loglikelihood = forward_algorithm(p1_trajectory, p2_trajectory, bond_length, optimal_params)
    
    return rate_on_optimized, rate_off_optimized, final_loglikelihood
end

#==============================================================================
# Example Usage
==============================================================================#
function run_simulation_example()
    # Parameters
    transition_rate_12 = 0.5  # Rate from free to dimer
    transition_rate_21 = 0.15  # Rate from dimer to free
    desired_state_changes = 6
    bond_length = 0.2
    
    # Matrix-based simulation
    println("\nRunning matrix-based simulation...")
    particle1_matrix, particle2_matrix = simulation(transition_rate_12, transition_rate_21, desired_state_changes)
    
    # Create animation from matrix data
    matrix_animation = animate_particles(particle1_matrix, particle2_matrix, "matrix_simulation.mp4")
    println("Matrix animation saved as: $matrix_animation")
    
    # Estimate parameters from matrix data
    initial_guess = [0.3, 0.3]
    rate_on_est, rate_off_est, loglikelihood = optimize_parameters(
        particle1_matrix, particle2_matrix, bond_length, initial_guess)
    
    println("\nMatrix-based parameter estimation:")
    println("True transition rates: k12 = $transition_rate_12, k21 = $transition_rate_21")
    println("Estimated transition rates: k12 = $rate_on_est, k21 = $rate_off_est")
    println("Log-likelihood: $loglikelihood")
    
    # Particle-based simulation
    println("\nRunning particle-based simulation...")
    p1_trajectory, p2_trajectory, states, times = simulation_with_particles(
        transition_rate_12, transition_rate_21, desired_state_changes, 
        bond_length=bond_length)
    
    # Create animation from particle data
    particle_animation = animate_particles(p1_trajectory, p2_trajectory, "particle_simulation.mp4")
    println("Particle animation saved as: $particle_animation")
    
    # Estimate parameters from particle data
    rate_on_est_p, rate_off_est_p, loglikelihood_p = optimize_parameters(
        p1_trajectory, p2_trajectory, bond_length, initial_guess)
    
    println("\nParticle-based parameter estimation:")
    println("True transition rates: k12 = $transition_rate_12, k21 = $transition_rate_21")
    println("Estimated transition rates: k12 = $rate_on_est_p, k21 = $rate_off_est_p")
    println("Log-likelihood: $loglikelihood_p")
    
    return (particle1_matrix, particle2_matrix), (p1_trajectory, p2_trajectory)
end
function path_correction!(particle_path::SubArray{Float64}, new_destination::Vector{Float64})
    # Use the same implementation as for Matrix
    if size(particle_path, 1) != 2
        error("Particle path must be a 2×n matrix")
    end
    
    if length(new_destination) != 2
        error("New destination must be a 2-element vector")
    end
    
    n = size(particle_path, 2)
    
    start_point = particle_path[:, 1]
    original_end = particle_path[:, end]
    
    original_displacement = original_end - start_point
    desired_displacement = new_destination - start_point
    
    for i in 2:n
        t = (i - 1) / (n - 1)
        
        expected_position = start_point + original_displacement * t
        diffusion_component = particle_path[:, i] - expected_position
        new_expected_position = start_point + desired_displacement * t
        
        particle_path[:, i] = new_expected_position + diffusion_component
    end
    
    return nothing
end
# Call this function to run a complete example
# matrix_data, particle_data = run_simulation_example()