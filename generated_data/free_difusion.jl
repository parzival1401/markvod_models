

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
