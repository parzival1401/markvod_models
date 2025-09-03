


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
