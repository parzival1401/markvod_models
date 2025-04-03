
using GLMakie

function constrained_diffusion(;
    initial_p1 = nothing,
    D = 0.01,
    r = 0.2,
    box = 1.0,
    dt = 0.016,
    steps = 500,
    animate = false,
    filename = "constrained_diffusion.mp4",
    framerate = 30
 )
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
    
    if animate
        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect(), 
              title="Constrained Diffusion (r=$r)",
              xlabel="X position", ylabel="Y position")
        limits!(ax, 0, box, 0, box)
        
        # Create particles
        s1 = scatter!(ax, [p1_pos[1]], [p1_pos[2]], color=:blue, markersize=15)
        s2 = scatter!(ax, [p2_pos[1]], [p2_pos[2]], color=:red, markersize=15)
        
    
        
        # Record animation
        record(fig, filename, 1:steps; framerate = framerate) do frame
            if frame > 1
                # Update particle positions
                p1_vel .= randn(2) .* sqrt(2D/dt)
                p1_pos .+= p1_vel .* dt
                
                # Enforce boundaries for p1
                for i in 1:2
                    p1_pos[i] = clamp(p1_pos[i], 0, box)
                end
                
                # Update p2 with random angle but fixed distance r
                new_angle = 2π * rand()
                p2_pos .= p1_pos + r .* [cos(new_angle), sin(new_angle)]
                
                # Handle boundary issues for p2
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
                
                # Store positions
                p1_positions[:, frame] = p1_pos
                p2_positions[:, frame] = p2_pos
            end
            
            # Update visual elements
            s1[1] = [p1_pos[1]]; s1[2] = [p1_pos[2]]
            s2[1] = [p2_pos[1]]; s2[2] = [p2_pos[2]]
        end
        
        println("Animation saved as '$filename'")
    else
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
    end
    
    return p1_positions, p2_positions
 end

 p1,p2 = constrained_diffusion(initial_p1 = nothing, D = 0.01,r = 0.01,box = 1.0,dt = 0.016,steps = 500,animate = true)