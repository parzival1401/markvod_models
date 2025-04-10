using GLMakie

function simulate_diffusion(;
    initial_positions = nothing,
    D = 0.01,
    box = 1.0,
    dt = 0.016,
    steps = 2000,
    animate = false,
    filename = "diffusion.mp4",
    framerate = 30
)
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
    
    if animate
        # Plot setup
        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect(),
              title="Free Diffusion Simulation",
              xlabel="X position", ylabel="Y position")
        limits!(ax, 0, box, 0, box)
        
        # Create particles
        s1 = scatter!(ax, [p1_pos[1]], [p1_pos[2]], color=:blue, markersize=20)
        s2 = scatter!(ax, [p2_pos[1]], [p2_pos[2]], color=:red, markersize=20)
        
        display(fig)
        
        # Record animation
        record(fig, filename, 1:steps; framerate = framerate) do frame
            if frame > 1
                # Update particle positions
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
            
            # Update visual elements
            s1[1] = [p1_pos[1]]; s1[2] = [p1_pos[2]]
            s2[1] = [p2_pos[1]]; s2[2] = [p2_pos[2]]
        end
        
        println("Animation saved as '$filename'")
    else
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
    end
    
    return p1_positions, p2_positions
end

## p1_path, p2_path = constrained_diffusion(initial_p1 = nothing, D = 0.01, r = 0.01, box = 1.0, dt = 0.016, steps = 500, animate = true)
## p1_path, p2_path = simulate_diffusion(initial_positions = nothing, D = 0.01, box = 1.0, dt = 0.016, steps = 500, animate = true)