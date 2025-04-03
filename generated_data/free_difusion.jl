using GLMakie

function simulate_diffusion(; 
    D = 0.01, 
    dt = 0.016, 
    steps = 2000, 
    box = 1.0, 
    make_animation = true, 
    output_file = "diffusion.mp4", 
    gif_frames = 500, 
    initial_positions = nothing
)
    # Initialize positions
    if initial_positions isa Tuple && length(initial_positions) == 2
        p1 = (initial_positions[1], randn(2) * sqrt(2D/dt))
        p2 = (initial_positions[2], randn(2) * sqrt(2D/dt))
    else
        p1 = (rand(2), randn(2) * sqrt(2D/dt))
        p2 = (rand(2), randn(2) * sqrt(2D/dt))
    end

    # Position storage
    p1_pos = zeros(2, steps)
    p2_pos = zeros(2, steps)

    if make_animation
        # Plot setup
        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect())
        limits!(ax, 0, box, 0, box)

        s1 = scatter!(ax, [p1[1][1]], [p1[1][2]], color=:blue, markersize=20)
        s2 = scatter!(ax, [p2[1][1]], [p2[1][2]], color=:red, markersize=20)

        display(fig)

        # Main simulation loop with animation update
        for t in 1:steps
            for (pos, vel) in (p1, p2)
                vel .= randn(2) .* sqrt(2D/dt)
                pos .+= vel .* dt
                for i in 1:2
                    pos[i] = clamp(pos[i], 0, box)
                end
            end

            s1[1] = [p1[1][1]]; s1[2] = [p1[1][2]]
            s2[1] = [p2[1][1]]; s2[2] = [p2[1][2]]

            p1_pos[:, t] = p1[1]
            p2_pos[:, t] = p2[1]

            sleep(dt)
        end

        # Save MP4 animation
        frames = Observable(1)
        record(fig, output_file, 1:gif_frames; framerate=30) do frame
            for (pos, vel) in (p1, p2)
                vel .= randn(2) .* sqrt(2D/dt)
                pos .+= vel .* dt
                for i in 1:2
                    pos[i] = clamp(pos[i], 0, box)
                end
            end

            s1[1] = [p1[1][1]]; s1[2] = [p1[1][2]]
            s2[1] = [p2[1][1]]; s2[2] = [p2[1][2]]

            frames[] = frame
        end
    else
        # Simulation without visualization
        for t in 1:steps
            for (pos, vel) in (p1, p2)
                vel .= randn(2) .* sqrt(2D/dt)
                pos .+= vel .* dt
                for i in 1:2
                    pos[i] = clamp(pos[i], 0, box)
                end
            end
            p1_pos[:, t] = p1[1]
            p2_pos[:, t] = p2[1]
        end
    end

    return p1_pos, p2_pos
end

p1_path, p2_path = simulate_diffusion(make_animation=true)
