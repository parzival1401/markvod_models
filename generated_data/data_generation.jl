include("dimer_difussion.jl")
include("free_difusion.jl")
include("state_generator.jl")


using GLMakie
function animate_particles(p1, p2, filename="particle_animation.mp4")
    n_steps = size(p1, 2)
    
    # Create figure
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Calculate axis limits
    all_x = vcat(p1[1,:], p2[1,:])
    all_y = vcat(p1[2,:], p2[2,:])
    
    x_range = maximum(all_x) - minimum(all_x)
    y_range = maximum(all_y) - minimum(all_y)
    
    limits!(ax, 
        minimum(all_x) - 0.1*x_range, 
        maximum(all_x) + 0.1*x_range,
        minimum(all_y) - 0.1*y_range, 
        maximum(all_y) + 0.1*y_range)
    
    # Create initial points
    point1 = scatter!([p1[1,1]], [p1[2,1]], color=:blue, markersize=10)
    point2 = scatter!([p2[1,1]], [p2[2,1]], color=:red, markersize=10)
    
    # Create animation
    record(fig, filename, 1:n_steps; framerate=10) do frame
        # Update particle positions for current frame
        point1[1] = [p1[1,frame]]
        point1[2] = [p1[2,frame]]
        
        point2[1] = [p2[1,frame]]
        point2[2] = [p2[2,frame]]
    end
    
    return filename
end




function reverse_columns_preserve_size(arr)
    num_rows, num_cols = size(arr)
    result = zeros(eltype(arr), num_rows, num_cols)
    
    for col in 1:num_cols
        result[:, col] = arr[:, num_cols - col + 1]
    end
    
    return result
end

k12 = 0.5
k21 = 0.15 
states, steps = simulate_states(k12, k21, 6)
transitions = get_state_transitions(states)



# 0 - 18  state 1 
steps = transitions[1][1]
positions = ([p1_path_2[2,1],p1_path_2[1,1]],[p2_path_2[2,1],p2_path_2[1,1]])
p1_path_1, p2_path_1 = simulate_diffusion(initial_positions=positions,steps = steps, animate= true,filename = "0-18.mp4",box = 1.0,dt = 0.016)
p1_path_reversed = reverse_columns_preserve_size(p1_path_1)
p2_path_reversed = reverse_columns_preserve_size(p2_path_1)

#####.    animate_particles(p1_path_reversed, p2_path_reversed, "0-18.mp4")

# 19 - 84 state 2
steps = transitions[2][1]-transitions[1][1]
p1_path_2, p2_path_2 = constrained_diffusion(steps = steps, animate= true,filename = "19-84.mp4",r = 0.01,box = 1.0,dt = 0.016)

# 85 - 102 state 1 part 1 
steps =round(Int, abs((transitions[3][1]/2) -transitions[2][1]))
positions = ([p1_path_2[2,end],p1_path_2[1,end]],[p2_path_2[2,end],p2_path_2[1,end]])
p1_path_3, p2_path_3 = simulate_diffusion(initial_positions=positions,steps = steps, animate= true,filename = "85-102.mp4",box = 1.0,dt = 0.016)

#### first part animation 

p1 = hcat(p1_path_reversed,p1_path_2,p1_path_3)
p2 = hcat(p2_path_reversed,p2_path_2,p2_path_3)
animate_particles(p1, p2, "first_part_animation.mp4")

# part 2 

# 103 - 212 state 2



# 213 - 227 state 1



# 228 - 318 state 2