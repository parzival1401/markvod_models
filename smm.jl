
using SMLMSim
using CairoMakie
using Random

function run_simulation(;density=0.01, t_max=50, box_size=10, k_off=0.5, r_react=4,save=false,boundary="reflecting")
    state_history, args = SMLMSim.InteractionDiffusion.smoluchowski(density=density,
        t_max=t_max, 
        box_size=box_size,
        k_off=k_off,
        r_react=r_react, ## reflection mode 
        boundary=boundary
        )
    dimer_history = SMLMSim.get_dimers(state_history)
    if save == true 
    SMLMSim.gen_movie(state_history,args; filename="defaultsim.mp4")
    end 
    return state_history, args, dimer_history
end


function extract_particle_trajectories(state_history)
    num_frames = size(state_history.frames, 1)
    particle1_x = zeros(num_frames)
    particle1_y = zeros(num_frames)
    particle1_state = zeros(num_frames)
    particle2_x = zeros(num_frames)
    particle2_y = zeros(num_frames)
    particle2_state = zeros(num_frames)

    for frame_idx in eachindex(state_history.frames)
        current_frame = state_history.frames[frame_idx]
        particle1_x[frame_idx] = current_frame.molecules[1].x
        particle1_y[frame_idx] = current_frame.molecules[1].y
        particle2_x[frame_idx] = current_frame.molecules[2].x
        particle2_y[frame_idx] = current_frame.molecules[2].y
        particle2_state[frame_idx] = current_frame.molecules[2].state
        particle1_state[frame_idx] = current_frame.molecules[1].state
    end

    return particle1_x, particle1_y, particle1_state, particle2_x, particle2_y, particle2_state
end



function calculate_distances(particle1_x, particle1_y, particle2_x, particle2_y)
    distance_x = particle2_x - particle1_x
    distance_y = particle2_y - particle1_y
    return distance_x, distance_y
end


function modified_bessel(dt, d1, d2, σ)
    result = 0.0
    
    x = (d1 * d2) / (σ^2)
    
    for θ in 0:dt:2π
        result += exp(x * cos(θ))
    end
    
    result *= dt / (2π)
    
    return result
end


function compute_free_density(pos_x, pos_y, σ, dt=0.01)
    distances = zeros(size(pos_x))
    density_vals = zeros(size(pos_x))
    
    for i in 1:(length(pos_x)-1)
        distances[i] = sqrt((pos_x[i+1] - pos_x[i])^2 + (pos_y[i+1] - pos_y[i])^2)
    end
    
    for i in 1:(length(distances)-1)
        density_vals[i] = (distances[i]/σ^2) *  exp((-(distances[i+1]^2) - (distances[i]^2))/σ^2) *  modified_bessel(dt, distances[i+1], distances[i], σ)
    end
    
    return density_vals, distances
end


function compute_dimer_density(pos_x, pos_y, σ, dimer_length, dt=0.01)
    distances = zeros(size(pos_x))
    density_vals = zeros(size(pos_x))
    
    for i in 1:(length(pos_x)-1)
        distances[i] = sqrt((pos_x[i+1] - pos_x[i])^2 + (pos_y[i+1] - pos_y[i])^2)
    end
    
    for i in 1:(length(distances)-1)
        density_vals[i] = ((distances[i])/σ^2) *  exp((-(dimer_length^2) - distances[i])/σ^2) *  modified_bessel(dt, dimer_length, distances[i], σ)
    end
    
    return density_vals, distances
end

function compute_density(d1, d2, σ)
    return (d1/σ^2) * exp((-(d2^2) - (d1^2))/σ^2) * modified_bessel(0.01, d2, d1, σ)
end

Random.seed!(1234)

# Run simulation and get data
state_history, args, dimer_history = run_simulation(box_size=15,save=true,k_off=0.5);
p1x, p1y, p1s, p2x, p2y, p2s = extract_particle_trajectories(state_history)
distance_x, distance_y = calculate_distances(p1x, p1y, p2x, p2y)

# First plot: Distance differences over time
fig = Figure(size=(1000, 500)) 

ax1 = Axis(fig[1, 1],
   xlabel = "Time Frame",
   ylabel = "Distance X (dnx)",
   title = "X Distance Difference Over Time")

lines!(ax1, 1:length(distance_x), distance_x, color = :blue, linewidth = 2)
scatter!(ax1, 1:length(distance_x), distance_x, color = :blue, markersize = 4)

ax2 = Axis(fig[1, 2],
   xlabel = "Time Frame",
   ylabel = "Distance Y (dny)",
   title = "Y Distance Difference Over Time")

lines!(ax2, 1:length(distance_y), distance_y, color = :red, linewidth = 2)
scatter!(ax2, 1:length(distance_y), distance_y, color = :red, markersize = 4)

display(fig)

# Calculate densities for different σ values
density_var1, dn_square1 = compute_free_density(distance_x, distance_y, 0.1)
density_var2, dn_square2 = compute_free_density(distance_x, distance_y, 1)
density_var3, dn_square3 = compute_free_density(distance_x, distance_y, 5)
density_var4, dn_square4 = compute_free_density(distance_x, distance_y, 10)
density_var5, dn_square5 = compute_free_density(distance_x, distance_y, 0.5)

# Second plot: Distance vs Density for different σ values
fig = Figure(size=(1000, 500))

ax1 = Axis(fig[1, 1], xlabel = "dn", ylabel = "g(x)", title = "X Distance Difference Over Time")

scatter!(ax1, dn_square1, density_var1, color = :blue, linewidth = 2, label = "δ = 0.1")
scatter!(ax1, dn_square2, density_var2, color = :red, linewidth = 2, label = "δ = 1.0")
scatter!(ax1, dn_square3, density_var3, color = :green, linewidth = 2, label = "δ = 5.0")
scatter!(ax1, dn_square4, density_var4, color = :purple, linewidth = 2, label = "δ = 10.0")
scatter!(ax1, dn_square5, density_var5, color = :orange, linewidth = 2, label = "δ = 0.5")

axislegend(ax1, position = :rt)
display(fig)

# Third plot: Density as a function of σ
σ_range = 0:0.01:1
density_values = [compute_density(dn_square1[2], dn_square1[1], σ) for σ in σ_range]

fig = Figure(size=(800, 600))
ax1 = Axis(fig[1, 1], xlabel = "σ", ylabel = "Density", title = "Density as a Function of σ")

lines!(ax1, σ_range, density_values, color = :blue, linewidth = 2, label = "d[2] and d[1]")

axislegend(position = :rt)
display(fig)


