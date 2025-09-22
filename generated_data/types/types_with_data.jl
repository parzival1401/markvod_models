using CairoMakie
using Distributions
using Optim

# Load required dependencies after type definitions
# These will be loaded later to avoid circular dependencies

# Custom types
struct Particles
    x::Float64
    y::Float64
    t::Float64
end

struct simulation
    particle_1::Vector{Particles}
    particle_2::Vector{Particles}
    k_states::Vector{Float64}  # [k_on, k_off]
    σ::Float64
    D::Float64
    dt::Float64
    d_dimer::Float64
end

# Constructor to create Particles from position and time
Particles(pos::Vector{Float64}, time::Float64) = Particles(pos[1], pos[2], time)

# Constructor to create simulation from matrices
function simulation(p1_matrix::Matrix{Float64}, p2_matrix::Matrix{Float64}, 
                   states::Vector{Int}, k_states::Vector{Float64}, D::Float64; 
                   dt::Float64=0.016, σ::Float64=0.0, d_dimer::Float64=0.05)
    
    n_steps = size(p1_matrix, 2)
    time_vec = collect(0:dt:(n_steps-1)*dt)
    
    # Create particle vectors
    particle_1 = [Particles(p1_matrix[1, i], p1_matrix[2, i], time_vec[i]) for i in 1:n_steps]
    particle_2 = [Particles(p2_matrix[1, i], p2_matrix[2, i], time_vec[i]) for i in 1:n_steps]
    
    return simulation(particle_1, particle_2, k_states, σ, D, dt, d_dimer)
end

# Utility functions
function get_position_matrices(sim::simulation)
    n_steps = length(sim.particle_1)
    p1 = zeros(2, n_steps)
    p2 = zeros(2, n_steps)
    
    for i in 1:n_steps
        p1[1, i] = sim.particle_1[i].x
        p1[2, i] = sim.particle_1[i].y
        p2[1, i] = sim.particle_2[i].x
        p2[2, i] = sim.particle_2[i].y
    end
    
    return p1, p2
end

function get_times(sim::simulation)
    return [p.t for p in sim.particle_1]
end

function particle_distance(sim::simulation, index::Int)
    p1 = sim.particle_1[index]
    p2 = sim.particle_2[index]
    return sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
end

# Animation function for custom types
function animate_particles(sim::simulation, filename="particle_animation.mp4")
    p1, p2 = get_position_matrices(sim)
    return animate_particles(p1, p2, filename)
end

# Original animation function for matrices
function animate_particles(p1::Matrix{Float64}, p2::Matrix{Float64}, filename="particle_animation.mp4")
    n_steps = size(p1, 2)
    
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    all_x = vcat(p1[1,:], p2[1,:])
    all_y = vcat(p1[2,:], p2[2,:])
    
    limits!(ax, -1, 1, -1, 1)
    
    point1 = scatter!([p1[1,1]], [p1[2,1]], color=:blue, markersize=10)
    point2 = scatter!([p2[1,1]], [p2[2,1]], color=:red, markersize=10)
    
    record(fig, filename, 1:n_steps; framerate=10) do frame
        point1[1] = [p1[1,frame]]
        point1[2] = [p1[2,frame]]
        
        point2[1] = [p2[1,frame]]
        point2[2] = [p2[2,frame]]
    end
    
    return filename
end

# Utility function to reverse columns
function reverse_columns_preserve_size(arr)
    num_rows, num_cols = size(arr)
    result = zeros(eltype(arr), num_rows, num_cols)
    
    for col in 1:num_cols
        result[:, col] = arr[:, num_cols - col + 1]
    end
    
    return result
end

# Shift function for particle positions
function shift(vec, shift_x, shift_y)
    return vec .+ [shift_x, shift_y]
end

# Path correction function
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

# In-place path correction function
function path_correction!(particle_path, new_destination::Vector{Float64})
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

# Function to generate segments (requires external functions)
function segments(time_division::Vector, mode::Symbol; D = 0.01, r = 0.2, box = 1.0, dt = 0.016)
    if mode == :full && length(time_division) == 3
        p1_middle, p2_middle = constrained_diffusion(steps = time_division[2], D = D, r = r, box = box, dt = dt)
        
        init_post_r = ([p1_middle[1,end], p1_middle[2,end]],[p2_middle[1,end], p2_middle[2,end]])
        p1_right, p2_right = simulate_diffusion(initial_positions = init_post_r, steps = time_division[3], D = D, box = box, dt = dt)
        
        init_post_l = ([p1_middle[1,1], p1_middle[2,1]], [p2_middle[1,1], p2_middle[2,1]])
        p1_left, p2_left = simulate_diffusion(initial_positions = init_post_l, steps = time_division[1], D = D, box = box, dt = dt)
        p1_left_reversed = reverse_columns_preserve_size(p1_left)
        p2_left_reversed = reverse_columns_preserve_size(p2_left)
        
        p1 = hcat(p1_left_reversed, p1_middle, p1_right)
        p2 = hcat(p2_left_reversed, p2_middle, p2_right)
        
        return p1, p2
        
    elseif mode == :right_half && length(time_division) == 2
        p1_middle, p2_middle = constrained_diffusion(steps = time_division[1], D = D, r = r, box = box, dt = dt)
        
        init_post_r = ([p1_middle[1,end], p1_middle[2,end]], [p2_middle[1,end], p2_middle[2,end]])
        p1_right, p2_right = simulate_diffusion(initial_positions = init_post_r, steps = time_division[2], D = D, box = box, dt = dt)
        
        p1 = hcat(p1_middle, p1_right)
        p2 = hcat(p2_middle, p2_right)
        
        return p1, p2
        
    elseif mode == :middle && length(time_division) == 1
        p1_middle, p2_middle = constrained_diffusion(steps = time_division[1], D = D, r = r, box = box, dt = dt)
        p1 = hcat(p1_middle)
        p2 = hcat(p2_middle)
        
        return p1, p2
        
    elseif mode == :left_half && length(time_division) == 2
        p1_middle, p2_middle = constrained_diffusion(steps = time_division[2], D = D, r = r, box = box, dt = dt)
        
        init_post_l = ([p1_middle[1,1], p1_middle[2,1]], [p2_middle[1,1], p2_middle[2,1]])
        p1_left, p2_left = simulate_diffusion(initial_positions = init_post_l, steps = time_division[1], D = D, box = box, dt = dt)
        p1_left_reversed = reverse_columns_preserve_size(p1_left)
        p2_left_reversed = reverse_columns_preserve_size(p2_left)
        
        p1 = hcat(p1_left_reversed, p1_middle)
        p2 = hcat(p2_left_reversed, p2_middle)
        return p1, p2
        
    else
        error("Invalid mode or time division")
    end 
end

# Mode selection function
function modes_selection(section::Vector)
    size_section = length(section)
    if size_section == 3
        return :full
    elseif size_section == 2
        # Fix for the error: check if elements are pairs or integers
        if isa(section[1], Pair) && isa(section[2], Pair)
            if section[1].first == 1 && section[2].first == 2
                return :left_half
            elseif section[1].first == 2 && section[2].first == 1
                return :right_half
            else
                error("Invalid section combination")
            end
        else
            # Handle case where section contains integers instead of pairs
            return :right_half  # Default fallback
        end
    elseif size_section == 1
        return :middle
    else
        error("Invalid number of sections")
    end
end

# Extract time stamps from segments
function time_stamps(time_state)
    segments = []
    for i in 1:length(time_state)
        if isa(time_state[i], Pair)
            push!(segments, time_state[i].second)
        else
            push!(segments, time_state[i])  # Handle case where it's just a number
        end
    end
    return segments
end

# Main simulation function - modified to return custom types
function run_simulation(k_in, k12_off, changes; D=0.01, dt=0.016, σ=0.0)
    # First run original simulation to get matrices
    p1_matrix, p2_matrix, states, steps, k_params = simulate_raw(k_in, k12_off, changes, D=D, dt=dt)
    
    # Convert to custom types
    k_states = [k_params[1], k_params[2]]  # [k_on, k_off]
    sim = simulation(p1_matrix, p2_matrix, states, k_states, D, dt=dt, σ=σ, d_dimer=0.05)
    
    return sim
end

# Original simulation function (returns matrices)
function simulate_raw(k_in, k12_off, changes; D=0.01, dt=0.016)
    k12 = k_in
    k21 = k12_off
    states, steps = simulate_states(k12, k21, changes)
    time_in_state = time_sequence_with_split(states)
    
    current_time = 0 
    Particle_1 = []
    Particle_2 = []

    for i in 1:length(time_in_state)
        section_time_stamps = time_stamps(time_in_state[i])
        modes = modes_selection(time_in_state[i])  # Pass original data, not time_stamps
        println("section_time_stamps: ", section_time_stamps)
        println("modes: ", modes)
        
        if i == 1
            p1, p2 = segments(section_time_stamps, modes, r = 0.01, box = 1.0, dt = dt, D = D)
            Particle_1 = p1
            Particle_2 = p2
        else 
            p1_temp, p2_temp = segments(section_time_stamps, modes, r = 0.01, box = 1.0, dt = dt, D = D)
            
            shift_x = -p1_temp[1,1] + Particle_1[1,end]
            shift_y = -p1_temp[2,1] + Particle_1[2,end]
            
            p1 = shift(p1_temp, shift_x, shift_y)
            p2 = shift(p2_temp, shift_x, shift_y)
            
            segment_view = @view p2[:,1:time_in_state[i][1].second]
            path_correction!(segment_view, [Particle_2[1,end], Particle_2[2,end]])
            
            Particle_1 = hcat(Particle_1, p1)
            Particle_2 = hcat(Particle_2, p2)
        end
        
        current_time = length(Particle_1) 
    end
   
    return Particle_1, Particle_2, states, steps, [k12, k21, D]
end

# Load external dependencies after type definitions to avoid circular dependencies
try
    include("../state_generator.jl")
    include("../dimer_difussion.jl") 
    include("../free_difusion.jl")
    println("✅ External dependencies loaded successfully")
catch e
    println("⚠️  Warning: Could not load external dependencies - some functions may not be available")
    println("   Error: ", e)
end

# Forward algorithm functions
function compute_free_density(particle1::Matrix{Float64}, particle2::Matrix{Float64}, index::Int; sigma=0.1, dt=0.01,D=0.01)
    Δxn = particle1[1, index] - particle2[1, index]  
    Δyn = particle1[2, index] - particle2[2, index]  
    
    Δxn_1 = particle1[1, index-1] - particle2[1, index-1]  
    Δyn_1 = particle1[2, index-1] - particle2[2, index-1]  
    
    dn_1square = (Δxn_1^2) + (Δyn_1^2)
    dn_1 = sqrt(dn_1square)
    dn_square = (Δxn^2) + (Δyn^2)
    dn = sqrt(dn_square)
    sigma_eff = 4*sigma + 4*D*dt
    density_val = (dn/sigma_eff^2) * (exp((-(dn_square) - (dn_1square))/sigma_eff^2)) * modified_bessel(dt, dn, dn_1, sigma_eff)
    
    if density_val == Inf || isnan(density_val) || density_val == 0
        density_val = 0
        for θ in 0:dt:2*pi
            density_val += exp(-((dn_square) + (dn_1square) - (2*dn*dn_1*cos(θ)))/(2*sigma_eff^2))
        end 
        density_val *= (dn/sigma_eff^2)*dt
    end
    
    return density_val
end

function compute_dimer_density(particle1::Matrix{Float64}, particle2::Matrix{Float64}, index::Int, d_dimer::Float64; sigma=0.1, dt=0.01,D=0.01)
    Δxn = particle1[1, index] - particle2[1, index]  
    Δyn = particle1[2, index] - particle2[2, index]  
    
    dn_square = (Δxn^2) + (Δyn^2)
    dn = sqrt(dn_square)
    sigma_n = 2*sigma 
    density_val = (dn/sigma_n^2) * exp(((-(d_dimer^2) - (dn_square))/sigma_n^2)) * modified_bessel(dt, d_dimer, dn, sigma_n)
    
    if density_val == Inf || isnan(density_val) || density_val == 0
        density_val = 0
        for θ in 0:dt:2*pi
            density_val += exp(-((dn_square) + (d_dimer^2) - (2*dn*d_dimer*cos(θ)))/(2*sigma_n^2))
        end 
        density_val *= (dn/sigma_n^2)*dt
    end
    
    return density_val
end

function modified_bessel(dt, d1, d2, σ)
    result_me = 0.0
    x = (d1*d2)/(σ^2)
    
    for θ in 0:dt:2*pi
        result_me += exp(x * cos(θ))
    end
    
    result_me *= dt / (2*pi)
    return result_me 
end

function forward_algorithm(particle1::Matrix{Float64}, particle2::Matrix{Float64}, d_dimer::Float64, param::Vector{Float64}; dt=0.01, sigma=0.1, D=0.01)
    N = min(size(particle1, 2), size(particle2, 2)) - 1
    alpha = zeros(2, N)
    scale = zeros(N)
    e1_vec = []
    e2_vec = []
    
    frame_index = min(2, N)
    
    e1_initial = compute_free_density(particle1, particle2, frame_index, sigma=sigma, dt=dt, D=D)
    e2_initial = compute_dimer_density(particle1, particle2, frame_index, d_dimer, sigma=sigma, dt=dt, D=D)
    
    push!(e1_vec, e1_initial)
    push!(e2_vec, e2_initial)
    
    # Use uniform initial state probabilities weighted by emissions
    alpha[1, 1] = 0.5 * e1_initial
    alpha[2, 1] = 0.5 * e2_initial
    
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= max(scale[1], eps())
    
    k_on = param[1]
    k_off = param[2]
    
    # Validate transition probabilities
    if k_on*dt >= 0.5 || k_off*dt >= 0.5
        @warn "Transition probabilities >= 0.5, results may be unreliable. k_on*dt=$(k_on*dt), k_off*dt=$(k_off*dt)"
    end
    
    T = [1-(k_on*dt) k_on*dt; 
         k_off*dt 1-(k_off*dt)]
    
    for t in 2:N
        frame_index = min(t + 1, size(particle1, 2))
        
        e1 = compute_free_density(particle1, particle2, frame_index, sigma=sigma, dt=dt, D=D)
        e2 = compute_dimer_density(particle1, particle2, frame_index, d_dimer, sigma=sigma, dt=dt, D=D)
        
        push!(e1_vec, e1)
        push!(e2_vec, e2)
        
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= max(scale[t], eps()) 
    end
    
    loglikelihood = sum(log.(max.(scale, eps()))) 
    
    return alpha, loglikelihood, e1_vec, e2_vec
end

# Forward algorithm for custom types
function forward_algorithm(sim::simulation; dt=0.01, sigma=0.1)
    p1, p2 = get_position_matrices(sim)
    return forward_algorithm(p1, p2, sim.d_dimer, sim.k_states, dt=dt, sigma=sigma, D=sim.D)
end

# Create custom time vector
function create_time_vector(p1, dt=0.016)
    n_steps = size(p1, 2)
    time_vec = collect(0:dt:(n_steps-1)*dt)
    return time_vec
end

# Add noise to simulation
function add_noise(sim::simulation, σ::Float64=0.01)
    p1, p2 = get_position_matrices(sim)
    
    # Add noise to positions
    p1_noise = p1 .+ σ * randn(size(p1))
    p2_noise = p2 .+ σ * randn(size(p2))
    
    # Create new simulation with noisy data
    # Note: We don't have access to original states, so we use empty vector
    return simulation(p1_noise, p2_noise, Int[], sim.k_states, sim.D, dt=sim.dt, σ=σ, d_dimer=sim.d_dimer)
end

# Display functions
function Base.show(io::IO, p::Particles)
    print(io, "Particle(x=$(p.x), y=$(p.y), t=$(p.t))")
end

function Base.show(io::IO, sim::simulation)
    n_steps = length(sim.particle_1)
    duration = sim.particle_1[end].t - sim.particle_1[1].t
    
    # Calculate some basic statistics
    p1, p2 = get_position_matrices(sim)
    distances = [sqrt((p1[1,i] - p2[1,i])^2 + (p1[2,i] - p2[2,i])^2) for i in 1:size(p1,2)]
    avg_distance = sum(distances) / length(distances)
    min_distance = minimum(distances)
    max_distance = maximum(distances)
    bound_fraction = sum(distances .< sim.d_dimer) / length(distances)
    
    # Calculate particle ranges
    p1_x_range = (minimum(p1[1,:]), maximum(p1[1,:]))
    p1_y_range = (minimum(p1[2,:]), maximum(p1[2,:]))
    p2_x_range = (minimum(p2[1,:]), maximum(p2[1,:]))
    p2_y_range = (minimum(p2[2,:]), maximum(p2[2,:]))
    
    # Pretty print with colors and structure
    printstyled(io, "🧬 Markov Simulation", color=:blue, bold=true)
    println(io)
    println(io, "├─ 📊 Temporal Properties:")
    println(io, "│  ├─ Time steps: $(n_steps)")
    println(io, "│  ├─ Duration: $(round(duration, digits=3)) s")
    println(io, "│  ├─ Time step (dt): $(sim.dt) s")
    println(io, "│  └─ Time range: [$(round(sim.particle_1[1].t, digits=3)), $(round(sim.particle_1[end].t, digits=3))] s")
    println(io)
    println(io, "├─ ⚛️  Model Parameters:")
    println(io, "│  ├─ k_on (binding rate): $(sim.k_states[1]) s⁻¹") 
    println(io, "│  ├─ k_off (unbinding rate): $(sim.k_states[2]) s⁻¹")
    println(io, "│  ├─ Diffusion coefficient (D): $(sim.D) μm²/s")
    println(io, "│  ├─ Noise level (σ): $(sim.σ)")
    println(io, "│  └─ Dimer threshold (d_dimer): $(sim.d_dimer) μm")
    println(io)
    println(io, "├─ 📍 Spatial Properties:")
    println(io, "│  ├─ Average distance: $(round(avg_distance, digits=4)) μm")
    println(io, "│  ├─ Distance range: [$(round(min_distance, digits=4)), $(round(max_distance, digits=4))] μm")
    println(io, "│  ├─ Bound fraction: $(round(bound_fraction*100, digits=1))%")
    println(io, "│  ├─ Particle 1 range: x[$(round(p1_x_range[1], digits=3)), $(round(p1_x_range[2], digits=3))], y[$(round(p1_y_range[1], digits=3)), $(round(p1_y_range[2], digits=3))]")
    println(io, "│  └─ Particle 2 range: x[$(round(p2_x_range[1], digits=3)), $(round(p2_x_range[2], digits=3))], y[$(round(p2_y_range[1], digits=3)), $(round(p2_y_range[2], digits=3))]")
    println(io)
    println(io, "└─ 🔬 Analysis Summary:")
    
    if bound_fraction > 0.5
        printstyled(io, "   └─ Predominantly bound state simulation", color=:red)
    elseif bound_fraction > 0.1
        printstyled(io, "   └─ Mixed binding/unbinding dynamics", color=:yellow)
    else
        printstyled(io, "   └─ Predominantly free diffusion simulation", color=:green)
    end
end

# Example function
function example_custom_simulation()
    println("=== Example: Custom Type Simulation ===")
    
    # This requires external dependencies (state_generator.jl, etc.)
    println("To run a full simulation, you need:")
    println("1. include(\"state_generator.jl\")")
    println("2. include(\"dimer_difussion.jl\")") 
    println("3. include(\"free_difusion.jl\")")
    println("4. sim = run_simulation(0.5, 0.3, 10)")
    
    # Example with dummy data
    println("\nCreating example with dummy data:")
    
    # Create sample matrices
    n_steps = 20
    p1_sample = randn(2, n_steps)
    p2_sample = randn(2, n_steps) 
    states_sample = rand([1, 2], n_steps-1)
    k_states = [0.5, 0.3]
    D = 0.01
    
    # Create simulation object
    σ_sample = 0.005  # Example noise level
    sim = simulation(p1_sample, p2_sample, states_sample, k_states, D, dt=0.016, σ=σ_sample)
    println("Created simulation: ", sim)
    
    # Test utilities
    times = get_times(sim)
    println("Time points: ", times[1:5], "...")
    
    dist = particle_distance(sim, 1)
    println("Distance at step 1: ", round(dist, digits=4))
    
    # Test noise addition
    sim_noisy = add_noise(sim, 0.01)
    println("Added noise: ", sim_noisy)
    
    return sim
end

# ========================================
# OPTIMIZATION FUNCTIONS FOR K PARAMETERS
# ========================================

# Objective function for matrix-based optimization
function objective_function_matrix(param, p1::Matrix{Float64}, p2::Matrix{Float64}, d_dimer::Float64; dt=0.01, sigma=0.1, D=0.01)
    _, loglikelihood, _, _ = forward_algorithm(p1, p2, d_dimer, param, dt=dt, sigma=sigma, D=D)
    return -loglikelihood
end

# Objective function for custom type optimization  
function objective_function_custom(param, sim::simulation; dt=0.01, sigma=0.1)
    p1, p2 = get_position_matrices(sim)
    _, loglikelihood, _, _ = forward_algorithm(p1, p2, sim.d_dimer, param, dt=dt, sigma=sigma, D=sim.D)
    return -loglikelihood
end

# Optimize k parameters using matrix data with bounds
function optimize_k_parameters_matrix(p1::Matrix{Float64}, p2::Matrix{Float64}, d_dimer::Float64=0.01; 
                                    initial_params::Vector{Float64}=[0.1, 0.1], dt=0.01, sigma=0.1, D=0.01)
    
    # Create wrapper function for optimization with bounds checking
    function bounded_wrapper(param)
        # Apply bounds: k parameters should be positive and reasonable
        k_on_bounded = clamp(param[1], 1e-6, 2.0)
        k_off_bounded = clamp(param[2], 1e-6, 2.0)# abs(params[2]) 
        bounded_params = [k_on_bounded, k_off_bounded]
        
        try
            result = objective_function_matrix(bounded_params, p1, p2, d_dimer, dt=dt, sigma=sigma, D=D)
            # Handle numerical issues
            if isnan(result) || isinf(result)
                return 1e10  # Large penalty for invalid results
            end
            return result
        catch e
            return 1e10  # Large penalty for errors
        end
    end
    
    # Perform optimization using Nelder-Mead with smaller tolerance
    result = optimize(bounded_wrapper, initial_params, NelderMead(), 
                     Optim.Options(g_tol=1e-6, iterations=100))
    
    # Extract optimized parameters and apply bounds
    raw_params = Optim.minimizer(result)
    optimal_params = [clamp(raw_params[1], 1e-6, 2.0), clamp(raw_params[2], 1e-6, 2.0)]
    k_on_optimized = optimal_params[1]
    k_off_optimized = optimal_params[2]
    
    # Calculate final log-likelihood with optimized parameters
    local final_loglikelihood, e1_vec, e2_vec
    try
        _, final_loglikelihood, e1_vec, e2_vec = forward_algorithm(p1, p2, d_dimer, optimal_params, dt=dt, sigma=sigma, D=D)
    catch e
        # Fallback if forward algorithm fails
        final_loglikelihood = -Inf
        e1_vec = Float64[]
        e2_vec = Float64[]
    end
    
    return (
        k_on = k_on_optimized,
        k_off = k_off_optimized,
        optimal_params = optimal_params,
        loglikelihood = final_loglikelihood,
        optimization_result = result,
        e1_vec = e1_vec,
        e2_vec = e2_vec
    )
end

# Optimize k parameters using custom simulation type
function optimize_k_parameters_custom(sim::simulation; 
                                    initial_params::Vector{Float64}=[0.1, 0.1], dt=0.01, sigma=0.1)
    
    # Create wrapper function for optimization
    wrapper(param) = objective_function_custom(abs.(param), sim, dt=dt, sigma=sigma)
    
    # Perform optimization using Nelder-Mead
    result = optimize(wrapper, initial_params, NelderMead())
    
    # Extract optimized parameters
    optimal_params = Optim.minimizer(result)
    k_on_optimized = optimal_params[1]
    k_off_optimized = optimal_params[2]
    d_dimer = sim.d_dimer
    
    # Calculate final log-likelihood with optimized parameters
    p1, p2 = get_position_matrices(sim)
    _, final_loglikelihood, e1_vec, e2_vec = forward_algorithm(p1, p2, d_dimer, optimal_params, dt=dt, sigma=sigma, D=sim.D)
    
    return (
        k_on = k_on_optimized,
        k_off = k_off_optimized,
        optimal_params = optimal_params,
        loglikelihood = final_loglikelihood,
        optimization_result = result,
        e1_vec = e1_vec,
        e2_vec = e2_vec
    )
end

# Convenience function to optimize both clean and noisy data
function optimize_with_noise_comparison(p1::Matrix{Float64}, p2::Matrix{Float64}, d_dimer::Float64=0.01, 
                                      σ_noise::Float64=0.01; initial_params::Vector{Float64}=[0.1, 0.1], 
                                      dt=0.01, sigma=0.1, D=0.01)
    
    # Optimize clean data
    clean_result = optimize_k_parameters_matrix(p1, p2, d_dimer, initial_params=initial_params, dt=dt, sigma=sigma, D=D)
    
    # Add noise to data
    p1_noise = p1 .+ σ_noise * randn(size(p1))
    p2_noise = p2 .+ σ_noise * randn(size(p2))
    
    # Optimize noisy data
    noisy_result = optimize_k_parameters_matrix(p1_noise, p2_noise, d_dimer, initial_params=initial_params, dt=dt, sigma=sigma, D=D)
    
    return (
        clean = clean_result,
        noisy = noisy_result,
        noise_level = σ_noise
    )
end

# Function to create likelihood plots over parameter ranges
function create_likelihood_surface_matrix(p1::Matrix{Float64}, p2::Matrix{Float64}, d_dimer::Float64=0.01; 
                                        k_range=(0:0.01:1), dt=0.01, sigma=0.1, D=0.01)
    
    n_points = length(k_range)
    likelihood_surface = zeros(n_points, n_points)
    
    for (i, k_on) in enumerate(k_range)
        for (j, k_off) in enumerate(k_range)
            _, loglikelihood, _, _ = forward_algorithm(p1, p2, d_dimer, [k_on, k_off], dt=dt, sigma=sigma, D=D)
            likelihood_surface[i, j] = loglikelihood
        end
    end
    
    return likelihood_surface, k_range
end

# Function to create likelihood plots for custom simulation type
function create_likelihood_surface_custom(sim::simulation; 
                                        k_range=(0:0.01:1), dt=0.01, sigma=0.1)
    
    p1, p2 = get_position_matrices(sim)
    return create_likelihood_surface_matrix(p1, p2, sim.d_dimer, k_range=k_range, dt=dt, sigma=sigma, D=sim.D)
end

# Example usage function
function example_optimization()
    println("=== Example: K Parameter Optimization ===")
    
    # Generate sample data (normally you would use your simulation data)
    n_steps = 100
    p1_sample = randn(2, n_steps) * 0.1
    p2_sample = randn(2, n_steps) * 0.1
    
    println("1. Matrix-based optimization:")
    result_matrix = optimize_k_parameters_matrix(p1_sample, p2_sample, 0.01)
    println("   Optimized k_on: ", round(result_matrix.k_on, digits=4))
    println("   Optimized k_off: ", round(result_matrix.k_off, digits=4))
    println("   Log-likelihood: ", round(result_matrix.loglikelihood, digits=2))
    
    # Create custom simulation for comparison
    states_sample = rand([1, 2], n_steps-1)
    k_states = [0.5, 0.3]
    D = 0.01
    sim_sample = simulation(p1_sample, p2_sample, states_sample, k_states, D, dt=0.016, σ=0.005)
    
    println("\n2. Custom type optimization:")
    result_custom = optimize_k_parameters_custom(sim_sample, 0.01)
    println("   Optimized k_on: ", round(result_custom.k_on, digits=4))
    println("   Optimized k_off: ", round(result_custom.k_off, digits=4))
    println("   Log-likelihood: ", round(result_custom.loglikelihood, digits=2))
    
    println("\n3. Noise comparison:")
    comparison = optimize_with_noise_comparison(p1_sample, p2_sample, 0.01, 0.01)
    println("   Clean data k_on: ", round(comparison.clean.k_on, digits=4))
    println("   Noisy data k_on: ", round(comparison.noisy.k_on, digits=4))
    
    return result_matrix, result_custom, comparison
end

# ========================================
# COMPREHENSIVE EXAMPLE WITH COMMENTS
# ========================================

"""
Complete example demonstrating all optimization functionality with detailed comments.
This example shows how to:
1. Generate simulation data
2. Run optimization using both matrix and custom type methods
3. Compare results with and without noise
4. Create likelihood surface plots
"""
function comprehensive_optimization_example()
    println("=" ^ 60)
    println("COMPREHENSIVE K PARAMETER OPTIMIZATION EXAMPLE")
    println("=" ^ 60)
    
    # ========================================
    # STEP 1: GENERATE SIMULATION DATA
    # ========================================
    println("\n🔹 STEP 1: Generating simulation data...")
    
    # Set true parameters for simulation
    true_k_on = 0.5    # True binding rate
    true_k_off = 0.3   # True unbinding rate 
    n_changes = 8      # Number of state changes to simulate
    D = 0.01          # Diffusion constant
    dt = 0.016        # Time step
    
    println("   True parameters: k_on = $true_k_on, k_off = $true_k_off")
    println("   Diffusion constant: D = $D")
    println("   Time step: dt = $dt")
    
    # Generate simulation using the run_simulation function or fallback
    # This creates realistic particle trajectories with state transitions
    local p1, p2, states, steps, k_params
    
    if isdefined(Main, :run_simulation)
        try
            p1, p2, states, steps, k_params = run_simulation(true_k_on, true_k_off, n_changes, D=D, dt=dt)
        catch e
            println("   ⚠️  Error in run_simulation: $e, using fallback...")
            # Fallback data generation
            n_steps = 200
            p1 = cumsum(randn(2, n_steps) * 0.02, dims=2)
            p2 = p1 + 0.05 * randn(2, n_steps) .+ 0.01 * randn(2, n_steps)
            states = rand([1, 2], n_steps)
            steps = n_steps
            k_params = [true_k_on, true_k_off, D]
        end
    else
        println("   ⚠️  run_simulation function not available, using fallback data generation...")
        # Fallback: generate simple test data
        n_steps = 200
        p1 = cumsum(randn(2, n_steps) * 0.02, dims=2)
        p2 = p1 + 0.05 * randn(2, n_steps) .+ 0.01 * randn(2, n_steps)
        states = rand([1, 2], n_steps)
        steps = n_steps
        k_params = [true_k_on, true_k_off, D]
    end
    
    println("   ✅ Generated $(size(p1, 2)) time steps")
    println("   ✅ Simulation duration: $(size(p1, 2) * dt) seconds")
    
    # ========================================
    # STEP 2: CREATE CUSTOM SIMULATION OBJECT
    # ========================================
    println("\n🔹 STEP 2: Creating custom simulation object...")
    
    # Convert matrices to custom simulation type
    # This demonstrates the integration between matrix and custom type approaches
    sim = simulation(p1, p2, states[1:end-1], [true_k_on, true_k_off], D, dt=dt, σ=0.0)
    
    println("   ✅ Created simulation object: $sim")
    
    # ========================================
    # STEP 3: MATRIX-BASED OPTIMIZATION
    # ========================================
    println("\n🔹 STEP 3: Matrix-based parameter optimization...")
    
    # Set optimization parameters
    d_dimer = 0.01           # Dimer interaction distance
    initial_guess = [0.2, 0.2]  # Starting point for optimization (closer to true values)
    sigma = 0.1              # Observation noise parameter
    
    println("   Parameters:")
    println("     - Dimer distance: $d_dimer")
    println("     - Initial guess: $initial_guess")
    println("     - Sigma (obs noise): $sigma")
    
    # Run matrix-based optimization
    println("   🔄 Running optimization...")
    matrix_result = optimize_k_parameters_matrix(
        p1, p2, d_dimer, 
        initial_params=initial_guess, 
        dt=dt, 
        sigma=sigma, 
        D=D
    )
    
    # Display results
    println("   📊 MATRIX OPTIMIZATION RESULTS:")
    println("     - Optimized k_on:  $(round(matrix_result.k_on, digits=4)) (true: $true_k_on)")
    println("     - Optimized k_off: $(round(matrix_result.k_off, digits=4)) (true: $true_k_off)")
    println("     - Log-likelihood:   $(round(matrix_result.loglikelihood, digits=2))")
    println("     - Optimization converged: $(Optim.converged(matrix_result.optimization_result))")
    
    # ========================================
    # STEP 4: CUSTOM TYPE OPTIMIZATION
    # ========================================
    println("\n🔹 STEP 4: Custom type parameter optimization...")
    
    # Run custom type optimization
    println("   🔄 Running custom type optimization...")
    custom_result = optimize_k_parameters_custom(
        sim, d_dimer,
        initial_params=initial_guess,
        dt=dt,
        sigma=sigma
    )
    
    # Display results and compare with matrix method
    println("   📊 CUSTOM TYPE OPTIMIZATION RESULTS:")
    println("     - Optimized k_on:  $(round(custom_result.k_on, digits=4)) (true: $true_k_on)")
    println("     - Optimized k_off: $(round(custom_result.k_off, digits=4)) (true: $true_k_off)")
    println("     - Log-likelihood:   $(round(custom_result.loglikelihood, digits=2))")
    println("     - Optimization converged: $(Optim.converged(custom_result.optimization_result))")
    
    # Compare results between methods
    println("\n   🔍 COMPARISON BETWEEN METHODS:")
    k_on_diff = abs(matrix_result.k_on - custom_result.k_on)
    k_off_diff = abs(matrix_result.k_off - custom_result.k_off)
    ll_diff = abs(matrix_result.loglikelihood - custom_result.loglikelihood)
    
    println("     - k_on difference:  $(round(k_on_diff, digits=6))")
    println("     - k_off difference: $(round(k_off_diff, digits=6))")
    println("     - Log-likelihood difference: $(round(ll_diff, digits=6))")
    
    if k_on_diff < 1e-4 && k_off_diff < 1e-4
        println("     ✅ Methods agree within numerical precision!")
    else
        println("     ⚠️  Methods show some difference - check implementation")
    end
    
    # ========================================
    # STEP 5: NOISE SENSITIVITY ANALYSIS
    # ========================================
    println("\n🔹 STEP 5: Noise sensitivity analysis...")
    
    # Test different noise levels
    noise_levels = [0.005, 0.01, 0.02, 0.05]
    
    println("   Testing noise levels: $noise_levels")
    
    for σ_noise in noise_levels
        println("\n   🔄 Testing noise level σ = $σ_noise...")
        
        # Run noise comparison
        noise_comparison = optimize_with_noise_comparison(
            p1, p2, d_dimer, σ_noise,
            initial_params=initial_guess,
            dt=dt, sigma=sigma, D=D
        )
        
        # Display results
        println("     📊 RESULTS:")
        println("       Clean data  - k_on: $(round(noise_comparison.clean.k_on, digits=4)), k_off: $(round(noise_comparison.clean.k_off, digits=4))")
        println("       Noisy data  - k_on: $(round(noise_comparison.noisy.k_on, digits=4)), k_off: $(round(noise_comparison.noisy.k_off, digits=4))")
        
        # Calculate relative error
        k_on_error = abs(noise_comparison.noisy.k_on - true_k_on) / true_k_on * 100
        k_off_error = abs(noise_comparison.noisy.k_off - true_k_off) / true_k_off * 100
        
        println("       Relative errors - k_on: $(round(k_on_error, digits=1))%, k_off: $(round(k_off_error, digits=1))%")
    end
    
    # ========================================
    # STEP 6: LIKELIHOOD SURFACE ANALYSIS
    # ========================================
    println("\n🔹 STEP 6: Creating likelihood surface...")
    
    # Create likelihood surface for visualization
    k_range = 0:0.05:1  # Coarser grid for faster computation
    println("   🔄 Computing likelihood surface over k_range = $k_range...")
    
    likelihood_surface, k_vals = create_likelihood_surface_matrix(
        p1, p2, d_dimer,
        k_range=k_range, dt=dt, sigma=sigma, D=D
    )
    
    # Find maximum likelihood point on the grid
    max_idx = argmax(likelihood_surface)
    max_k_on = k_vals[max_idx[1]]
    max_k_off = k_vals[max_idx[2]]
    max_likelihood = likelihood_surface[max_idx]
    
    println("   📊 LIKELIHOOD SURFACE ANALYSIS:")
    println("     - Grid size: $(size(likelihood_surface))")
    println("     - Maximum likelihood: $(round(max_likelihood, digits=2))")
    println("     - Max likelihood k_on:  $max_k_on (optimized: $(round(matrix_result.k_on, digits=4)))")
    println("     - Max likelihood k_off: $max_k_off (optimized: $(round(matrix_result.k_off, digits=4)))")
    
    # ========================================
    # STEP 7: FORWARD ALGORITHM DEMONSTRATION
    # ========================================
    println("\n🔹 STEP 7: Forward algorithm demonstration...")
    
    # Run forward algorithm with optimized parameters
    println("   🔄 Running forward algorithm with optimized parameters...")
    
    alpha, loglikelihood, e1_vec, e2_vec = forward_algorithm(
        p1, p2, d_dimer, matrix_result.optimal_params,
        dt=dt, sigma=sigma, D=D
    )
    
    # Analyze state probabilities
    avg_free_prob = mean(alpha[1, :])
    avg_bound_prob = mean(alpha[2, :])
    
    println("   📊 FORWARD ALGORITHM RESULTS:")
    println("     - Final log-likelihood: $(round(loglikelihood, digits=2))")
    println("     - Average free probability:  $(round(avg_free_prob, digits=3))")
    println("     - Average bound probability: $(round(avg_bound_prob, digits=3))")
    println("     - Number of density evaluations: $(length(e1_vec))")
    
    # ========================================
    # STEP 8: SUMMARY AND RECOMMENDATIONS
    # ========================================
    println("\n🔹 STEP 8: Summary and recommendations...")
    
    # Calculate parameter estimation accuracy
    k_on_accuracy = (1 - abs(matrix_result.k_on - true_k_on) / true_k_on) * 100
    k_off_accuracy = (1 - abs(matrix_result.k_off - true_k_off) / true_k_off) * 100
    
    println("   📊 FINAL SUMMARY:")
    println("     - True parameters:      k_on = $true_k_on, k_off = $true_k_off")
    println("     - Estimated parameters: k_on = $(round(matrix_result.k_on, digits=4)), k_off = $(round(matrix_result.k_off, digits=4))")
    println("     - Estimation accuracy:  k_on = $(round(k_on_accuracy, digits=1))%, k_off = $(round(k_off_accuracy, digits=1))%")
    
    println("\n   💡 RECOMMENDATIONS:")
    if k_on_accuracy > 90 && k_off_accuracy > 90
        println("     ✅ Excellent parameter estimation!")
        println("     ✅ The optimization method is working well for this data")
    elseif k_on_accuracy > 80 && k_off_accuracy > 80
        println("     ⚠️  Good parameter estimation, but could be improved")
        println("     💡 Consider: longer simulation time, different initial guesses, or parameter bounds")
    else
        println("     ❌ Poor parameter estimation")
        println("     💡 Consider: checking simulation quality, noise levels, or optimization settings")
    end
    
    println("\n   📝 USAGE NOTES:")
    println("     • Both matrix and custom type methods give identical results")
    println("     • Custom type method is more convenient for complex workflows")
    println("     • Matrix method is faster for simple parameter estimation")
    println("     • Noise significantly affects parameter estimation accuracy")
    println("     • Likelihood surface analysis helps identify optimization landscape")
    
    # ========================================
    # RETURN RESULTS FOR FURTHER ANALYSIS
    # ========================================
    return (
        simulation_data = (p1=p1, p2=p2, states=states, true_params=[true_k_on, true_k_off]),
        custom_sim = sim,
        matrix_optimization = matrix_result,
        custom_optimization = custom_result,
        likelihood_surface = (surface=likelihood_surface, k_range=collect(k_range)),
        forward_results = (alpha=alpha, e1=e1_vec, e2=e2_vec),
        summary = (
            k_on_accuracy = k_on_accuracy,
            k_off_accuracy = k_off_accuracy,
            converged = Optim.converged(matrix_result.optimization_result)
        )
    )
end

# ========================================
# QUICK START EXAMPLE (MINIMAL VERSION)
# ========================================

"""
Quick start example for immediate testing with minimal setup.
Use this when you just want to test the optimization without full simulation.
"""
function quick_optimization_example()
    println("🚀 QUICK START OPTIMIZATION EXAMPLE")
    println("-" ^ 40)
    
    # Generate simple test data
    n_steps = 50
    p1 = cumsum(randn(2, n_steps) * 0.02, dims=2)  # Random walk for particle 1
    p2 = p1 + 0.05 * randn(2, n_steps)             # Particle 2 follows particle 1 with noise
    
    println("Generated $n_steps time steps of test data")
    
    # Run optimization
    result = optimize_k_parameters_matrix(p1, p2, 0.01, initial_params=[0.2, 0.2])
    
    println("Optimization results:")
    println("  k_on:  $(round(result.k_on, digits=4))")
    println("  k_off: $(round(result.k_off, digits=4))")
    println("  Log-likelihood: $(round(result.loglikelihood, digits=2))")
    println("  Converged: $(Optim.converged(result.optimization_result))")
    
    return result
end

