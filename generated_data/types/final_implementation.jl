using GLMakie
using Random
using Distributions
using Optim

# ========================================
# LOAD EXTERNAL DEPENDENCIES
# ========================================

# Load state generator functions
include("../state_generator.jl")
include("../dimer_difussion.jl")
include("../free_difusion.jl")

println("✅ External dependencies loaded successfully")

# ========================================
# CUSTOM TYPES DEFINITION
# ========================================

"""
Particle structure containing position and time information
"""
struct Particles
    x::Float64
    y::Float64
    t::Float64
end

"""
Simulation structure containing particle trajectories and parameters
"""
struct simulation
    particle_1::Vector{Particles}
    particle_2::Vector{Particles}
    k_states::Vector{Float64}
    sigma::Float64
    D::Float64
    dt::Float64
    d_dimer::Float64
end

# ========================================
# UTILITY FUNCTIONS FROM TYPES_WITH_DATA
# ========================================

"""
Constructor to create Particles from position and time
"""
Particles(pos::Vector{Float64}, time::Float64) = Particles(pos[1], pos[2], time)

"""
Get position matrices from simulation
"""
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

"""
Get time vector from simulation
"""
function get_times(sim::simulation)
    return [p.t for p in sim.particle_1]
end

"""
Calculate distance between particles at given index
"""
function particle_distance(sim::simulation, index::Int)
    p1 = sim.particle_1[index]
    p2 = sim.particle_2[index]
    return sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
end

"""
Reverse columns while preserving size
"""
function reverse_columns_preserve_size(arr)
    num_rows, num_cols = size(arr)
    result = zeros(eltype(arr), num_rows, num_cols)

    for col in 1:num_cols
        result[:, col] = arr[:, num_cols - col + 1]
    end

    return result
end

"""
Shift particle positions
"""
function shift(vec, shift_x, shift_y)
    return vec .+ [shift_x, shift_y]
end

"""
Path correction (non-mutating version)
"""
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

"""
Path correction (in-place mutating version)
"""
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

"""
Generate trajectory segments based on mode
"""
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

"""
Select mode based on section structure
"""
function modes_selection(section::Vector)
    size_section = length(section)
    if size_section == 3
        return :full
    elseif size_section == 2
        if isa(section[1], Pair) && isa(section[2], Pair)
            if section[1].first == 1 && section[2].first == 2
                return :left_half
            elseif section[1].first == 2 && section[2].first == 1
                return :right_half
            else
                error("Invalid section combination")
            end
        else
            return :right_half
        end
    elseif size_section == 1
        return :middle
    else
        error("Invalid number of sections")
    end
end

"""
Extract time stamps from segments
"""
function time_stamps(time_state)
    segments = []
    for i in 1:length(time_state)
        if isa(time_state[i], Pair)
            push!(segments, time_state[i].second)
        else
            push!(segments, time_state[i])
        end
    end
    return segments
end

# ========================================
# SIMULATION GENERATION (PHYSICS-BASED)
# ========================================

"""
Raw simulation function that generates matrices (from types_with_data.jl)
"""
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
        modes = modes_selection(time_in_state[i])

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

"""
Main simulation function - returns custom simulation type
"""
function run_simulation(k_in, k12_off, changes; D=0.01, dt=0.016, σ=0.0, d_dimer=0.05)
    println("🔬 Creating physics-based simulation...")
    println("   Parameters: k_on=$(k_in), k_off=$(k12_off), changes=$(changes)")
    println("   Diffusion: D=$(D), dt=$(dt)")

    # Run raw simulation to get matrices
    p1_matrix, p2_matrix, states, steps, k_params = simulate_raw(k_in, k12_off, changes, D=D, dt=dt)

    # Convert to custom types
    k_states = [k_params[1], k_params[2]]

    # Create time vector
    n_steps = size(p1_matrix, 2)
    time_vec = collect(0:dt:(n_steps-1)*dt)

    # Create particle vectors
    particle_1 = [Particles(p1_matrix[1, i], p1_matrix[2, i], time_vec[i]) for i in 1:n_steps]
    particle_2 = [Particles(p2_matrix[1, i], p2_matrix[2, i], time_vec[i]) for i in 1:n_steps]

    sim = simulation(particle_1, particle_2, k_states, σ, D, dt, d_dimer)

    println("   ✅ Simulation completed")
    println("      Total time steps: $(n_steps)")
    println("      Duration: $(round(time_vec[end], digits=2)) s")
    println("      State changes: $(changes)")

    return sim, states
end

# ========================================
# FORWARD ALGORITHM (PHYSICS-BASED WITH BESSEL)
# ========================================

"""
Modified Bessel function approximation
"""
function modified_bessel(dt, d1, d2, σ)
    result_me = 0.0
    x = (d1*d2)/(σ^2)

    for θ in 0:dt:2*pi
        result_me += exp(x * cos(θ))
    end

    result_me *= dt / (2*pi)
    return result_me
end

"""
Compute free density using physics-based model
"""
function compute_free_density(particle1::Matrix{Float64}, particle2::Matrix{Float64}, index::Int; sigma=0.1, dt=0.01, D=0.01)
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

"""
Compute dimer density using physics-based model
"""
function compute_dimer_density(particle1::Matrix{Float64}, particle2::Matrix{Float64}, index::Int, d_dimer::Float64; sigma=0.1, dt=0.01, D=0.01)
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

"""
Forward algorithm using physics-based densities
"""
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

"""
Forward algorithm for custom simulation type
"""
function forward_algorithm(sim::simulation; dt=0.01, sigma=0.1)
    p1, p2 = get_position_matrices(sim)
    return forward_algorithm(p1, p2, sim.d_dimer, sim.k_states, dt=dt, sigma=sigma, D=sim.D)
end

# ========================================
# OPTIMIZATION (BOUNDED APPROACH)
# ========================================

"""
Optimize k parameters using matrix data with bounded approach (from types_with_data.jl)
"""
function optimize_k_parameters_matrix(p1::Matrix{Float64}, p2::Matrix{Float64}, d_dimer::Float64=0.01;
                                    initial_params::Vector{Float64}=[0.1, 0.1], dt=0.01, sigma=0.1, D=0.01)

    # Create wrapper function for optimization with bounds checking
    function bounded_wrapper(param)
        # Apply bounds: k parameters should be positive and reasonable
        k_on_bounded = clamp(param[1], 1e-6, 2.0)
        k_off_bounded = clamp(param[2], 1e-6, 2.0)
        bounded_params = [k_on_bounded, k_off_bounded]

        try
            _, loglikelihood, _, _ = forward_algorithm(p1, p2, d_dimer, bounded_params, dt=dt, sigma=sigma, D=D)
            result = -loglikelihood
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
                     Optim.Options(g_tol=1e-6, iterations=1000))

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

"""
Optimize k parameters using custom simulation type
"""
function optimize_k_parameters_custom(sim::simulation; initial_params::Vector{Float64}=[0.1, 0.1], dt=0.01, sigma=0.1)
    p1, p2 = get_position_matrices(sim)
    return optimize_k_parameters_matrix(p1, p2, sim.d_dimer, initial_params=initial_params, dt=dt, sigma=sigma, D=sim.D)
end

"""
Optimize with noise comparison
"""
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

# ========================================
# ANALYSIS FUNCTIONS
# ========================================

"""
Extract free segments from simulation
"""
function extract_free_segments(sim::simulation, threshold_distance = nothing)
    threshold = isnothing(threshold_distance) ? sim.d_dimer : threshold_distance

    n_steps = length(sim.particle_1)
    free_segments = []
    current_segment_start = nothing

    for i in 1:n_steps
        dist = sqrt((sim.particle_1[i].x - sim.particle_2[i].x)^2 +
                   (sim.particle_1[i].y - sim.particle_2[i].y)^2)

        if dist > threshold
            if isnothing(current_segment_start)
                current_segment_start = i
            end
        else
            if !isnothing(current_segment_start)
                push!(free_segments, (start=current_segment_start, stop=i-1))
                current_segment_start = nothing
            end
        end
    end

    if !isnothing(current_segment_start)
        push!(free_segments, (start=current_segment_start, stop=n_steps))
    end

    return free_segments
end

"""
Extract bound segments from simulation
"""
function extract_bound_segments(sim::simulation, threshold_distance = nothing)
    threshold = isnothing(threshold_distance) ? sim.d_dimer : threshold_distance

    n_steps = length(sim.particle_1)
    bound_segments = []
    current_segment_start = nothing

    for i in 1:n_steps
        dist = sqrt((sim.particle_1[i].x - sim.particle_2[i].x)^2 +
                   (sim.particle_1[i].y - sim.particle_2[i].y)^2)

        if dist <= threshold
            if isnothing(current_segment_start)
                current_segment_start = i
            end
        else
            if !isnothing(current_segment_start)
                push!(bound_segments, (start=current_segment_start, stop=i-1))
                current_segment_start = nothing
            end
        end
    end

    if !isnothing(current_segment_start)
        push!(bound_segments, (start=current_segment_start, stop=n_steps))
    end

    return bound_segments
end

"""
Add noise to simulation
"""
function add_noise(sim::simulation, σ::Float64=0.01)
    p1, p2 = get_position_matrices(sim)

    # Add noise to positions
    p1_noise = p1 .+ σ * randn(size(p1))
    p2_noise = p2 .+ σ * randn(size(p2))

    # Create new simulation with noisy data
    n_steps = size(p1_noise, 2)
    time_vec = get_times(sim)

    particle_1_noise = [Particles(p1_noise[1, i], p1_noise[2, i], time_vec[i]) for i in 1:n_steps]
    particle_2_noise = [Particles(p2_noise[1, i], p2_noise[2, i], time_vec[i]) for i in 1:n_steps]

    return simulation(particle_1_noise, particle_2_noise, sim.k_states, σ, sim.D, sim.dt, sim.d_dimer)
end

# ========================================
# ANIMATION FUNCTIONS
# ========================================

"""
Animate particles from matrices
"""
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

"""
Animate simulation (custom type)
"""
function animate_simulation(sim::simulation, filename="simulation_animation.mp4"; fps=30)
    p1, p2 = get_position_matrices(sim)
    return animate_particles(p1, p2, filename)
end

# ========================================
# COMPREHENSIVE ANALYSIS
# ========================================

"""
Complete analysis of simulation including forward algorithm and optimization
"""
function analyze_simulation_complete(sim::simulation; dt=0.01, sigma=0.1,
                                   optimize_params=true, create_animation=false)
    println("📊 Complete simulation analysis...")

    # 1. Forward algorithm
    println("\n🔍 Running forward algorithm...")
    alpha, loglik, e1_vec, e2_vec = forward_algorithm(sim; dt=dt, sigma=sigma)

    println("   Log-likelihood: $(round(loglik, digits=2))")

    # 2. Extract segments
    println("\n📏 Extracting binding segments...")
    free_segments = extract_free_segments(sim)
    bound_segments = extract_bound_segments(sim)

    println("   Free segments: $(length(free_segments))")
    println("   Bound segments: $(length(bound_segments))")

    # 3. Parameter optimization
    opt_result = nothing
    if optimize_params
        println("\n⚙️  Optimizing parameters...")
        opt_result = optimize_k_parameters_custom(sim; dt=dt, sigma=sigma)

        println("   Original k_on:  $(round(sim.k_states[1], digits=4))")
        println("   Original k_off: $(round(sim.k_states[2], digits=4))")
        println("   Optimized k_on:  $(round(opt_result.k_on, digits=4))")
        println("   Optimized k_off: $(round(opt_result.k_off, digits=4))")
        println("   Final log-likelihood: $(round(opt_result.loglikelihood, digits=2))")
    end

    # 4. Create animation
    animation_file = nothing
    if create_animation
        println("\n🎬 Creating animation...")
        animation_file = animate_simulation(sim, "complete_analysis.mp4", fps=25)
    end

    # 5. Summary
    println("\n📈 Analysis Summary:")
    println("   Simulation length: $(length(sim.particle_1)) steps")
    println("   Total time: $(round(sim.particle_1[end].t, digits=2)) s")
    println("   Binding events: $(length(bound_segments))")
    println("   Free periods: $(length(free_segments))")
    if opt_result !== nothing
        println("   Optimization converged: $(Optim.converged(opt_result.optimization_result))")
    end

    return (
        alpha = alpha,
        loglikelihood = loglik,
        e1_vec = e1_vec,
        e2_vec = e2_vec,
        free_segments = free_segments,
        bound_segments = bound_segments,
        optimization_result = opt_result,
        animation_file = animation_file
    )
end

# ========================================
# EXAMPLE WORKFLOWS
# ========================================

"""
Example: Physics-based simulation with complete analysis
"""
function example_physics_workflow()
    println("=" ^ 60)
    println("EXAMPLE: Physics-Based Simulation Workflow")
    println("=" ^ 60)

    Random.seed!(1234)

    # Create simulation using physics-based approach
    sim, states = run_simulation(0.5, 0.3, 10, D=0.01, dt=0.016, σ=0.0, d_dimer=0.01)

    # Complete analysis
    analysis = analyze_simulation_complete(sim;
                                         optimize_params=true,
                                         create_animation=true,
                                         dt=0.016,
                                         sigma=0.01)

    println("\n✅ Physics workflow completed!")
    return sim, states, analysis
end

"""
Example: Noise comparison
"""
function example_noise_comparison()
    println("=" ^ 60)
    println("EXAMPLE: Noise Sensitivity Analysis")
    println("=" ^ 60)

    Random.seed!(1234)

    # Create clean simulation
    sim_clean, states = run_simulation(0.5, 0.3, 10, D=0.01, dt=0.016)

    # Add noise
    sim_noisy = add_noise(sim_clean, 0.01)

    # Analyze both
    println("\n📊 Analyzing clean simulation...")
    analysis_clean = analyze_simulation_complete(sim_clean; optimize_params=true, create_animation=false)

    println("\n📊 Analyzing noisy simulation...")
    analysis_noisy = analyze_simulation_complete(sim_noisy; optimize_params=true, create_animation=false)

    # Comparison
    println("\n🎯 Comparison:")
    println("   Clean log-likelihood: $(round(analysis_clean.loglikelihood, digits=2))")
    println("   Noisy log-likelihood: $(round(analysis_noisy.loglikelihood, digits=2))")

    if analysis_clean.optimization_result !== nothing && analysis_noisy.optimization_result !== nothing
        println("   Clean k_on:  $(round(analysis_clean.optimization_result.k_on, digits=4))")
        println("   Noisy k_on:  $(round(analysis_noisy.optimization_result.k_on, digits=4))")
        println("   Clean k_off: $(round(analysis_clean.optimization_result.k_off, digits=4))")
        println("   Noisy k_off: $(round(analysis_noisy.optimization_result.k_off, digits=4))")
    end

    println("\n✅ Noise comparison completed!")
    return sim_clean, sim_noisy, analysis_clean, analysis_noisy
end

println("🎯 Final Implementation Module Loaded Successfully!")
println("=" ^ 60)
println("Available workflows:")
println("  🔬 example_physics_workflow()     - Physics-based simulation")
println("  📊 example_noise_comparison()     - Noise sensitivity analysis")
println()
println("Core functions:")
println("  🔬 run_simulation()               - Create physics-based simulation")
println("  📊 analyze_simulation_complete()  - Full analysis with optimization")
println("  🎬 animate_simulation()           - Create animation")
println("  ⚙️  optimize_k_parameters_custom() - Optimize k parameters")
println("  🔍 forward_algorithm()            - HMM state estimation")
println("=" ^ 60)
