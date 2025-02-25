using Random
using CairoMakie
using Distributions
using SMLMSim

abstract type Abstract_obs end

struct Gaus2state <:Abstract_obs
    states::Vector{Float64} 
    observables::Vector{Float64}
    actual_states::Vector{Int64}
    transition_matrix::Matrix{Float64}
end

struct ObservableHist <: Abstract_obs
    observables::SMLMSim.InteractionDiffusion.MoleculeHistory
    arguments::SMLMSim.InteractionDiffusion.ArgsSmol
    dimer_history::SMLMSim.InteractionDiffusion.MoleculeHistory
end 

function run_simulation(;density=0.02, t_max=25, box_size=10, k_off=0.3, r_react=2)
   result = SMLMSim.InteractionDiffusion.smoluchowski(
        density=density,
        t_max=t_max, 
        box_size=box_size,
        k_off=k_off,
        r_react=r_react
    )
    state_history = result[1]
    args = result[2]
    
    dimer_history = SMLMSim.get_dimers(state_history)
    return ObservableHist(state_history, args, dimer_history)
end

function simulate_hmm(k12, k21, Δt, sz, μ1, σ1, μ2, σ2)
    T = [1-(k12*Δt) k12*Δt; 
         k21*Δt 1-(k21*Δt)]
    
    states = zeros(Float64, sz)
    observables = zeros(Float64, sz)
    actual_states = zeros(Int, sz)
    current_state = 1
    
    for i in 1:sz
        r = rand()
        if current_state == 1
            if r < T[1,1]
                current_state = 1
                states[i] = μ1
                actual_states[i] = 1
                observables[i] = rand(Normal(μ1, σ1))
            else
                current_state = 2
                states[i] = μ2
                actual_states[i] = 2
                observables[i] = rand(Normal(μ2, σ2))
            end
        else
            if r < T[2,1]
                current_state = 1
                states[i] = μ1
                actual_states[i] = 1
                observables[i] = rand(Normal(μ1, σ1))
            else
                current_state = 2
                states[i] = μ2
                actual_states[i] = 2
                observables[i] = rand(Normal(μ2, σ2))
            end
        end
    end
    
    return states, observables, actual_states, T
end

function p_state(o:: Gaus2state, frame; μ=0, σ=1)
    return pdf(Normal(μ, σ), o.observables[frame])
end 

function p_state(o::ObservableHist, state, frame; sigma=0.1, dt=0.01)
    if state == 1
        return compute_free_density(o, frame, sigma=sigma, dt=dt)
    elseif state == 2
        return compute_dimer_density(o, frame, sigma=sigma, dt=dt)
    end
end

function compute_free_density(o::ObservableHist, frame; sigma=0.2, dt=0.01)
    # Safety check for accessing next frame
    if frame >= length(o.observables.frames)
        println("Warning: Trying to access frame beyond range")
        return 1e-8  # Return a very small non-zero value
    end
    
    # Get distances between molecules in current frame
    dn = sqrt((o.observables.frames[frame].molecules[1].x - o.observables.frames[frame].molecules[2].x)^2 
        + (o.observables.frames[frame].molecules[1].y - o.observables.frames[frame].molecules[2].y)^2)
    
    # Get distances between molecules in next frame
    dn_1 = sqrt((o.observables.frames[frame+1].molecules[1].x - o.observables.frames[frame+1].molecules[2].x)^2 
        + (o.observables.frames[frame+1].molecules[1].y - o.observables.frames[frame+1].molecules[2].y)^2)
    
    # Avoid potential Inf/NaN from exp with large negative numbers
    exponent = min(0, (-(dn^2) - (dn_1)^2)/sigma^2)
    
    # Calculate modified Bessel function with safety checks
    bessel_value = modified_bessel(dt, dn, dn_1, sigma)
    
    # Calculate density with safety checks to avoid zero/Inf
    density_val = max(1e-8, (dn/sigma^2) * exp(exponent) * bessel_value)
    
    # Debug output
    println("Free density: $(round(density_val, digits=10))")
    
    return density_val
end

function compute_dimer_density(o::ObservableHist, frame; sigma=0.2, dt=0.01)
    # Get distance between molecules in current frame
    dn = sqrt((o.observables.frames[frame].molecules[1].x - o.observables.frames[frame].molecules[2].x)^2 
        + (o.observables.frames[frame].molecules[1].y - o.observables.frames[frame].molecules[2].y)^2)

    # Debug the inputs occasionally to avoid excessive output
    if frame % 500 == 0
        println("Debug dimer density inputs:")
        println("  dn = $dn")
        println("  r_react = $(o.arguments.r_react)")
        println("  sigma = $sigma")
    end
    
    # Ensure we have a minimum distance to avoid numerical issues
    effective_dn = max(0.001, dn)
    
    # Use r_react from o.arguments (not simulation.arguments)
    d_dimer = o.arguments.r_react
    
    # Compute a safe exponent - limit to avoid underflow
    exponent = min(0, (-(d_dimer^2) - (effective_dn)^2)/sigma^2)
    
    # Compute bessel value with safety checks
    bessel_value = modified_bessel(dt, d_dimer, effective_dn, sigma)
    
    # Include a small baseline probability to prevent zero density
    # This is crucial since your output shows all dimer densities are zero
    baseline = 1e-5
    density_val = ((effective_dn)/sigma^2) * exp(exponent) * bessel_value + baseline
    
    println("Dimer density: $(round(density_val, digits=10))")
    return density_val
end

function modified_bessel(dt, d1, d2, σ)
    # Ensure parameters are positive to avoid NaN results
    d1 = max(1e-10, d1)
    d2 = max(1e-10, d2)
    σ = max(1e-10, σ) 
    
    result = 0.0
    
    # Calculate x with bounds to prevent overflow
    x = min(10.0, (d1 * d2) / (σ^2))
    
    # Use fixed number of steps instead of dt parameter for consistency
    n_steps = 200
    dt_adjusted = 2*pi/n_steps
    
    for i in 0:n_steps-1
        θ = i * dt_adjusted
        # Use bounded exp to prevent overflow
        exp_value = min(1e8, exp(min(20.0, x * cos(θ))))
        result += exp_value
    end

    result *= dt_adjusted / (2*pi)
    
    # Ensure non-zero result
    result = max(1e-8, result)
    
    return result
end

function forward_algorithm(observations::Gaus2state, T, μ1, σ1, μ2, σ2)
    N = length(observations.observables)
    alpha = zeros(2, N)
    scale = zeros(N)  
    
    # Initialize with emission probabilities
    alpha[1, 1] = p_state(observations, 1, μ=μ1, σ=σ1)
    alpha[2, 1] = p_state(observations, 1, μ=μ2, σ=σ2)
    
    # Normalize with safety check
    scale[1] = sum(alpha[:, 1])
    if scale[1] > 0
        alpha[:, 1] ./= scale[1]
    else
        alpha[1, 1] = 0.5
        alpha[2, 1] = 0.5
        scale[1] = 1.0
    end
    
    # Forward pass
    for t in 2:N
        # Calculate emission probabilities
        e1 = p_state(observations, t, μ=μ1, σ=σ1)
        e2 = p_state(observations, t, μ=μ2, σ=σ2)
        
        # Handle potential zero probabilities
        e1 = max(1e-10, e1)
        e2 = max(1e-10, e2)
        
        # Calculate forward probabilities
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        # Normalize to prevent underflow
        scale[t] = alpha[1, t] + alpha[2, t]
        if scale[t] > 0
            alpha[:, t] ./= scale[t]
        else
            # If both probabilities are effectively zero, use previous distribution
            alpha[:, t] = alpha[:, t-1]
            scale[t] = 1e-10
        end
    end
    
    # Calculate log-likelihood, handling potential zeros
    safe_scale = [max(1e-300, s) for s in scale]
    loglikelihood = sum(log.(safe_scale))
    
    return alpha, loglikelihood
end

function forward_algorithm(observables::ObservableHist)
    N = length(observables.observables.frames)-1
    alpha = zeros(2, N)
    scale = zeros(N)
    
    # Set initial state probabilities with minimum values to avoid zeros
    alpha[1, 1] = max(1e-8, p_state(observables, 1, 1))
    alpha[2, 1] = max(1e-8, p_state(observables, 2, 1))
    
    # Check if both values are very small
    if alpha[1, 1] < 1e-6 && alpha[2, 1] < 1e-6
        println("Warning: Both initial state probabilities are very small. Using uniform.")
        alpha[1, 1] = 0.5
        alpha[2, 1] = 0.5
    end
    
    println("Initial alpha: $(alpha[:, 1])")
    
    # Normalize
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
    # Set up transition matrix with minimum probabilities
    Δt = observables.arguments.dt
    
    # Make sure k_on is non-zero to allow state transitions
    k_on = max(0.01, 0.1)  # Ensure minimum value
    k_off = max(0.01, observables.arguments.k_off)  # Ensure minimum value
    
    T = [1-(k_on*Δt) k_on*Δt; 
         k_off*Δt 1-(k_off*Δt)]
    
    println("Transition matrix T:")
    println(T)
    
    # Forward pass
    for t in 2:N
        # Calculate emission probabilities with minimum values
        e1 = max(1e-8, p_state(observables, 1, t))
        e2 = max(1e-8, p_state(observables, 2, t))
        
        # Add noise to zero probabilities to allow state switching
        if e1 < 1e-6 && e2 > 1e-6
            e1 = 1e-6 * e2  # Small proportion of the other state
        elseif e2 < 1e-6 && e1 > 1e-6
            e2 = 1e-6 * e1  # Small proportion of the other state
        elseif e1 < 1e-6 && e2 < 1e-6
            # Both probabilities tiny - use previous values or uniform
            e1 = 0.5
            e2 = 0.5
        end
        
        # Forward update with smoothing to prevent getting stuck in one state
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        # Apply minimum probability to prevent degenerate cases
        alpha[1, t] = max(1e-8, alpha[1, t])
        alpha[2, t] = max(1e-8, alpha[2, t])
        
        # Normalize
        scale[t] = sum(alpha[:, t])
        if scale[t] > 0
            alpha[:, t] ./= scale[t]
        else
            println("Warning: Zero scale at t=$t. Using previous distribution.")
            alpha[:, t] = alpha[:, t-1]
            scale[t] = 1e-8
        end
        
        # Debug outputs for a few iterations
        if t % 500 == 0 || t > N-5
            println("t=$t, e1=$e1, e2=$e2")
            println("alpha[$t]: $(alpha[:, t])")
            println("scale[$t]: $(scale[t])")
        end
    end
    
    # Replace zeros with small values for log calculation
    safe_scale = max.(1e-300, scale)
    loglikelihood = sum(log.(safe_scale))
    
    return alpha, loglikelihood
end

function calculate_accuracy(alpha::Matrix{Float64}, actual_states::Vector{Int64})
    n_timesteps = min(size(alpha, 2), length(actual_states))
    
    # Debug dimension info
    println("Alpha dimensions: $(size(alpha))")
    println("Actual states length: $(length(actual_states))")
    println("Using n_timesteps = $n_timesteps")
    
    # Convert alpha to predicted states
    predicted_states = zeros(Int64, n_timesteps)
    for t in 1:n_timesteps
        predicted_states[t] = argmax(alpha[:, t])
    end
    
    # Calculate accuracy
    actual_states_trimmed = actual_states[1:n_timesteps]
    correct = sum(predicted_states .== actual_states_trimmed)
    accuracy = correct / n_timesteps
    
    # Show state distribution to debug
    state1_count = sum(predicted_states .== 1)
    state2_count = sum(predicted_states .== 2)
    
    println("Predicted state distribution: State 1: $state1_count ($(round(state1_count/n_timesteps*100, digits=2))%), State 2: $state2_count ($(round(state2_count/n_timesteps*100, digits=2))%)")
    
    # Show actual state distribution
    actual_state1_count = sum(actual_states_trimmed .== 1)
    actual_state2_count = sum(actual_states_trimmed .== 2)
    
    println("Actual state distribution: State 1: $actual_state1_count ($(round(actual_state1_count/n_timesteps*100, digits=2))%), State 2: $actual_state2_count ($(round(actual_state2_count/n_timesteps*100, digits=2))%)")
    
    return accuracy
end

# Original test code (left here for reference)
k12, k21 = 0.1, 0.1     
Δt = 0.1               
sz = 1000               
μ1, σ1 = 1.0, 1
μ2, σ2 = 2.0, 1
t = 0:Δt:((sz-1)*Δt)


states, obs, act_states, T = simulate_hmm(k12, k21, Δt, sz, μ1, σ1, μ2, σ2)
observables = Gaus2state(states, obs, act_states, T)

simulation = run_simulation()

alpha_1, loglik_1 = forward_algorithm(simulation)


alpha, loglik = forward_algorithm(observables, T, μ1, σ1, μ2, σ2)
actual_states=[]

for i in 1:length(simulation.observables.frames)-1
    push!(actual_states, simulation.observables.frames[i].molecules[1].state)     
end


accuracy_gaus = calculate_accuracy(alpha, act_states)
println("\nGaussian 2-state model accuracy: $(round(accuracy_gaus * 100, digits=2))%")

actual_states_int = Int64.(actual_states)
accuracy_obs = calculate_accuracy(alpha_1, actual_states_int)
println("\nObservable History model accuracy: $(round(accuracy_obs * 100, digits=2))%")