using Random
using CairoMakie
using Distributions
using SMLMSim

abstract type Abstract_obs end

struct Gaus2state <: Abstract_obs
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

function p_state(o::Gaus2state, frame; μ=0, σ=1)
    return pdf(Normal(μ, σ), o.observables[frame])
end 

function p_state(o::ObservableHist, state, frame; dl_dimer=0, sigma=0.2, dt=0.01)
    if state == 1
        return compute_free_density(o, frame, sigma=sigma, dt=dt)
    elseif state == 2
        return compute_dimer_density(o, frame, sigma=sigma, dt=dt)
    end
end 

function compute_free_density(o::ObservableHist, frame; sigma=0.2, dt=0.01)
    # Calculate distance between molecules
    dn = sqrt((o.observables.frames[frame].molecules[1].x - o.observables.frames[frame].molecules[2].x)^2 
              + (o.observables.frames[frame].molecules[1].y - o.observables.frames[frame].molecules[2].y)^2)

    dn_1 = sqrt((o.observables.frames[frame+1].molecules[1].x - o.observables.frames[frame+1].molecules[2].x)^2 
                + (o.observables.frames[frame+1].molecules[1].y - o.observables.frames[frame+1].molecules[2].y)^2)

    # Log-scale calculations for numerical stability
    log_density = log(dn) - 2*log(sigma) - (dn^2 + dn_1^2)/(2*sigma^2) + 
                 log(modified_bessel(dt, dn, dn_1, sigma))
    
    # Return exp of log-density, with a safety check
    return exp(min(log_density, 700.0))  # Prevent overflow
end

function compute_dimer_density(o::ObservableHist, frame; sigma=0.2, dt=0.01)
    # Calculate distance
    dn = sqrt((o.observables.frames[frame].molecules[1].x - o.observables.frames[frame].molecules[2].x)^2 
              + (o.observables.frames[frame].molecules[1].y - o.observables.frames[frame].molecules[2].y)^2)

    # Log-scale calculations
    log_density = log(dn) - 2*log(sigma) - 
                 (o.arguments.d_dimer^2 + dn^2)/(2*sigma^2) + 
                 log(modified_bessel(dt, o.arguments.r_react, dn, sigma))
    
    # Return exp of log-density, with a safety check
    return exp(min(log_density, 700.0))  # Prevent overflow
end

function modified_bessel(dt, d1, d2, σ)
    result = 0.0
    n_points = ceil(Int, 2π/dt)
    
    for i in 0:n_points
        θ = i * dt
        exponent = min((d1 * d2 * cos(θ))/σ^2, 700.0)  # Prevent overflow
        result += exp(exponent)/(2π)
    end
    
    return max(result, 1e-300)  # Prevent underflow
end

function forward_algorithm(observables::ObservableHist; debug=false)
    N = length(observables.observables.frames)-1
    alpha = zeros(2, N)
    scale = zeros(N)  
    
    # Calculate distance-dependent transition probabilities for first frame
    init_dist = sqrt(
        (observables.observables.frames[1].molecules[1].x - observables.observables.frames[1].molecules[2].x)^2 + 
        (observables.observables.frames[1].molecules[1].y - observables.observables.frames[1].molecules[2].y)^2
    )
    
    if debug
        println("Initial distance between molecules: $init_dist")
        println("Reaction radius: $(observables.arguments.r_react)")
    end
    
    # Initialize first frame based on distance
    if init_dist <= observables.arguments.r_react
        alpha[1, 1] = 0.2  # Lower probability of free state if close
        alpha[2, 1] = 0.8  # Higher probability of dimer state if close
    else
        alpha[1, 1] = 0.8  # Higher probability of free state if far
        alpha[2, 1] = 0.2  # Lower probability of dimer state if far
    end
    
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
    if debug
        println("Initial alpha: ", alpha[:, 1])
    end
    
    for t in 2:N
        # Calculate current distance
        curr_dist = sqrt(
            (observables.observables.frames[t].molecules[1].x - observables.observables.frames[t].molecules[2].x)^2 + 
            (observables.observables.frames[t].molecules[1].y - observables.observables.frames[t].molecules[2].y)^2
        )
        
        # Calculate distance-dependent transition probabilities
        p_to_dimer = exp(-curr_dist/observables.arguments.r_react)
        p_to_dimer = min(max(p_to_dimer, 0.1), 0.9)  # Keep probabilities bounded
        
        # Construct transition matrix based on current distance
        T = [1-p_to_dimer p_to_dimer;
             observables.arguments.k_off 1-observables.arguments.k_off]
        
        if debug && t <= 5
            println("\nFrame $t:")
            println("Distance: $curr_dist")
            println("P(to dimer): $p_to_dimer")
            println("Transition matrix:")
            println(T)
        end
        
        # Calculate emission probabilities
        e1 = p_state(observables, 1, t)
        e2 = p_state(observables, 2, t)
        
        if debug && t <= 5
            println("Emission probabilities - Free: $e1, Dimer: $e2")
        end
        
        # Forward algorithm update
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        # Scale to prevent numerical underflow
        scale[t] = sum(alpha[:, t])
        if scale[t] > 0
            alpha[:, t] ./= scale[t]
        end
        
        if debug && t <= 5
            println("Alpha after scaling: ", alpha[:, t])
        end
    end
    
    loglikelihood = sum(log.(scale[scale .> 0]))
    
    return alpha, loglikelihood
end

function forward_algorithm(observations::Gaus2state, T, μ1, σ1, μ2, σ2; debug=false)
    N = length(observations.observables)
    alpha = zeros(2, N)
    scale = zeros(N)  
    
    alpha[1, 1] = p_state(observations, 1, μ=μ1, σ=σ1)
    alpha[2, 1] = p_state(observations, 1, μ=μ2, σ=σ2)
    
    if debug
        println("Initial probabilities - State 1: $(alpha[1, 1]), State 2: $(alpha[2, 1])")
    end
    
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
    for t in 2:N
        e1 = p_state(observations, t, μ=μ1, σ=σ1)
        e2 = p_state(observations, t, μ=μ2, σ=σ2)
        
        if debug && t <= 5
            println("\nFrame $t:")
            println("Emission probabilities - State 1: $e1, State 2: $e2")
        end
        
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= scale[t]
        
        if debug && t <= 5
            println("Alpha after scaling: ", alpha[:, t])
        end
    end
    
    loglikelihood = sum(log.(scale))
    
    return alpha, loglikelihood
end

function analyze_states(alpha::Matrix{Float64}, actual_states::Vector{Int64})
    predicted_states = zeros(Int64, size(alpha, 2))
    for t in 1:size(alpha, 2)
        predicted_states[t] = argmax(alpha[:, t])
    end
    
    # Print first few entries to verify alignment
    println("\nFirst 10 entries comparison:")
    println("Index | Predicted | Actual | Alpha probabilities")
    println("-" ^ 50)
    for i in 1:min(10, length(predicted_states))
        println("$i | $(predicted_states[i]) | $(actual_states[i]) | [$(round(alpha[1,i], digits=3)), $(round(alpha[2,i], digits=3))]")
    end
    
    # Print state distribution
    n_state1_pred = sum(predicted_states .== 1)
    n_state2_pred = sum(predicted_states .== 2)
    n_state1_actual = sum(actual_states .== 1)
    n_state2_actual = sum(actual_states .== 2)
    
    println("\nState Distribution:")
    println("Predicted - State 1: $n_state1_pred, State 2: $n_state2_pred")
    println("Actual    - State 1: $n_state1_actual, State 2: $n_state2_actual")
    
    # Calculate accuracy
    correct = sum(predicted_states .== actual_states)
    accuracy = correct / length(actual_states)
    
    return accuracy, predicted_states
end

# Run the analysis
k12, k21 = 0.1, 0.1     
Δt = 0.1               
sz = 1000               
μ1, σ1 = 1.0, 1.0
μ2, σ2 = 2.0, 1.0
t = 0:Δt:((sz-1)*Δt)

# Gaussian 2-state model
println("\nRunning Gaussian 2-state model...")
states, obs, act_states, T = simulate_hmm(k12, k21, Δt, sz, μ1, σ1, μ2, σ2)
observables = Gaus2state(states, obs, act_states, T)
alpha, loglik = forward_algorithm(observables, T, μ1, σ1, μ2, σ2, debug=true)
println("\nAnalyzing Gaussian Model:")
accuracy_gaus, predicted_states_gaus = analyze_states(alpha, act_states)
println("\nGaussian 2-state model accuracy: $(round(accuracy_gaus * 100, digits=2))%")

# Observable History model
println("\nRunning Observable History model...")
simulation = run_simulation()
alpha_1, loglik_1 = forward_algorithm(simulation, debug=true)

# Collect actual states with debugging information
println("\nCollecting actual states...")
actual_states = Int64[]
for i in 1:length(simulation.observables.frames)-1
    # Print the first few molecule states for debugging
    if i <= 5
        println("Frame $i - Molecule 1 state: $(simulation.observables.frames[i].molecules[1].state)")
        println("         Distance between molecules: $( sqrt(
            (simulation.observables.frames[i].molecules[1].x - simulation.observables.frames[i].molecules[2].x)^2 + 
            (simulation.observables.frames[i].molecules[1].y - simulation.observables.frames[i].molecules[2].y)^2))")
    end
    push!(actual_states, simulation.observables.frames[i].molecules[1].state)     
end

println("\nAnalyzing Observable History Model:")
accuracy_obs, predicted_states_obs = analyze_states(alpha_1, actual_states)
println("\nObservable History model accuracy: $(round(accuracy_obs * 100, digits=2))%")

# Visualization of results
function plot_state_predictions(t, predicted_states, actual_states, model_name)
    fig = Figure(size=(900, 400))
    
    ax = Axis(fig[1, 1],
        xlabel = "Time",
        ylabel = "State",
        title = "$(model_name) - Predicted vs Actual States")
    
    # Plot actual states
    scatter!(ax, t, actual_states, color=:blue, label="Actual States", markersize=2)
    
    # Plot predicted states
    lines!(ax, t, predicted_states, color=:red, label="Predicted States")
    
    # Add legend
    axislegend(ax)
    
    # Plot state probabilities
    ax2 = Axis(fig[2, 1],
        xlabel = "Time",
        ylabel = "Probability",
        title = "State Probabilities")
    
    if model_name == "Gaussian_Model"
        lines!(ax2, t, alpha[1, :], label="P(State 1)", color=:blue)
        lines!(ax2, t, alpha[2, :], label="P(State 2)", color=:red)
    else
        lines!(ax2, t, alpha_1[1, :], label="P(State 1)", color=:blue)
        lines!(ax2, t, alpha_1[2, :], label="P(State 2)", color=:red)
    end
    
    axislegend(ax2)
    
    display(fig)
    save("$(model_name)_predictions.png", fig)
end

# Calculate time vectors for plotting
t_gaus = 0:Δt:((sz-1)*Δt)
t_obs = 0:Δt:((length(actual_states)-1)*Δt)

# Plot results for both models
println("\nGenerating visualizations...")
plot_state_predictions(t_gaus, predicted_states_gaus, act_states, "Gaussian_Model")
plot_state_predictions(t_obs, predicted_states_obs, actual_states, "Observable_History_Model")

# Additional Analysis
println("\nAdditional Analysis:")

# Transition statistics
function analyze_transitions(states)
    transitions = Dict{Tuple{Int,Int}, Int}()
    for i in 1:(length(states)-1)
        transition = (states[i], states[i+1])
        transitions[transition] = get(transitions, transition, 0) + 1
    end
    
    total = sum(values(transitions))
    println("\nTransition probabilities:")
    for (trans, count) in transitions
        prob = count / total
        println("$(trans[1]) → $(trans[2]): $(round(prob, digits=3))")
    end
end

println("\nGaussian Model Transitions:")
analyze_transitions(act_states)

println("\nObservable History Model Transitions:")
analyze_transitions(actual_states)

# State duration analysis
function analyze_state_durations(states)
    current_state = states[1]
    current_duration = 1
    durations = Dict{Int, Vector{Int}}()
    
    for i in 2:length(states)
        if states[i] == current_state
            current_duration += 1
        else
            if !haskey(durations, current_state)
                durations[current_state] = Int[]
            end
            push!(durations[current_state], current_duration)
            current_state = states[i]
            current_duration = 1
        end
    end
    
    # Add the last duration
    if !haskey(durations, current_state)
        durations[current_state] = Int[]
    end
    push!(durations[current_state], current_duration)
    
    println("\nState duration statistics:")
    for (state, durs) in durations
        println("State $state:")
        println("  Mean duration: $(mean(durs))")
        println("  Max duration: $(maximum(durs))")
        println("  Min duration: $(minimum(durs))")
    end
end

println("\nGaussian Model State Durations:")
analyze_state_durations(act_states)

println("\nObservable History Model State Durations:")
analyze_state_durations(actual_states)