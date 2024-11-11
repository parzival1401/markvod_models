using Random
using Distributions
using LinearAlgebra
using CairoMakie

"""
Simulate a Hidden Markov Model with two states
"""
function simulate_hmm(k12::Float64, k21::Float64, Δt::Float64, sz::Int, 
                     μ1::Float64, σ1::Float64, μ2::Float64, σ2::Float64)
    
    # Transition matrix
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

"""
Calculate negative log-likelihood for HMM parameters
"""
function negative_log_likelihood(params::Vector{Float64}, 
                               observations::Vector{Float64},
                               Δt::Float64, 
                               μ1::Float64, σ1::Float64,
                               μ2::Float64, σ2::Float64)
    k12, k21 = params
    
    # Check if parameters are physically meaningful
    if k12 < 0 || k21 < 0 || k12 > 1/Δt || k21 > 1/Δt
        return Inf
    end
    
    # Construct transition matrix
    T = [1-(k12*Δt) k12*Δt; 
         k21*Δt 1-(k21*Δt)]
    
    N = length(observations)
    alpha = zeros(2, N)
    scale = zeros(N)
    
    # Initialize forward variables
    alpha[1, 1] = pdf(Normal(μ1, σ1), observations[1])
    alpha[2, 1] = pdf(Normal(μ2, σ2), observations[1])
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
    # Forward algorithm
    for t in 2:N
        e1 = pdf(Normal(μ1, σ1), observations[t])
        e2 = pdf(Normal(μ2, σ2), observations[t])
        
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= scale[t]
    end
    
    return -sum(log.(scale))
end

"""
Nelder-Mead simplex optimization algorithm
"""
function nelder_mead_optimize(f::Function, x0::Vector{Float64};
                            args::Tuple=(), 
                            α::Float64=1.0,   # reflection
                            γ::Float64=2.0,   # expansion
                            ρ::Float64=0.5,   # contraction
                            σ::Float64=0.5,   # shrink
                            max_iter::Int=1000,
                            tol::Float64=1e-8)
    
    n = length(x0)
    
    # Create initial simplex
    simplex = zeros(n + 1, n)
    f_values = zeros(n + 1)
    
    # First vertex is initial guess
    simplex[1, :] = x0
    f_values[1] = f(x0, args...)
    
    # Create other vertices
    for i in 1:n
        point = copy(x0)
        point[i] = point[i] == 0 ? 0.00025 : point[i] * 1.05
        simplex[i + 1, :] = point
        f_values[i + 1] = f(point, args...)
    end
    
    for iter in 1:max_iter
        # Sort vertices
        perm = sortperm(f_values)
        f_values = f_values[perm]
        simplex = simplex[perm, :]
        
        # Check convergence
        if maximum(abs.(f_values[end] - f_values[1])) < tol
            return simplex[1, :], f_values[1]
        end
        
        # Centroid of best n points
        x̄ = vec(mean(simplex[1:end-1, :], dims=1))
        
        # Reflection
        xr = x̄ + α * (x̄ - simplex[end, :])
        fr = f(xr, args...)
        
        if f_values[1] ≤ fr < f_values[end-1]
            simplex[end, :] = xr
            f_values[end] = fr
            continue
        end
        
        # Expansion
        if fr < f_values[1]
            xe = x̄ + γ * (xr - x̄)
            fe = f(xe, args...)
            if fe < fr
                simplex[end, :] = xe
                f_values[end] = fe
            else
                simplex[end, :] = xr
                f_values[end] = fr
            end
            continue
        end
        
        # Contraction
        xc = x̄ + ρ * (simplex[end, :] - x̄)
        fc = f(xc, args...)
        
        if fc < f_values[end]
            simplex[end, :] = xc
            f_values[end] = fc
            continue
        end
        
        # Shrink
        for i in 2:n+1
            simplex[i, :] = simplex[1, :] + σ * (simplex[i, :] - simplex[1, :])
            f_values[i] = f(simplex[i, :], args...)
        end
    end
    
    return simplex[1, :], f_values[1]
end

"""
Forward algorithm for HMM
"""
function forward_algorithm(observations::Vector{Float64}, T::Matrix{Float64}, 
                         μ1::Float64, σ1::Float64, μ2::Float64, σ2::Float64)
    N = length(observations)
    alpha = zeros(2, N)
    scale = zeros(N)
    
    # Initialize
    alpha[1, 1] = pdf(Normal(μ1, σ1), observations[1])
    alpha[2, 1] = pdf(Normal(μ2, σ2), observations[1])
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
    # Forward recursion
    for t in 2:N
        e1 = pdf(Normal(μ1, σ1), observations[t])
        e2 = pdf(Normal(μ2, σ2), observations[t])
        
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= scale[t]
    end
    
    loglikelihood = sum(log.(scale))
    
    return alpha, loglikelihood
end

"""
Plot HMM results
"""
function plot_hmm_results(t::Vector{Float64}, states::Vector{Float64}, 
                         observables::Vector{Float64}, actual_states::Vector{Int}, 
                         alpha::Matrix{Float64})
    fig = Figure(size=(1200, 800))
    
    ax1 = Axis(fig[1, 1],
        xlabel = "Time",
        ylabel = "Value",
        title = "HMM Simulation")
    
    lines!(ax1, t, observables, color = (:blue, 0.5), label = "Observations")
    stairs!(ax1, t, states, color = (:red, 0.5), label = "True States")
    axislegend(ax1)
    
    ax2 = Axis(fig[2, 1],
        xlabel = "Time",
        ylabel = "Probability",
        title = "Forward Probabilities (Normalized)")
    
    # Add background coloring for true states
    for i in 1:length(t)-1
        if actual_states[i] == 1
            vspan!(ax2, t[i], t[i+1], color = (:blue, 0.1))
        else
            vspan!(ax2, t[i], t[i+1], color = (:red, 0.1))
        end
    end
    
    lines!(ax2, t, alpha[1, :], color = :blue, label = "P(State 1)")
    lines!(ax2, t, alpha[2, :], color = :red, label = "P(State 2)")
    axislegend(ax2)
    
    ylims!(ax2, 0, 1)
    
    return fig
end

"""
Calculate accuracy of state predictions
"""
function calculate_accuracy(alpha::Matrix{Float64}, actual_states::Vector{Int})
    predicted_states = [argmax(alpha[:, t]) for t in 1:size(alpha, 2)]
    accuracy = mean(predicted_states .== actual_states)
    return accuracy
end

"""
Optimize HMM parameters using multiple random starts
"""
function optimize_hmm_parameters(observations::Vector{Float64}, 
                              Δt::Float64,
                              μ1::Float64, σ1::Float64,
                              μ2::Float64, σ2::Float64;
                              n_restarts::Int=5)
    
    objective(params) = negative_log_likelihood(params, observations, Δt, μ1, σ1, μ2, σ2)
    
    best_params = zeros(2)
    best_nll = Inf
    
    # Try multiple random starting points
    for i in 1:n_restarts
        # Random initial guess between 0 and 1/Δt
        initial_guess = rand(2) .* (1/Δt)
        
        params, nll = nelder_mead_optimize(objective, initial_guess)
        
        if nll < best_nll
            best_nll = nll
            best_params = params
        end
    end
    
    return best_params, best_nll
end

"""
Run complete HMM analysis with optimization
"""
function run_hmm_analysis(; Δt::Float64=0.1,
                         μ1::Float64=1.0, σ1::Float64=1.0,
                         μ2::Float64=2.0, σ2::Float64=1.0,
                         true_k12::Float64=0.1, true_k21::Float64=0.1,
                         sz::Int=1000)
    
    # Generate true data
    println("Generating true data...")
    states, observations, actual_states, T = simulate_hmm(
        true_k12, true_k21, Δt, sz, μ1, σ1, μ2, σ2
    )
    
    # Optimize parameters
    println("Optimizing parameters...")
    optimal_params, min_nll = optimize_hmm_parameters(
        observations, Δt, μ1, σ1, μ2, σ2
    )
    k12_opt, k21_opt = optimal_params
    
    # Calculate forward probabilities with optimal parameters
    T_opt = [1-(k12_opt*Δt) k12_opt*Δt; 
             k21_opt*Δt 1-(k21_opt*Δt)]
    alpha_opt, loglik_opt = forward_algorithm(observations, T_opt, μ1, σ1, μ2, σ2)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(alpha_opt, actual_states)
    
    # Print results
    println("\nResults:")
    println("True parameters: k12 = $true_k12, k21 = $true_k21")
    println("Optimized parameters: k12 = $(round(k12_opt, digits=3)), k21 = $(round(k21_opt, digits=3))")
    println("Final log-likelihood: $(round(loglik_opt, digits=3))")
    println("State prediction accuracy: $(round(accuracy * 100, digits=2))%")
    
    # Plot results
    t = 0:Δt:((sz-1)*Δt)
    fig = plot_hmm_results(t, states, observations, actual_states, alpha_opt)
    
    return fig, (k12_opt, k21_opt), loglik_opt, accuracy
end

# Example usage
using Random
Random.seed!(123)  # for reproducibility

# Run the analysis
fig, optimal_params, loglik, accuracy = run_hmm_analysis(
    Δt = 0.1,
    μ1 = 1.0, σ1 = 1.0,
    μ2 = 2.0, σ2 = 1.0,
    true_k12 = 0.1, true_k21 = 0.1,
    sz = 1000
)

# Save the figure
save("hmm_results.png", fig)