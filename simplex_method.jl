using LinearAlgebra
using Random
using CairoMakie
using Distributions
include("forward_alg.jl")
include("smm.jl")




function calc_accuracy(alpha::Matrix{Float64}, true_states::Vector{Int64})
    pred_states = [argmax(alpha[:,i]) for i in 1:size(alpha,2)]
    return mean(pred_states .== true_states)
end

function calc_nll(params, obs, dt, μ1, σ1, μ2, σ2)
    k12, k21 = params
    
    if any(k -> k < 0 || k > 1/dt, [k12, k21])
        return Inf
    end
    
    T = [
        1-k12*dt k12*dt
        k21*dt 1-k21*dt
    ]
    
    _, ll = forward_algorithm(obs, T, μ1, σ1, μ2, σ2)
    return -ll
end


function optimize(f, x0; args=(), max_iter=1000, tol=1e-8)
    n = length(x0)
    points = zeros(n + 1, n)
    f_vals = zeros(n + 1)
    
    points[1,:] = x0
    f_vals[1] = f(x0, args...)
    
    for i in 1:n
        p = x0
        p[i] *= 1.05
        points[i+1,:] = p
        f_vals[i+1] = f(p, args...)
    end
    
    for _ in 1:max_iter
        order = sortperm(f_vals)
        points = points[order,:]
        f_vals = f_vals[order]
        
        if abs(f_vals[end] - f_vals[1]) < tol
            return points[1,:], f_vals[1]
        end
        
        center = vec(mean(points[1:end-1,:], dims=1))
        
        reflect = center + (center - points[end,:])
        f_reflect = f(reflect, args...)
        
        if f_vals[1] ≤ f_reflect < f_vals[end-1]
            points[end,:] = reflect
            f_vals[end] = f_reflect
            continue
        end
        
        if f_reflect < f_vals[1]
            expand = center + 2(reflect - center)
            f_expand = f(expand, args...)
            points[end,:] = f_expand < f_reflect ? expand : reflect
            f_vals[end] = min(f_expand, f_reflect)
            continue
        end
        
        contract = center + 0.5(points[end,:] - center)
        f_contract = f(contract, args...)
        
        if f_contract < f_vals[end]
            points[end,:] = contract
            f_vals[end] = f_contract
            continue
        end
        
        for i in 2:n+1
            points[i,:] = points[1,:] + 0.5(points[i,:] - points[1,:])
            f_vals[i] = f(points[i,:], args...)
        end
    end
    
    return points[1,:], f_vals[1]
end


function find_hmm_params(obs, dt, μ1, σ1, μ2, σ2; n_tries=5)
    best_params = zeros(2)
    best_nll = Inf
    
    for _ in 1:n_tries
        guess = rand(2) .* (1/dt)
        params, nll = optimize(p -> calc_nll(p, obs, dt, μ1, σ1, μ2, σ2), guess)
        
        if nll < best_nll
            best_nll = nll
            best_params = params
        end
    end
    
    return best_params, best_nll
end



#Random.seed!(123)

dt = 0.1
n_samples = 1000
μ1, σ1 = 1.0, 2
μ2, σ2 = 2.0, 2
true_k12, true_k21 = 0.1, 0.1

states, obs, actual, T = simulate_hmm(true_k12, true_k21, dt, n_samples, μ1, σ1, μ2, σ2)
params, nll = find_hmm_params(obs, dt, μ1, σ1, μ2, σ2)
k12, k21 = params

T_opt = [1-k12*dt k12*dt; k21*dt 1-k21*dt]
alpha, ll = forward_algorithm(obs, T_opt, μ1, σ1, μ2, σ2)
accuracy = calc_accuracy(alpha, actual)

println("True parameters: k12 = $true_k12, k21 = $true_k21")
println("Optimized parameters: k12 = $(round(k12, digits=3)), k21 = $(round(k21, digits=3))")
println("Log-likelihood: $(round(ll, digits=3))")
println("Accuracy: $(round(accuracy * 100, digits=2))%")


