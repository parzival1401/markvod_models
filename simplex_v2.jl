using LinearAlgebra
using Random
using CairoMakie
using Distributions


include("forwardv2.jl")

 


function calc_accuracy(alpha::Matrix{Float64}, true_states::Vector{Int64})
    pred_states = [argmax(alpha[:,i]) for i in 1:size(alpha,2)]
    return mean(pred_states .== true_states)
end


function calc_nll(params::Vector{Float64}, obs::ObservableHist,dt)
    k12, k21 = params
    
    if any(k -> k < 0 || k > 1/dt, [k12, k21])
        return Inf
    end
    
    
    _, ll = forward_algorithm(obs,params)
    
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


function find_hmm_params(obs::ObservableHist,dt; n_tries=2)
    best_params = zeros(2)
    best_nll = Inf
    
    for i in 1:n_tries
        if i == 1
            guess = [1/obs.arguments.t_max,obs.arguments.k_off]
        else
            guess = rand(2) .* (1/dt)
        end
        println("guess: $guess")
        params, nll = optimize(p -> calc_nll(p, obs, dt), guess)
        println("params: $params ,  nll: $nll"  )
        if nll < best_nll
            best_nll = nll
            best_params = params
        end
    end
    
    return best_params, best_nll
end

###############

simulation = run_simulation()
noisy_simulation = add_position_noise(simulation, 0.1)
params, nll = find_hmm_params(simulation,noisy_simulation.arguments.dt)


alpha_2, loglik_2 = forward_algorithm(noisy_simulation,params)
actual_states_noise=[]
actual_states = []
for i in 1:length(noisy_simulation.observables.frames)-1
    push!(actual_states_noise, noisy_simulation.observables.frames[i].molecules[1].state)
end

actual_states_int = Int64.(actual_states)
actual_states_noise_int = Int64.(actual_states_noise)
acurracy_noise = calculate_accuracy(alpha_2, actual_states_noise_int)
println("\nNoisy Observable History model accuracy: $(round(acurracy_noise * 100, digits=2))%")



