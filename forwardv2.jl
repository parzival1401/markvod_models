
using Random
using CairoMakie
using Distributions
using SMLMSim




abstract type Abstract_obs end


struct ObservableHist<:Abstract_obs
    obserbvables::Array
end 
struct Gaus2state <:Abstract_obs
    observables::Float64
end

function p_state(o:: Gaus2state,μ=0, σ=1)
    return pdf(Normal(μ, σ), o.observables)
end 
 

function p_state(o::ObservableHist)
    return  

end 


function run_simulation(;density=0.02, t_max=25, box_size=10, k_off=0.3, r_react=2)
    state_history, args = SMLMSim.InteractionDiffusion.smoluchowski(density=density,
        t_max=t_max, 
        box_size=box_size,
        k_off=k_off,
        r_react=r_react
    )
    dimer_history = SMLMSim.get_dimers(state_history)
    return state_history, args, dimer_history
end
 
function compute_free_density(pos_x, pos_y, σ, dt=0.01)
    distances = zeros(size(pos_x))
    density_vals = zeros(size(pos_x))
    
    for i in 1:(length(pos_x)-1)
        distances[i] = sqrt((pos_x[i+1] - pos_x[i])^2 + (pos_y[i+1] - pos_y[i])^2)
    end
    
    for i in 1:(length(distances)-1)
        density_vals[i] = (distances[i]/σ^2) * 
                         exp((-(distances[i+1]^2) - (distances[i]^2))/σ^2) * 
                         modified_bessel(dt, distances[i+1], distances[i], σ)
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
        density_vals[i] = ((distances[i])/σ^2) *  exp((-(dimer_length^2) - (distances[i])^2)/σ^2) *  modified_bessel(dt, dimer_length, distances[i], σ)
    end
    
    return density_vals, distances
end

function compute_density(d1, d2, σ)
    return (d1/σ^2) * exp((-(d2^2) - (d1^2))/σ^2) * modified_bessel(0.01, d2, d1, σ)
end

function modified_bessel(dt, d1, d2, σ)
    result = 0
    for θ in 0:dt:2π
        result += (1/(2π)) * exp((((d1 * d2)^2) * cos(θ))/σ^2)
    end
    return result
end


function forward_algorithm(observations, T, μ1, σ1, μ2, σ2)
    N = length(observations)
    alpha = zeros(2, N)
    scale = zeros(N)  
    
    
    alpha[1, 1] = pdf(Normal(μ1, σ1), observations[1])
    alpha[2, 1] = pdf(Normal(μ2, σ2), observations[1])
    
    
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
   
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


state_history, args, dimer_history = run_simulation()
