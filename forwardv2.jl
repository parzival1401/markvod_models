
using Random
using CairoMakie
using Distributions


abstract type Abstract_obs end


struct ObservableHist<:Abstract_obs
    obserbvables::Vector{:Abstract_obs}
end 
struct Gaus2state <:Abstract_obs
    observables::Float64
end

function p_state(o:: Gaus2state,μ=0, σ=1)
    return pdf(Normal(μ, σ), o.observables)
end 
 

function p_state()
    return  

end 
 
function compute_free_density(pos_x, pos_y, σ, dt=0.01)
    distances = zeros(size(pos_x))
    density_vals = zeros(size(pos_x))
    
    for i in 1:(length(pos_x)-1)
        distances[i] = (pos_x[i+1] - pos_x[i])^2 + (pos_y[i+1] - pos_y[i])^2
    end
    
    for i in 1:(length(distances)-1)
        density_vals[i] = (sqrt(distances[i])/σ^2) * 
                         exp((-distances[i+1] - distances[i])/σ^2) * 
                         modified_bessel(dt, distances[i+1], distances[i], σ)
    end
    
    return density_vals, distances
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


