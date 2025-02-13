
using Random
using CairoMakie
using Distributions
using SMLMSim




abstract type Abstract_obs end



struct Gaus2state <:Abstract_obs
    observables::Vector{Float64}
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


simulation = run_simulation()

function p_state(o:: Gaus2state,μ=0, σ=1)
    return pdf(Normal(μ, σ), o.observables)
end 
 

function p_state(o::ObservableHist, molecule,frame,dl_dimer=0, sigma=0.1, dt=0.01)
    
    if o.observables.frames[frame].molecules[molecule].state == 1
        return compute_free_density(o, molecule, frame,sigma=sigma, dt=dt)

    elseif  o.observables.frames[frame].molecules[molecule].state == 2

        return compute_free_density(o, molecule, frame,sigma=sigma, dt=dt,dimer_length=dl_dimer)
       
    end


end 



function compute_free_density(o::ObservableHist, molecule,frame,sigma=0.1, dt=0.01)
   
        dn = sqrt((o.observables.frames[frame].molecules[molecule].x - o.observables.frames[frame].molecules[molecule].x)^2 
        + (o.observables.frames[frame].molecules[molecule].y - o.observables.frames[frame].molecules[molecule].y)^2)

        dn_1 = sqrt((o.observables.frames[frame-1].molecules[molecule].x - o.observables.frames[frame-1].molecules[molecule].x)^2 
        + (o.observables.frames[frame-1].molecules[molecule].y - o.observables.frames[frame-1].molecules[molecule].y)^2)

        density_val = (dn/sigma^2) *  (exp((-(dn^2) - (dn_)^2))/sigma^2) * modified_bessel(dt, dn, dn_1,sigma)
    
    
    return density_val
end

function compute_dimer_density(o::ObservableHist, molecule,frame,dimer_length,sigma=0.1, dt=0.01,)
    
    
    dn = sqrt((o.observables.frames[frame].molecules[molecule].x - o.observables.frames[frame].molecules[molecule].x)^2 
        + (o.observables.frames[frame].molecules[molecule].y - o.observables.frames[frame].molecules[molecule].y)^2)


    density_val = ((dn)/sigma^2) *  exp((-(dimer_length^2) - (dn)^2)/sigma^2) *  modified_bessel(dt, dimer_length, dn, sigma)
    
    return density_val
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





