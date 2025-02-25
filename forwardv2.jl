
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


function p_state(o:: Gaus2state,frame;μ=0, σ=1)
    return pdf(Normal(μ, σ), o.observables[frame])
end 
 

function p_state(o::ObservableHist, state,frame;  sigma=0.1, dt=0.01,)
    
    if state == 1
        return compute_free_density(o, frame,sigma=sigma, dt=dt)

    elseif  state==2

        return compute_dimer_density(o, frame,sigma=sigma, dt=dt)
       
    end

end 



function compute_free_density(o::ObservableHist,frame;sigma=0.2, dt=0.01)
   
        dn = sqrt((o.observables.frames[frame].molecules[1].x - o.observables.frames[frame].molecules[2].x)^2 
        + (o.observables.frames[frame].molecules[1].y - o.observables.frames[frame].molecules[2].y)^2)
        println("\n dn free: $dn")

        dn_1 = sqrt((o.observables.frames[frame+1].molecules[1].x - o.observables.frames[frame+1].molecules[2].x)^2 
        + (o.observables.frames[frame+1].molecules[1].y - o.observables.frames[frame+1].molecules[2].y)^2)
        println("\n dn-1 free: $dn_1")
        density_val = (dn/sigma^2) *  (exp((-(dn^2) - (dn_1)^2))/sigma^2) * modified_bessel(dt, dn, dn_1,sigma)

        println("\nfreee density val: $density_val")
       
    return density_val
end

function compute_dimer_density(o::ObservableHist,frame;sigma=0.2, dt=0.01,)
    
    
    dn = sqrt((o.observables.frames[frame].molecules[1].x - o.observables.frames[frame].molecules[2].x)^2 
        + (o.observables.frames[frame].molecules[1].y - o.observables.frames[frame].molecules[2].y)^2)


    density_val = ((dn)/sigma^2) *  exp((-(o.arguments.d_dimer^2) - (dn)^2)/sigma^2) *  modified_bessel(dt, o.arguments.r_react, dn, sigma)
    println("\nfreee dimer density : $density_val")
    return density_val
end



function modified_bessel(dt, d1, d2, σ)
    result = 0.0
    
    x = (d1 * d2) / (σ^2)
    
    for θ in 0:dt:2*pi
        result += exp(x * cos(θ))
    end

    result *= dt / (2*pi)
    
    return result
end

function forward_algorithm(observables::ObservableHist)

    N = length(observables.observables.frames)-1
    alpha = zeros(2, N)
    scale = zeros(N)  
    
    
    alpha[1, 1] = p_state(observables,1,1)
    alpha[2, 1] = p_state(observables,2,1)

    println(" alpha[1, 1]: $(alpha[1, 1])")
    println(" alpha[2, 1]: $(alpha[2, 1])")
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]

    Δt=observables.arguments.dt

    k_on=0.1
    T = [1-(k_on*Δt) k_on*Δt; 
    observables.arguments.k_off*Δt 1-(observables.arguments.k_off*Δt)]

    for t in 2:N
       
        e1 = p_state(observables, 1,t)
        e2 = p_state(observables, 2,t)
        
        
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= scale[t]
        if t>N-5
            println("e1:$e1")
            println("e2:$e2")
            println("alpha[1,t]: $(alpha[1, t])")
            println("alpha[2,t]: $(alpha[2, t])")
            println("scale[t]: $(scale[t])")
            

        end 
    end
    
    
    loglikelihood = sum(log.(scale))
    
    return alpha, loglikelihood

end 

function forward_algorithm(observations::Gaus2state, T, μ1, σ1, μ2, σ2)
    N = length(observations.observables)
    alpha = zeros(2, N)
    scale = zeros(N)  
    
    
    alpha[1, 1] = p_state(observations,1, μ=μ1, σ=σ1)
    alpha[2, 1] = p_state(observations,1, μ=μ2, σ=σ2)
    
    
    scale[1] = sum(alpha[:, 1])
    alpha[:, 1] ./= scale[1]
    
   
    for t in 2:N
       
        e1 = p_state(observations,t, μ=μ1, σ=σ1)
        e2 = p_state(observations,t, μ=μ2, σ=σ2)
        
        
        alpha[1, t] = (alpha[1, t-1]*T[1,1] + alpha[2, t-1]*T[2,1]) * e1
        alpha[2, t] = (alpha[1, t-1]*T[1,2] + alpha[2, t-1]*T[2,2]) * e2
        
        
        scale[t] = sum(alpha[:, t])
        alpha[:, t] ./= scale[t]


        
    end
    
    
    loglikelihood = sum(log.(scale))
    
    return alpha, loglikelihood
end

function calculate_accuracy(alpha::Matrix{Float64}, actual_states::Vector{Int64})
    
    n_timesteps = min(size(alpha, 2), length(actual_states))
    
   
    predicted_states = zeros(Int64, n_timesteps)
    for t in 1:n_timesteps
        predicted_states[t] = argmax(alpha[:, t])
    end
    
   
    actual_states_trimmed = actual_states[1:n_timesteps]
  
    correct = sum(predicted_states .== actual_states_trimmed)
    accuracy = correct / n_timesteps
    
   
  
    return accuracy
end



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


