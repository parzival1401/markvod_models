using Random
using CairoMakie
using Distributions

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


function viterbi_algorithm(observations, T, μ1, σ1, μ2, σ2)
    N = length(observations)
    Δ = zeros(2, N)    
    Ψ = zeros(Int, 2, N) 
    
    
    Δ[1, 1] = log(pdf(Normal(μ1, σ1), observations[1]))
    Δ[2, 1] = log(pdf(Normal(μ2, σ2), observations[1]))
    
   
    for t in 2:N
       
        e1 = log(pdf(Normal(μ1, σ1), observations[t]))
        e2 = log(pdf(Normal(μ2, σ2), observations[t]))
        
        
        trans_probs = [Δ[1, t-1] + log(T[1,1]), Δ[2, t-1] + log(T[2,1])]
        Δ[1, t] = maximum(trans_probs) + e1
        Ψ[1, t] = argmax(trans_probs)
        
        trans_probs = [Δ[1, t-1] + log(T[1,2]),  Δ[2, t-1] + log(T[2,2])]
        Δ[2, t] = maximum(trans_probs) + e2
        Ψ[2, t] = argmax(trans_probs)
    end
    
    
    viterbi_states = zeros(Int, N)
    
    
    viterbi_states[N] = argmax(Δ[:, N])
    max_prob = maximum(Δ[:, N])
    
    for t in N-1:-1:1
        viterbi_states[t] = Ψ[viterbi_states[t+1], t+1]
    end
    
    return viterbi_states, max_prob
end


function plot_viterbi_results(t, states, observables, actual_states, viterbi_states)
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
        ylabel = "State",
        title = "Viterbi Path vs True States")
    
    stairs!(ax2, t, actual_states, color = (:red, 0.5), label = "True States")
    stairs!(ax2, t, viterbi_states, color = (:blue, 0.5), label = "Viterbi Path")
    
    axislegend(ax2)
    ylims!(ax2, 0.5, 2.5)
    
    display(fig)
end


k12, k21 = 0.1, 0.1     
Δt = 0.1                
sz = 1000               
μ1, σ1 = 1.0, 0.5      
μ2, σ2 = 2.0, 0.4      


states, observables, actual_states, T = simulate_hmm(k12, k21, Δt, sz, μ1, σ1, μ2, σ2)
t = 0:Δt:((sz-1)*Δt)


viterbi_states, max_prob = viterbi_algorithm(observables, T, μ1, σ1, μ2, σ2)


plot_viterbi_results(t, states, observables, actual_states, viterbi_states)


accuracy = mean(viterbi_states .== actual_states)
println("\nViterbi Algorithm Statistics:")
println("Path log-probability: ", max_prob)
println("State prediction accuracy: ", round(accuracy * 100, digits=2), "%")