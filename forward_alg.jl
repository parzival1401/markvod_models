
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

function plot_hmm_results(t, states, observables, actual_states, alpha)
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
    
    display(fig)
    save("algorithm_results.png", fig)
    return fig
end


function calculate_accuracy(alpha, actual_states)
    predicted_states = [argmax(alpha[:, t]) for t in 1:size(alpha, 2)]
    accuracy = mean(predicted_states .== actual_states)
    return accuracy
end


k12, k21 = 0.1, 0.1     
Δt = 0.1               
sz = 1000               
μ1, σ1 = 1.0, 1
μ2, σ2 = 2.0, 1



states, observables, actual_states, T = simulate_hmm(k12, k21, Δt, sz, μ1, σ1, μ2, σ2)
t = 0:Δt:((sz-1)*Δt)

alpha, loglik = forward_algorithm(observables, T, μ1, σ1, μ2, σ2)

plot_hmm_results(t, states, observables, actual_states, alpha)


accuracy = calculate_accuracy(alpha, actual_states)
println("Log-likelihood: ", loglik)
println("State prediction accuracy: ", round(accuracy * 100, digits=2), "%")


t_example = 300
println("\nAt time t = $(t[t_example]):")
println("P(State 1) = $(alpha[1, t_example])")
println("P(State 2) = $(alpha[2, t_example])")
println("True state: $(actual_states[t_example])")
println("Predicted state: $(argmax(alpha[:, t_example]))")

=#