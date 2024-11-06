using Random
using CairoMakie
using Distributions
using BenchmarkTools
using HiddenMarkovModels
k12 = 0.1
k21 = 0.1
Δt = 0.1

T = [1-(k12*Δt) k12*Δt; 
     k21*Δt 1-(k21*Δt)]

pi0 = [1, 0]


μ1, σ1 = 1.0, 0.5 
μ2, σ2 = 2.0, 0.4  

sz = 1000
states = zeros(Float64, sz) 
observables = zeros(Float64, sz)
current_state = 1  

for i in 1:sz
    r = rand()
    if current_state == 1
        if r < T[1,1]
            current_state = 1
            states[i] = μ1
            observables[i] = rand(Normal(μ1, σ1))
        else
            current_state = 2
            states[i] = μ2
            observables[i] = rand(Normal(μ2, σ2))
        end
    else  
        if r < T[2,1]
            current_state = 1
            states[i] = μ1
            observables[i] = rand(Normal(μ1, σ1))
        else
            current_state = 2
            states[i] = μ2
            observables[i] = rand(Normal(μ2, σ2))
        end
    end
end

t = 0:Δt:((sz-1)*Δt)

fig = Figure(size=(1200, 600))
ax = Axis(fig[1, 1],
    xlabel = "Time",
    ylabel = "Value",
    title = " Observables Over Time")

lines!(ax, t, observables, color = (:blue, 0.5), label = "Observable")
stairs!(ax, t, states, color = (:red, 0.5), label = "State")

display(fig)

save("hmm_observables.png", fig)




