using Random
using CairoMakie



function simulate_states(k12, k21, state_changes=10,Δt = 0.1; max_steps=100000)
    
   
    p12 = k12 * Δt
    p21 = k21 * Δt
    
  
    states = Int[]
    current_state = 1
    push!(states, current_state)
    
    changes_count = 0
    steps = 0
    
    
    while changes_count < state_changes && steps < max_steps
        steps += 1
        r = rand()
        
       
        next_state = current_state
        if current_state == 1
            if r < p12
                next_state = 2
            end
        else
            if r < p21
                next_state = 1
            end
        end
        
        
        if next_state != current_state
            changes_count += 1
        end
        
        current_state = next_state
        push!(states, current_state)
    end
    
   
    
    
    return states, steps
end

function get_state_transitions(states::Vector{Int})
    transitions = []
    for i in 2:length(states)
        if states[i] != states[i - 1]
            t = (i - 1) 
            from_state = states[i - 1]
            to_state = states[i]
            push!(transitions, ( t, from_state, to_state))
        end
    end
    return transitions
end
#=

k12 = 0.5
k21 = 0.15 
states, steps = simulate_states(k12, k21, 6)
transitions = get_state_transitions(states)

for tr in transitions
    println("At t = $(tr.time): State changed from $(tr.from) → $(tr.to)")
end

println("Simulation completed with $(length(states)) time points and $steps steps")
println("Number of state changes: $(sum(abs.(diff(states))))")

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1],  xlabel="Time Steps",  ylabel="State", title="Two-State Markov Chain with k12=$k12, k21=$k21")


lines!(ax, 1:length(states), states, color=:blue, linewidth=1.5)

save("generated_data/markov_chain_simulation.png", fig)
display(fig)

=#