using Random
using CairoMakie

function simulate_markov_with_changes(k12, k21, state_changes=10,Δt = 0.1; max_steps=100000)
    
   
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


k12 = 0.1 
k21 = 0.005 
states, steps = simulate_markov_with_changes(k12, k21, 6)

println("Simulation completed with $(length(states)) time points and $steps steps")
println("Number of state changes: $(sum(abs.(diff(states))))")

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], 
    xlabel="Time Steps", 
    ylabel="State",
    title="Two-State Markov Chain with k12=$k12, k21=$k21"
)


lines!(ax, 1:length(states), states, color=:blue, linewidth=1.5)


display(fig)