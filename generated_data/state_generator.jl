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
function time_sequence(states::Vector{Int})
    if isempty(states)
        return []
    end
    
    state_time_sequence = []
    current_state = states[1]
    current_count = 1
    
    # Iterate through states starting from the second element
    for i in 2:length(states)
        if states[i] == current_state
            # Same state, increment count
            current_count += 1
        else
            # State changed, record the previous state and its duration
            push!(state_time_sequence, current_state => current_count)
            # Start counting the new state
            current_state = states[i]
            current_count = 1
        end
    end
    
    # Don't forget to add the last state
    push!(state_time_sequence, current_state => current_count)
    
    return state_time_sequence
end
function divide_into_sections(transitions::Vector, section_size::Int=3)
    sections = []
    current_section = []
    
    for (i, transition) in enumerate(transitions)
        push!(current_section, transition)
        
        # When we reach section_size or the end of the sequence, add the section
        if i % section_size == 0 || i == length(transitions)
            push!(sections, current_section)
            current_section = []
        end
    end
    
    return sections
end

function time_sequence_with_split(states::Vector{Int})
    if isempty(states)
        return []
    end
    
    # Calculate raw state durations
    durations = []
    current_state = states[1]
    current_count = 1
    
    for i in 2:length(states)
        if states[i] == current_state
            current_count += 1
        else
            push!(durations, current_state => current_count)
            current_state = states[i]
            current_count = 1
        end
    end
    push!(durations, current_state => current_count)
    
    # Create sections
    sections = []
    current_section = []
    
    # Process the first element normally (won't be split)
    if !isempty(durations)
        push!(current_section, durations[1])
    end
    
    # Process the rest of the durations
    i = 2
    while i <= length(durations)
        state, duration = durations[i].first, durations[i].second
        
        if state == 1 && duration > 1
            # Split state 1 in half
            half = div(duration, 2)
            
            # Add first half to current section
            push!(current_section, 1 => half)
            
            # Check if section is full
            if length(current_section) == 3
                push!(sections, current_section)
                current_section = []
            end
            
            # Start new section with second half
            push!(current_section, 1 => (duration - half))
        else
            # Regular state, add normally
            push!(current_section, state => duration)
        end
        
        # If section is full, add it and start a new one
        if length(current_section) == 3
            push!(sections, current_section)
            current_section = []
        end
        
        i += 1
    end
    
    # Add any remaining elements
    if !isempty(current_section)
        push!(sections, current_section)
    end
    
    return sections
end
#=
k12 = 0.5
k21 = 0.15 
states, steps = simulate_states(k12, k21, 6)
transitions = get_state_transitions(states)
time_in_state = time_sequence(states)
sections = divide_into_sections(time_in_state, 3)

split= time_sequence_with_split(states)

for tr in transitions
    println("At t = $(tr.time): State changed from $(tr.from) → $(tr.to)")
end

println("Simulation completed with $(length(states)) time points and $steps steps")
println("Number of state changes: $(sum(abs.(diff(states))))")

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1],  xlabel="Time Steps",  ylabel="State", title="Two-State Markov Chain with k12=$k12, k21=$k21")


lines!(ax, 1:length(states), states, color=:blue, linewidth=1.5)

save("generated_data/markov_chain_simulation.png", fig)
display(fig)=#

