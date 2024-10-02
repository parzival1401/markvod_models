using Random
using LinearAlgebra  # Add this for transpose operation

# Define the states
states = ["Sunny", "Cloudy", "Rainy"]

# Define the transition matrix
transition_matrix = [
    0.7 0.2 0.1;  # Sunny  -> Sunny (0.7), Cloudy (0.2), Rainy (0.1)
    0.3 0.5 0.2;  # Cloudy -> Sunny (0.3), Cloudy (0.5), Rainy (0.2)
    0.2 0.4 0.4   # Rainy  -> Sunny (0.2), Cloudy (0.4), Rainy (0.4)
]

# Function to simulate the next state
function next_state(current_state, transition_matrix)
    probabilities = transition_matrix[current_state, :]
    r = rand()
    cumulative_prob = 0.0
    for (i, prob) in enumerate(probabilities)
        cumulative_prob += prob
        if r <= cumulative_prob
            return i
        end
    end
    return length(probabilities)  # This should never happen, but just in case
end

# Function to simulate the Markov chain
function simulate_markov_chain(initial_state, transition_matrix, n_steps)
    chain = [initial_state]
    current_state = initial_state
    for _ in 1:n_steps-1
        current_state = next_state(current_state, transition_matrix)
        push!(chain, current_state)
    end
    return chain
end

# Simulate the weather for 10 days
initial_state = 1  # 1 corresponds to "Sunny"
days = 10
weather_forecast = simulate_markov_chain(initial_state, transition_matrix, days)

println("Weather forecast for the next $days days:")
for (day, weather) in enumerate(weather_forecast)
    println("Day $day: $(states[weather])")
end

# Function to calculate the stationary distribution
function calculate_stationary_distribution(transition_matrix, n_iterations=1000)
    n_states = size(transition_matrix, 1)
    distribution = ones(1, n_states) / n_states  # Make this a row vector
    
    for _ in 1:n_iterations
        distribution = distribution * transition_matrix
    end
    
    return vec(distribution)  # Convert back to a column vector for consistency
end

# Calculate the stationary distribution
stationary_dist = calculate_stationary_distribution(transition_matrix)

println("\nStationary distribution:")
for (state, prob) in zip(states, stationary_dist)
    println("$state: $(round(prob, digits=3))")
end