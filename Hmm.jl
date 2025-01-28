using Distributions
using GLMakie
using Random

# Set random seed for reproducibility
Random.seed!(123)

# Parameters
seq_length = 1000
frequency = 0.05  # Frequency of the state oscillation

# Define the Gaussian distributions for each state
state_1_dist = Normal(0, 1)
state_2_dist = Normal(3, 1)

# Generate time vector
t = 1:seq_length

# Generate the state sequence
state_values = sin.(2*3.14 * frequency .* t)
states = [val > 0 ? 1 : 2 for val in state_values]

# Generate observations based on the states
observations = zeros(seq_length)
for i in 1:seq_length
    observations[i] = rand(states[i] == 1 ? state_1_dist : state_2_dist)
end

# Create the plot
fig = Figure(size=(1000, 600))
ax1 = Axis(fig[1, 1], ylabel="Observation", title="Oscillating State Data with Gaussian Distributions")
ax2 = Axis(fig[2, 1], ylabel="State", xlabel="Time Step")

# Plot observations
scatter!(ax1, t, observations, color=:blue, markersize=3, label="Observations")

# Plot states
scatter!(ax2, t, states, color=:green, markersize=5, label="States")

# Set y-axis limits and ticks for the state plot
ylims!(ax2, 0.5, 2.5)
ax2.yticks = (1:2, ["State 1", "State 2"])

# Add legends
axislegend(ax1, position=:rt)
axislegend(ax2, position=:rt)

# Link x-axes
linkyaxes!(ax1, ax2)

# Display the plot
display(fig)

# Calculate the proportion of time spent in each state
state_1_proportion = sum(states .== 1) / seq_length
state_2_proportion = 1 - state_1_proportion
println("Proportion of time in state 1: ", state_1_proportion)
println("Proportion of time in state 2: ", state_2_proportion)