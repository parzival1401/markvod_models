using CairoMakie

# Create the data as plain arrays
frequencies = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
gains = [-11.65, -11.65, -11.65, -11.44, -11.06,- 7.92,- 2.279,- 0.774, -0.256]

# Create the plot
fig = Figure(size=(800, 500))
ax = CairoMakie.Axis(fig[1, 1],
    xscale=log10,
    xlabel="Frequency (Hz)",
    ylabel="Gain",
    title="Frequency Response of Amplifier circuit 1",
    xticks = LogTicks(LinearTicks(6)))

# Plot the data
lines!(ax, frequencies, gains, color=:blue, linewidth=2)
scatter!(ax, frequencies, gains, color=:blue, markersize=10)

# Add grid lines
#ax.xgrid = true
#ax.ygrid = true

# Set axis limits
#xlims!(ax, 10, 10^7)  # 10Hz to 10MHz
#ylims!(ax, 0, maximum(gains) * 1.1)  # From 0 to slightly above max gain

# Save the plot
save("frequency_response.png", fig)

fig

