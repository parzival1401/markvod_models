include("final_implementation.jl")
using Statistics
using Optim
using CairoMakie

# ========================================
# ERROR METRICS FUNCTIONS
# ========================================

"""
Calculate RMSE from existing error_val data
"""
function calculate_rmse_from_errors()
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    
    rmse_k_on = sqrt(mean(k_on_errors.^2))
    rmse_k_off = sqrt(mean(k_off_errors.^2))
    rmse_overall = sqrt(mean([k_on_errors; k_off_errors].^2))
    
    return (k_on=rmse_k_on, k_off=rmse_k_off, overall=rmse_overall)
end

"""
Calculate MAE from existing error_val data
"""
function calculate_mae_from_errors()
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    
    mae_k_on = mean(k_on_errors)
    mae_k_off = mean(k_off_errors)
    mae_overall = mean([k_on_errors; k_off_errors])
    
    return (k_on=mae_k_on, k_off=mae_k_off, overall=mae_overall)
end

"""
Calculate relative errors as percentages
"""
function calculate_relative_errors()
    rel_errors_k_on = []
    rel_errors_k_off = []
    
    for i in 1:length(error_val)
        true_k_on = k_on_v[i]
        true_k_off = k_off_v[i]
        abs_error_k_on = error_val[i][1]
        abs_error_k_off = error_val[i][2]
        
        rel_k_on = (abs_error_k_on / true_k_on) * 100
        rel_k_off = (abs_error_k_off / true_k_off) * 100
        
        push!(rel_errors_k_on, rel_k_on)
        push!(rel_errors_k_off, rel_k_off)
    end
    
    return (k_on=rel_errors_k_on, k_off=rel_errors_k_off)
end

"""
Analyze convergence from optimization results
"""
function analyze_convergence()
    converged_count = 0
    total_iterations = []
    
    for result in optimize_val
        if Optim.converged(result.optimization_result)
            converged_count += 1
        end
        push!(total_iterations, Optim.iterations(result.optimization_result))
    end
    
    convergence_rate = (converged_count / length(optimize_val)) * 100
    avg_iterations = mean(total_iterations)
    
    return (rate=convergence_rate, avg_iterations=avg_iterations, converged=converged_count, total=length(optimize_val))
end

"""
Print comprehensive error summary
"""
function print_error_summary()
    println("\n" * "="^50)
    println("ERROR METRICS SUMMARY")
    println("="^50)
    
    # Calculate metrics
    rmse = calculate_rmse_from_errors()
    mae = calculate_mae_from_errors()
    rel_errors = calculate_relative_errors()
    convergence = analyze_convergence()
    
    # Print results
    println("📊 Overall Performance:")
    println("   Trials: $tries")
    println("   Convergence: $(convergence.converged)/$(convergence.total) ($(round(convergence.rate, digits=1))%)")
    println("   Avg iterations: $(round(convergence.avg_iterations, digits=1))")
    
    println("\n📏 Absolute Error Metrics:")
    println("   RMSE k_on:  $(round(rmse.k_on, digits=4))")
    println("   RMSE k_off: $(round(rmse.k_off, digits=4))")
    println("   RMSE overall: $(round(rmse.overall, digits=4))")
    
    println("   MAE k_on:   $(round(mae.k_on, digits=4))")
    println("   MAE k_off:  $(round(mae.k_off, digits=4))")
    println("   MAE overall: $(round(mae.overall, digits=4))")
    
    println("\n📈 Relative Error Statistics:")
    println("   k_on:  $(round(mean(rel_errors.k_on), digits=1))% ± $(round(std(rel_errors.k_on), digits=1))%")
    println("   k_off: $(round(mean(rel_errors.k_off), digits=1))% ± $(round(std(rel_errors.k_off), digits=1))%")
    
    # Best and worst cases
    overall_errors = [err[1] + err[2] for err in error_val]
    best_idx = argmin(overall_errors)
    worst_idx = argmax(overall_errors)
    
    println("\n🏆 Best trial: #$best_idx")
    println("   k_on error: $(round(error_val[best_idx][1], digits=4)) ($(round(rel_errors.k_on[best_idx], digits=1))%)")
    println("   k_off error: $(round(error_val[best_idx][2], digits=4)) ($(round(rel_errors.k_off[best_idx], digits=1))%)")
    
    println("\n❌ Worst trial: #$worst_idx") 
    println("   k_on error: $(round(error_val[worst_idx][1], digits=4)) ($(round(rel_errors.k_on[worst_idx], digits=1))%)")
    println("   k_off error: $(round(error_val[worst_idx][2], digits=4)) ($(round(rel_errors.k_off[worst_idx], digits=1))%)")
    
    # Noise correlation
    noise_error_corr = cor(noise_v, overall_errors)
    println("\n🔊 Noise-Error Correlation: $(round(noise_error_corr, digits=3))")
end

# ========================================
# GRAPHING FUNCTIONS FOR ERROR VISUALIZATION
# ========================================

"""
Plot absolute errors for each trial
"""
function plot_absolute_errors()
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    
    fig = Figure(size=(1000, 600))
    
    # Plot 1: Absolute errors by trial
    ax1 = Axis(fig[1, 1], 
               title="Absolute Errors by Trial", 
               xlabel="Trial Number", 
               ylabel="Absolute Error")
    
    scatter!(ax1, 1:tries, k_on_errors, color=:blue, markersize=10, label="k_on error")
    scatter!(ax1, 1:tries, k_off_errors, color=:red, markersize=10, label="k_off error")
    axislegend(ax1, position=:rt)
    
    # Plot 2: Error distribution
    ax2 = Axis(fig[1, 2], 
               title="Error Distribution", 
               xlabel="Absolute Error", 
               ylabel="Frequency")
    
    hist!(ax2, k_on_errors, bins=5, color=(:blue, 0.5), label="k_on")
    hist!(ax2, k_off_errors, bins=5, color=(:red, 0.5), label="k_off")
    axislegend(ax2, position=:rt)
    
    save("absolute_errors_plot.png", fig)
    display(fig)
    return fig
end

"""
Plot relative errors as percentages
"""
function plot_relative_errors()
    rel_errors = calculate_relative_errors()
    
    fig = Figure(size=(1000, 600))
    
    # Plot 1: Relative errors by trial
    ax1 = Axis(fig[1, 1], 
               title="Relative Errors by Trial", 
               xlabel="Trial Number", 
               ylabel="Relative Error (%)")
    
    scatter!(ax1, 1:tries, rel_errors.k_on, color=:blue, markersize=10, label="k_on error (%)")
    scatter!(ax1, 1:tries, rel_errors.k_off, color=:red, markersize=10, label="k_off error (%)")
    axislegend(ax1, position=:rt)
    
    # Plot 2: Box plot comparison
    ax2 = Axis(fig[1, 2], 
               title="Relative Error Distribution", 
               xlabel="Parameter", 
               ylabel="Relative Error (%)")
    
    boxplot!(ax2, fill(1, length(rel_errors.k_on)), rel_errors.k_on, color=:blue, width=0.4)
    boxplot!(ax2, fill(2, length(rel_errors.k_off)), rel_errors.k_off, color=:red, width=0.4)
    ax2.xticks = ([1, 2], ["k_on", "k_off"])
    
    save("relative_errors_plot.png", fig)
    display(fig)
    return fig
end

"""
Plot noise vs error correlation
"""
function plot_noise_correlation()
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    overall_errors = [err[1] + err[2] for err in error_val]
    
    fig = Figure(size=(1200, 400))
    
    # Plot 1: k_on error vs noise
    ax1 = Axis(fig[1, 1], 
               title="k_on Error vs Noise Level", 
               xlabel="Noise Level (σ)", 
               ylabel="k_on Absolute Error")
    
    scatter!(ax1, noise_v, k_on_errors, color=:blue, markersize=8)
    
    # Add trend line
    z = polyfit(noise_v, k_on_errors, 1)
    x_trend = range(minimum(noise_v), maximum(noise_v), length=100)
    y_trend = z[1] .* x_trend .+ z[2]
    lines!(ax1, x_trend, y_trend, color=:blue, linestyle=:dash, alpha=0.7)
    
    # Plot 2: k_off error vs noise
    ax2 = Axis(fig[1, 2], 
               title="k_off Error vs Noise Level", 
               xlabel="Noise Level (σ)", 
               ylabel="k_off Absolute Error")
    
    scatter!(ax2, noise_v, k_off_errors, color=:red, markersize=8)
    
    # Add trend line
    z = polyfit(noise_v, k_off_errors, 1)
    y_trend = z[1] .* x_trend .+ z[2]
    lines!(ax2, x_trend, y_trend, color=:red, linestyle=:dash, alpha=0.7)
    
    # Plot 3: Overall error vs noise
    ax3 = Axis(fig[1, 3], 
               title="Total Error vs Noise Level", 
               xlabel="Noise Level (σ)", 
               ylabel="Total Absolute Error")
    
    scatter!(ax3, noise_v, overall_errors, color=:purple, markersize=8)
    
    # Add trend line
    z = polyfit(noise_v, overall_errors, 1)
    y_trend = z[1] .* x_trend .+ z[2]
    lines!(ax3, x_trend, y_trend, color=:purple, linestyle=:dash, alpha=0.7)
    
    # Add correlation coefficients as text
    corr_k_on = round(cor(noise_v, k_on_errors), digits=3)
    corr_k_off = round(cor(noise_v, k_off_errors), digits=3)
    corr_total = round(cor(noise_v, overall_errors), digits=3)
    
    text!(ax1, 0.05, 0.95, "r = $corr_k_on", space=:relative, fontsize=12)
    text!(ax2, 0.05, 0.95, "r = $corr_k_off", space=:relative, fontsize=12)
    text!(ax3, 0.05, 0.95, "r = $corr_total", space=:relative, fontsize=12)
    
    save("noise_correlation_plot.png", fig)
    display(fig)
    return fig
end

"""
Plot convergence analysis
"""
function plot_convergence_analysis()
    convergence_data = analyze_convergence()
    iterations = [Optim.iterations(result.optimization_result) for result in optimize_val]
    converged = [Optim.converged(result.optimization_result) for result in optimize_val]
    loglikelihoods = [result.loglikelihood for result in optimize_val]
    
    fig = Figure(size=(1200, 800))
    
    # Plot 1: Iterations by trial
    ax1 = Axis(fig[1, 1], 
               title="Optimization Iterations by Trial", 
               xlabel="Trial Number", 
               ylabel="Iterations")
    
    colors = [conv ? :green : :red for conv in converged]
    scatter!(ax1, 1:tries, iterations, color=colors, markersize=10)
    
    # Add legend
    scatter!(ax1, [0], [0], color=:green, markersize=10, label="Converged")
    scatter!(ax1, [0], [0], color=:red, markersize=10, label="Not Converged")
    axislegend(ax1, position=:rt)
    
    # Plot 2: Log-likelihood by trial
    ax2 = Axis(fig[1, 2], 
               title="Log-likelihood by Trial", 
               xlabel="Trial Number", 
               ylabel="Log-likelihood")
    
    scatter!(ax2, 1:tries, loglikelihoods, color=colors, markersize=10)
    
    # Plot 3: Convergence pie chart (using bar plot)
    ax3 = Axis(fig[2, 1], 
               title="Convergence Summary", 
               xlabel="Status", 
               ylabel="Count")
    
    conv_counts = [sum(converged), sum(.!converged)]
    barplot!(ax3, [1, 2], conv_counts, color=[:green, :red])
    ax3.xticks = ([1, 2], ["Converged", "Failed"])
    
    # Plot 4: Iterations histogram
    ax4 = Axis(fig[2, 2], 
               title="Iterations Distribution", 
               xlabel="Number of Iterations", 
               ylabel="Frequency")
    
    hist!(ax4, iterations, bins=5, color=:blue, alpha=0.7)
    
    save("convergence_analysis_plot.png", fig)
    display(fig)
    return fig
end

"""
Plot true vs estimated k_on values
"""
function plot_kon_comparison()
    estimated_k_on = [result.k_on for result in optimize_val]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              title="True vs Optimized k_on Values", 
              xlabel="True k_on", 
              ylabel="Optimized k_on",
              aspect=1)
    
    # Scatter plot of true vs estimated
    scatter!(ax, k_on_v, estimated_k_on, color=:blue, markersize=12, alpha=0.7, label="Data points")
    
    # Perfect correlation line (y = x)
    min_val = min(minimum(k_on_v), minimum(estimated_k_on))
    max_val = max(maximum(k_on_v), maximum(estimated_k_on))
    lines!(ax, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=2, label="Perfect correlation")
    
    # Calculate and add best fit line
    z = polyfit(k_on_v, estimated_k_on, 1)
    x_fit = range(minimum(k_on_v), maximum(k_on_v), length=100)
    y_fit = z[1] .* x_fit .+ z[2]
    lines!(ax, x_fit, y_fit, color=:red, linewidth=2, alpha=0.8, label="Best fit")
    
    # Add statistics text
    corr_k_on = round(cor(k_on_v, estimated_k_on), digits=3)
    rmse_k_on = round(sqrt(mean((k_on_v .- estimated_k_on).^2)), digits=4)
    mae_k_on = round(mean(abs.(k_on_v .- estimated_k_on)), digits=4)
    
    text!(ax, 0.05, 0.95, "Correlation: r = $corr_k_on", space=:relative, fontsize=14, color=:black)
    text!(ax, 0.05, 0.88, "RMSE = $rmse_k_on", space=:relative, fontsize=14, color=:black)
    text!(ax, 0.05, 0.81, "MAE = $mae_k_on", space=:relative, fontsize=14, color=:black)
    text!(ax, 0.05, 0.74, "Slope = $(round(z[1], digits=3))", space=:relative, fontsize=14, color=:black)
    
    axislegend(ax, position=:rb)
    
    save("kon_comparison.png", fig)
    display(fig)
    println("📊 k_on comparison plot saved as 'kon_comparison.png'")
    return fig
end

"""
Plot true vs estimated k_off values
"""
function plot_koff_comparison()
    estimated_k_off = [result.k_off for result in optimize_val]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              title="True vs Optimized k_off Values", 
              xlabel="True k_off", 
              ylabel="Optimized k_off",
              aspect=1)
    
    # Scatter plot of true vs estimated
    scatter!(ax, k_off_v, estimated_k_off, color=:red, markersize=12, alpha=0.7, label="Data points")
    
    # Perfect correlation line (y = x)
    min_val = min(minimum(k_off_v), minimum(estimated_k_off))
    max_val = max(maximum(k_off_v), maximum(estimated_k_off))
    lines!(ax, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=2, label="Perfect correlation")
    
    # Calculate and add best fit line
    z = polyfit(k_off_v, estimated_k_off, 1)
    x_fit = range(minimum(k_off_v), maximum(k_off_v), length=100)
    y_fit = z[1] .* x_fit .+ z[2]
    lines!(ax, x_fit, y_fit, color=:blue, linewidth=2, alpha=0.8, label="Best fit")
    
    # Add statistics text
    corr_k_off = round(cor(k_off_v, estimated_k_off), digits=3)
    rmse_k_off = round(sqrt(mean((k_off_v .- estimated_k_off).^2)), digits=4)
    mae_k_off = round(mean(abs.(k_off_v .- estimated_k_off)), digits=4)
    
    text!(ax, 0.05, 0.95, "Correlation: r = $corr_k_off", space=:relative, fontsize=14, color=:black)
    text!(ax, 0.05, 0.88, "RMSE = $rmse_k_off", space=:relative, fontsize=14, color=:black)
    text!(ax, 0.05, 0.81, "MAE = $mae_k_off", space=:relative, fontsize=14, color=:black)
    text!(ax, 0.05, 0.74, "Slope = $(round(z[1], digits=3))", space=:relative, fontsize=14, color=:black)
    
    axislegend(ax, position=:rb)
    
    save("koff_comparison.png", fig)
    display(fig)
    println("📊 k_off comparison plot saved as 'koff_comparison.png'")
    return fig
end

"""
Plot both k_on and k_off comparisons side by side
"""
function plot_k_values_comparison()
    estimated_k_on = [result.k_on for result in optimize_val]
    estimated_k_off = [result.k_off for result in optimize_val]
    
    fig = Figure(size=(1400, 600))
    
    # k_on plot
    ax1 = Axis(fig[1, 1], 
               title="True vs Optimized k_on", 
               xlabel="True k_on", 
               ylabel="Optimized k_on",
               aspect=1)
    
    scatter!(ax1, k_on_v, estimated_k_on, color=:blue, markersize=10, alpha=0.7)
    
    # Perfect correlation line for k_on
    min_val1 = min(minimum(k_on_v), minimum(estimated_k_on))
    max_val1 = max(maximum(k_on_v), maximum(estimated_k_on))
    lines!(ax1, [min_val1, max_val1], [min_val1, max_val1], color=:black, linestyle=:dash, linewidth=2)
    
    # k_off plot
    ax2 = Axis(fig[1, 2], 
               title="True vs Optimized k_off", 
               xlabel="True k_off", 
               ylabel="Optimized k_off",
               aspect=1)
    
    scatter!(ax2, k_off_v, estimated_k_off, color=:red, markersize=10, alpha=0.7)
    
    # Perfect correlation line for k_off
    min_val2 = min(minimum(k_off_v), minimum(estimated_k_off))
    max_val2 = max(maximum(k_off_v), maximum(estimated_k_off))
    lines!(ax2, [min_val2, max_val2], [min_val2, max_val2], color=:black, linestyle=:dash, linewidth=2)
    
    # Add correlation coefficients
    corr_k_on = round(cor(k_on_v, estimated_k_on), digits=3)
    corr_k_off = round(cor(k_off_v, estimated_k_off), digits=3)
    
    text!(ax1, 0.05, 0.95, "r = $corr_k_on", space=:relative, fontsize=14)
    text!(ax2, 0.05, 0.95, "r = $corr_k_off", space=:relative, fontsize=14)
    
    save("k_values_comparison.png", fig)
    display(fig)
    println("📊 K values comparison plot saved as 'k_values_comparison.png'")
    return fig
end

"""
Plot true vs estimated parameters (legacy function - kept for compatibility)
"""
function plot_true_vs_estimated()
    return plot_k_values_comparison()
end

"""
Create comprehensive error dashboard with all plots
"""
function plot_error_dashboard()
    println("📊 Creating comprehensive error visualization dashboard...")
    
    try
        # Check if we have data
        if isempty(error_val) || isempty(optimize_val)
            println("❌ Error: No data available for plotting. Run the optimization loop first.")
            return nothing
        end
        
        println("📈 Creating absolute errors plot...")
        fig1 = plot_absolute_errors()
        
        println("📊 Creating relative errors plot...")
        fig2 = plot_relative_errors()
        
        println("🔊 Creating noise correlation plot...")
        fig3 = plot_noise_correlation()
        
        println("🔄 Creating convergence analysis plot...")
        fig4 = plot_convergence_analysis()
        
        println("🎯 Creating true vs estimated plot...")
        fig5 = plot_true_vs_estimated()
        
        println("✅ All error plots created and saved!")
        println("📁 Saved files:")
        println("   - absolute_errors_plot.png")
        println("   - relative_errors_plot.png")
        println("   - noise_correlation_plot.png")
        println("   - convergence_analysis_plot.png")
        println("   - true_vs_estimated_plot.png")
        
        return (fig1, fig2, fig3, fig4, fig5)
        
    catch e
        println("❌ Error creating plots: $e")
        println("💡 Try calling individual plot functions to identify the issue:")
        println("   - plot_absolute_errors()")
        println("   - plot_relative_errors()")
        println("   - plot_noise_correlation()")
        println("   - plot_convergence_analysis()")
        println("   - plot_true_vs_estimated()")
        return nothing
    end
end

"""
Simple polynomial fit for trend lines
"""
function polyfit(x, y, degree)
    # Simple linear fit (degree 1)
    if degree == 1
        n = length(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x .* y)
        sum_x2 = sum(x .^ 2)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        intercept = (sum_y - slope * sum_x) / n
        
        return [slope, intercept]
    end
end

tries = 30

k_on_v = rand(tries)
k_off_v = rand(tries)
k_off_initial_v =rand(tries) 
k_on_initial_v = rand(tries)
noise_v = rand(tries) .* 0.02
simulation_val = []
optimize_val = []
error_val = []

for i in 1:tries
    println("🔄 Running trial $i/$tries...")
    sim = run_simulation_smlms(k_off=k_off_v[i], k_on=k_on_v[i])
    sim_noisey = add_noise(sim, noise_level=noise_v[i])
    result = optimize_k_values(sim_noisey;initial_k_on=k_on_initial_v[i], initial_k_off=k_off_initial_v[i])
    erorr_k_off = abs(sim.k_states[2] - result.k_off)
    erorr_k_on = abs(sim.k_states[1] - result.k_on)
    push!(error_val,(erorr_k_on, erorr_k_off))
    push!(simulation_val,sim_noisey)
    push!(optimize_val,result)
    

end

# Call the error summary function after the loop completes
print_error_summary()
plot_error_dashboard()

#####  ########################################
"""errror with constant values of k_on and k_off"""
##############################################
tries = 100
k_on_v = fill(0.3, tries)
k_off_v = fill(0.5, tries)
k_off_initial_v =rand(tries) 
k_on_initial_v = rand(tries)
noise_v = fill(0.01, tries)
simulation_val = []
optimize_val = []
error_val = [] 
for i in 1:tries
    println("🔄 Running trial $i/$tries...")
    sim = run_simulation_smlms(k_off=k_off_v[i], k_on=k_on_v[i])
    sim_noisey = add_noise(sim, noise_level=noise_v[i])
    result = optimize_k_values(sim_noisey;initial_k_on=k_on_initial_v[i], initial_k_off=k_off_initial_v[i])
    erorr_k_off = abs(sim.k_states[2] - result.k_off)
    erorr_k_on = abs(sim.k_states[1] - result.k_on)
    push!(error_val,(erorr_k_on, erorr_k_off))
    push!(simulation_val,sim_noisey)
    push!(optimize_val,result)
end


fig= Figure()
ax1= Axis(fig[1,1], title="k_on values", xlabel="K_on_values perdict", ylabel="error")
lines!(ax1, k_off_v, color=:red, linestyle=:dash, label="true k_on")
scatter!(ax1, [result.k_on for result in optimize_val], color=:blue, markersize=8, label="optimized k_on")
axislegend(ax1, position=:rt)  
ax2= Axis(fig[1,2], title="k_off values", xlabel="K_off_values perdict", ylabel="error")
lines!(ax2, k_on_v, color=:red, linestyle=:dash, label="true k_off")
scatter!(ax2, [result.k_off for result in optimize_val], color=:blue, markersize=8, label="optimized k_off")
axislegend(ax2, position=:rt)  
display(fig)

# ========================================
# ERROR METRICS FOR CONSTANT VALUES EXPERIMENT
# ========================================

"""
Calculate error metrics for constant k values experiment
"""
function calculate_constant_error_metrics()
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    
    # Basic statistics for k_on
    rmse_k_on = sqrt(mean(k_on_errors.^2))
    mae_k_on = mean(k_on_errors)
    std_k_on = std(k_on_errors)
    min_k_on = minimum(k_on_errors)
    max_k_on = maximum(k_on_errors)
    
    # Basic statistics for k_off
    rmse_k_off = sqrt(mean(k_off_errors.^2))
    mae_k_off = mean(k_off_errors)
    std_k_off = std(k_off_errors)
    min_k_off = minimum(k_off_errors)
    max_k_off = maximum(k_off_errors)
    
    # Relative errors (since true values are constant)
    true_k_on = k_on_v[1]  # All values are the same
    true_k_off = k_off_v[1]  # All values are the same
    
    rel_errors_k_on = (k_on_errors ./ true_k_on) .* 100
    rel_errors_k_off = (k_off_errors ./ true_k_off) .* 100
    
    mean_rel_k_on = mean(rel_errors_k_on)
    mean_rel_k_off = mean(rel_errors_k_off)
    std_rel_k_on = std(rel_errors_k_on)
    std_rel_k_off = std(rel_errors_k_off)
    
    return (
        k_on = (rmse=rmse_k_on, mae=mae_k_on, std=std_k_on, min=min_k_on, max=max_k_on, 
                mean_rel=mean_rel_k_on, std_rel=std_rel_k_on),
        k_off = (rmse=rmse_k_off, mae=mae_k_off, std=std_k_off, min=min_k_off, max=max_k_off,
                 mean_rel=mean_rel_k_off, std_rel=std_rel_k_off)
    )
end

"""
Analyze convergence for constant values experiment
"""
function analyze_constant_convergence()
    converged_count = 0
    iterations = []
    loglikelihoods = []
    
    for result in optimize_val
        if Optim.converged(result.optimization_result)
            converged_count += 1
        end
        push!(iterations, Optim.iterations(result.optimization_result))
        push!(loglikelihoods, result.loglikelihood)
    end
    
    convergence_rate = (converged_count / length(optimize_val)) * 100
    
    return (
        rate = convergence_rate,
        converged = converged_count,
        total = length(optimize_val),
        avg_iterations = mean(iterations),
        std_iterations = std(iterations),
        avg_loglik = mean(loglikelihoods),
        std_loglik = std(loglikelihoods)
    )
end

"""
Print comprehensive error summary for constant values experiment
"""
function print_constant_error_summary()
    println("\n" * "="^60)
    println("CONSTANT VALUES EXPERIMENT ERROR ANALYSIS")
    println("="^60)
    
    metrics = calculate_constant_error_metrics()
    convergence = analyze_constant_convergence()
    
    # Print experiment parameters
    println("🔧 Experiment Parameters:")
    println("   Trials: $tries")
    println("   True k_on: $(k_on_v[1])")
    println("   True k_off: $(k_off_v[1])")
    println("   Noise level: $(noise_v[1])")
    
    # Print convergence analysis
    println("\n📊 Convergence Analysis:")
    println("   Convergence rate: $(convergence.converged)/$(convergence.total) ($(round(convergence.rate, digits=1))%)")
    println("   Avg iterations: $(round(convergence.avg_iterations, digits=1)) ± $(round(convergence.std_iterations, digits=1))")
    println("   Avg log-likelihood: $(round(convergence.avg_loglik, digits=2)) ± $(round(convergence.std_loglik, digits=2))")
    
    # Print k_on error metrics
    println("\n📏 k_on Error Metrics:")
    println("   RMSE: $(round(metrics.k_on.rmse, digits=4))")
    println("   MAE:  $(round(metrics.k_on.mae, digits=4))")
    println("   Std:  $(round(metrics.k_on.std, digits=4))")
    println("   Range: [$(round(metrics.k_on.min, digits=4)), $(round(metrics.k_on.max, digits=4))]")
    println("   Relative error: $(round(metrics.k_on.mean_rel, digits=1))% ± $(round(metrics.k_on.std_rel, digits=1))%")
    
    # Print k_off error metrics
    println("\n📏 k_off Error Metrics:")
    println("   RMSE: $(round(metrics.k_off.rmse, digits=4))")
    println("   MAE:  $(round(metrics.k_off.mae, digits=4))")
    println("   Std:  $(round(metrics.k_off.std, digits=4))")
    println("   Range: [$(round(metrics.k_off.min, digits=4)), $(round(metrics.k_off.max, digits=4))]")
    println("   Relative error: $(round(metrics.k_off.mean_rel, digits=1))% ± $(round(metrics.k_off.std_rel, digits=1))%")
    
    # Precision analysis
    println("\n🎯 Precision Analysis:")
    precision_k_on = 1 / metrics.k_on.std  # Higher is better
    precision_k_off = 1 / metrics.k_off.std
    println("   k_on precision: $(round(precision_k_on, digits=2))")
    println("   k_off precision: $(round(precision_k_off, digits=2))")
    
    # Best and worst cases
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    overall_errors = k_on_errors .+ k_off_errors
    
    best_idx = argmin(overall_errors)
    worst_idx = argmax(overall_errors)
    
    println("\n🏆 Best trial: #$best_idx")
    println("   k_on error: $(round(k_on_errors[best_idx], digits=4))")
    println("   k_off error: $(round(k_off_errors[best_idx], digits=4))")
    println("   Total error: $(round(overall_errors[best_idx], digits=4))")
    
    println("\n❌ Worst trial: #$worst_idx")
    println("   k_on error: $(round(k_on_errors[worst_idx], digits=4))")
    println("   k_off error: $(round(k_off_errors[worst_idx], digits=4))")
    println("   Total error: $(round(overall_errors[worst_idx], digits=4))")
    
    return metrics, convergence
end

"""
Plot enhanced error distribution for constant values
"""
function plot_constant_error_distribution()
    k_on_errors = [err[1] for err in error_val]
    k_off_errors = [err[2] for err in error_val]
    estimated_k_on = [result.k_on for result in optimize_val]
    estimated_k_off = [result.k_off for result in optimize_val]
    
    fig = Figure(size=(1400, 1000))
    
    # Plot 1: Error distributions
    ax1 = Axis(fig[1, 1], 
               title="Error Distribution", 
               xlabel="Absolute Error", 
               ylabel="Frequency")
    
    hist!(ax1, k_on_errors, bins=20, color=(:blue, 0.6), label="k_on errors")
    hist!(ax1, k_off_errors, bins=20, color=(:red, 0.6), label="k_off errors")
    axislegend(ax1, position=:rt)
    
    # Plot 2: Estimated values over trials
    ax2 = Axis(fig[1, 2], 
               title="Estimated Values by Trial", 
               xlabel="Trial Number", 
               ylabel="Estimated Value")
    
    scatter!(ax2, 1:tries, estimated_k_on, color=:blue, markersize=6, alpha=0.7, label="k_on estimates")
    scatter!(ax2, 1:tries, estimated_k_off, color=:red, markersize=6, alpha=0.7, label="k_off estimates")
    hlines!(ax2, [k_on_v[1]], color=:blue, linestyle=:dash, linewidth=2, label="True k_on")
    hlines!(ax2, [k_off_v[1]], color=:red, linestyle=:dash, linewidth=2, label="True k_off")
    axislegend(ax2, position=:rt)
    
    # Plot 3: Error by trial
    ax3 = Axis(fig[2, 1], 
               title="Errors by Trial", 
               xlabel="Trial Number", 
               ylabel="Absolute Error")
    
    scatter!(ax3, 1:tries, k_on_errors, color=:blue, markersize=6, alpha=0.7, label="k_on errors")
    scatter!(ax3, 1:tries, k_off_errors, color=:red, markersize=6, alpha=0.7, label="k_off errors")
    axislegend(ax3, position=:rt)
    
    # Plot 4: Box plot comparison
    ax4 = Axis(fig[2, 2], 
               title="Error Distribution Comparison", 
               xlabel="Parameter", 
               ylabel="Absolute Error")
    
    boxplot!(ax4, fill(1, length(k_on_errors)), k_on_errors, color=:blue, width=0.4)
    boxplot!(ax4, fill(2, length(k_off_errors)), k_off_errors, color=:red, width=0.4)
    ax4.xticks = ([1, 2], ["k_on", "k_off"])
    
    save("constant_values_error_analysis.png", fig)
    display(fig)
    println("📊 Constant values error analysis saved as 'constant_values_error_analysis.png'")
    return fig
end

# Calculate and display error metrics for constant values experiment
println("🔍 Analyzing constant values experiment...")
metrics, convergence = print_constant_error_summary()
plot_constant_error_distribution()
