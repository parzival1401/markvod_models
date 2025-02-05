using SMLMSim
using CairoMakie
state_history, args = SMLMSim.InteractionDiffusion.smoluchowski(density=0.02,t_max=25,box_size=10,k_off=0.3,r_react=2)
dimer_history = SMLMSim.get_dimers(state_history)
SMLMSim.gen_movie(state_history,args; filename="defaultsim.mp4")

x_position_particle1 = zeros(size(state_history.frames,1))
y_position_particle1 =zeros(size(state_history.frames,1))
state_particle1 =zeros(size(state_history.frames,1))
x_position_particle2 = zeros(size(state_history.frames,1))
y_position_particle2=zeros(size(state_history.frames,1))
state_particle2=zeros(size(state_history.frames,1))


for i in eachindex(state_history.frames)
    x_position_particle1[i] = state_history.frames[i].molecules[1].x
    y_position_particle1[i] = state_history.frames[i].molecules[1].y
    x_position_particle2[i] = state_history.frames[i].molecules[2].x
    y_position_particle2[i] = state_history.frames[i].molecules[2].y
    state_particle2[i] = state_history.frames[i].molecules[2].state
    state_particle1[i] = state_history.frames[i].molecules[1].state
end


dnx = x_position_particle2 - x_position_particle1

dny = y_position_particle2 - y_position_particle1








fig = Figure(size=(1000, 500))

ax1 = Axis(fig[1, 1],
    xlabel = "Time Frame",
    ylabel = "Distance X (dnx)",
    title = "X Distance Difference Over Time")


lines!(ax1, 1:length(dnx), dnx, 
    color = :blue, 
    linewidth = 2)
scatter!(ax1, 1:length(dnx), dnx, 
    color = :blue, 
    markersize = 4)


ax2 = Axis(fig[1, 2],
    xlabel = "Time Frame",
    ylabel = "Distance Y (dny)",
    title = "Y Distance Difference Over Time")


lines!(ax2, 1:length(dny), dny, 
    color = :red, 
    linewidth = 2)
scatter!(ax2, 1:length(dny), dny, 
    color = :red, 
    markersize = 4)








function bessel_function(dt,dn_square,dn_1square,sigma)
    bessel =0


        for i in 0:dt:2*pi
            bessel += (1/(2*pi))*exp((sqrt(dn_square*dn_1square)*cos(i))/sigma^2)
        end
    return bessel

end 

function    density(x_position::Vector ,y_position::Vector,sigma,dt=.01)
    dn_square = zeros(size(x_position))
    density = zeros(size(x_position))

    for i in 1:length(x_position)
        if i <length(x_position)
            dn_square[i]= (x_position[i+1]-x_position[i])^2 + (y_position[i+1]-y_position[i])^2
        else 
            break
        end 
    end 

    for i in 1:length(dn_square)
        if i <length(dn_square)
    
        density[i] = (sqrt(dn_square[i])/(sigma)^2)*(exp((-dn_square[i+1]-dn_square[i])/(sigma)^2))*(bessel_function(dt,dn_square[i+1],dn_square[i],sigma))
        else 
            break
        end
    end  

    return density, dn_square
end 
function density(dn_square::Float64,dn_1square::Float64,sigma)
    
    density= (sqrt(dn_square)/(sigma)^2)*(exp((-dn_1square-dn_square)/(sigma)^2))*(bessel_function(0.01,dn_1square,dn_square,sigma))
    return density
end

density_var1, dn_square1 = density(dnx, dny, 0.1)
density_var2, dn_square2 = density(dnx, dny, 1)
density_var3, dn_square3 = density(dnx, dny, 5)
density_var4, dn_square4 = density(dnx, dny, 10)
density_var5, dn_square5 = density(dnx, dny, 0.5)

fig = Figure(size=(1000, 500))

ax1 = Axis(fig[1, 1],
    xlabel = "dn",
    ylabel = "g(x)",
    title = "X Distance Difference Over Time")

scatter!(ax1, dn_square1, density_var1, color = :blue, linewidth = 2, label = "δ = 0.1")
scatter!(ax1, dn_square2, density_var2, color = :red, linewidth = 2, label = "δ = 1.0")
scatter!(ax1, dn_square3, density_var3, color = :green, linewidth = 2, label = "δ = 5.0")
scatter!(ax1, dn_square4, density_var4, color = :purple, linewidth = 2, label = "δ = 10.0")
scatter!(ax1, dn_square5, density_var5, color = :orange, linewidth = 2, label = "δ = 0.5")

axislegend(ax1, position = :rt) 
fig 





sigma_range = 0.1:0.01:1  



density_values = [density(dn_square1[2] , dn_square1[1], σ) for σ in sigma_range]
density_values1 = [density(dn_square1[3] , dn_square1[2], σ) for σ in sigma_range]


fig = Figure(size=(800, 600))
ax1 = Axis(fig[1, 1],
    xlabel = "σ",
    ylabel = "Density",
    title = "Density as a Function of σ")
ax2 = Axis(fig[1, 2],
    xlabel = "σ",
    ylabel = "Density",
    title = "Density as a Function of σ")

lines!(ax1, sigma_range, density_values, 
    color = :blue, 
    linewidth = 2,
    label = "d[2] and d[1]")
lines!(ax2, sigma_range, density_values1, 
    color = :red, 
    linewidth = 2,
    label = "d[3] and d[2]")

axislegend(position = :rt)


fig