using SMLMSim
using CairoMakie
state_history, args = SMLMSim.InteractionDiffusion.smoluchowski(density=0.02,t_max=50,box_size=10,k_off=0.3,r_react=2)
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



fig





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

    return density
end 



