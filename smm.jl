using SMLMSim
state_history, args = SMLMSim.InteractionDiffusion.smoluchowski(density=0.02,t_max=35,box_size=10,k_off=0.2,r_react=1)
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
