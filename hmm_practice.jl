using Random
using CairoMakie


k12= 0.1
k21= 0.1
Δt= .1
T= [1-(k12*Δt) k12*Δt; 
    k21*Δt 1-(k21*Δt)]

pi0= [1;
      0]

sz = 1000
states = zeros(Int, sz)
current_state = 0

for i in 1:sz
    r = rand()
    if current_state == 0
        if r < T[1,1]
            current_state = 0
        else
            current_state = 1
        end
    else  
        if r < T[2,1]
            current_state = 0
        else
            current_state = 1
        end
    end
    
    states[i] = current_state
end
fig = Figure()
ax = Axis(fig[1, 1])


lines!(ax, 1:sz, states, color = :blue)


display(fig)


