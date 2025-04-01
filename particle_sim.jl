using GLMakie

# position and velocity
p1 = (rand(2), [0.3, 0.4])  
p2 = (rand(2), [-0.2, 0.3])
box = 1.0


fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect())  
limits!(ax, 0, box, 0, box)


s1 = scatter!(ax, [p1[1][1]], [p1[1][2]], color=:blue, markersize=20)
s2 = scatter!(ax, [p2[1][1]], [p2[1][2]], color=:red, markersize=20)


display(fig)
p1_pos=zeros(2,2000)
p2_pos=zeros(2,2000)
for t in 1:2000
    
    p1[1] .+= p1[2] .* 0.016
    p2[1] .+= p2[2] .* 0.016
    
    
    for i in 1:2
        if p1[1][i] <= 0 || p1[1][i] >= box; p1[2][i] *= -1; end
        if p2[1][i] <= 0 || p2[1][i] >= box; p2[2][i] *= -1; end
    end
    
    
    s1[1]= [p1[1][1]]
    s1[2]=[p1[1][2]]

    p1_pos[1,t] = p1[1][1]
    p1_pos[2,t] = p1[1][2]

    s2[1] = [p2[1][1]]
    s2[2] = [p2[1][2]]

    p2_pos[1,t] = p2[1][1]
    p2_pos[2,t] = p2[1][2]

    sleep(0.016)
end


############################
############################
############################
############################
###############
using GLMakie


D = 0.01
p1 = (rand(2), randn(2)*sqrt(2D/0.016))
p2 = (rand(2), randn(2)*sqrt(2D/0.016))
box = 1.0

p1_pos = zeros(2, 2000)
p2_pos = zeros(2, 2000)


fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect())
limits!(ax, 0, box, 0, box)
s1 = scatter!(ax, [p1[1][1]], [p1[1][2]], color=:blue, markersize=20)
s2 = scatter!(ax, [p2[1][1]], [p2[1][2]], color=:red, markersize=20)


display(fig)
for t in 1:2000
    p1[2] .= randn(2) .* sqrt(2D/0.016)
    p2[2] .= randn(2) .* sqrt(2D/0.016)
    p1[1] .+= p1[2] .* 0.016
    p2[1] .+= p2[2] .* 0.016
    
    for i in 1:2
        if p1[1][i] <= 0 || p1[1][i] >= box; p1[1][i] = clamp(p1[1][i], 0, box); end
        if p2[1][i] <= 0 || p2[1][i] >= box; p2[1][i] = clamp(p2[1][i], 0, box); end
    end
    
    s1[1] = [p1[1][1]]; s1[2] = [p1[1][2]]
    s2[1] = [p2[1][1]]; s2[2] = [p2[1][2]]
    
    
    p1_pos[:,t] = p1[1]
    p2_pos[:,t] = p2[1]
    
    sleep(0.016)
end
frames = Observable(1)
record(fig, "diffusion.gif", 1:500; framerate = 30) do frame
    p1[2] .= randn(2) .* sqrt(2D/0.016)
    p2[2] .= randn(2) .* sqrt(2D/0.016)
    p1[1] .+= p1[2] .* 0.016
    p2[1] .+= p2[2] .* 0.016
    
    for i in 1:2
        if p1[1][i] <= 0 || p1[1][i] >= box; p1[1][i] = clamp(p1[1][i], 0, box); end
        if p2[1][i] <= 0 || p2[1][i] >= box; p2[1][i] = clamp(p2[1][i], 0, box); end
    end
    
    s1[1] = [p1[1][1]]; s1[2] = [p1[1][2]]
    s2[1] = [p2[1][1]]; s2[2] = [p2[1][2]]
    
    frames[] = frame
end