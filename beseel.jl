using SpecialFunctions
using CairoMakie



function modified_bessel(x,dt)

    result_me=0

    for θ in 0:dt:2*pi
        result_me += exp(x * cos(θ))
    end

    result_me  *= dt / (2*pi)


    return result_me
end

x = 100:.01:110

y_1 = besseli.(0,x)
y_1/=maximum(y_1)

y_2=modified_bessel.(x,.001)
y_2/=maximum(y_2)







fig=Figure()
ax=Axis(fig[1,1],title="besely")
ax_1=Axis(fig[2,1],title="custome ")

lines!(ax,x,y_1, label="special functions")
lines!(ax_1,x,y_2,label="custome")
axislegend(ax)
fig