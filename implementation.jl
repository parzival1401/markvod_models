using Random
using CairoMakie
using Distributions
using LinearAlgebra
using BenchmarkTools

mutable struct Molecule
    x::Float64
    y::Float64
    state::Int64
end

mutable struct Frame
    molecules::Vector{Molecule}
end

mutable struct StateHistory
    frames::Vector{Frame}
end

function smoluchowski(; density=0.02, t_max=35, box_size=10, k_off=0.2, r_react=1)
    state_history = StateHistory(Vector{Frame}())
    args = Dict(
        "density" => density,
        "t_max" => t_max,
        "box_size" => box_size,
        "k_off" => k_off,
        "r_react" => r_react
    )
    return state_history, args
end

function get_dimers(state_history::StateHistory)
    dimer_history = []
    for frame in state_history.frames
        dimers = []
        for mol in frame.molecules
            if mol.state == 2
                push!(dimers, mol)
            end
        end
        push!(dimer_history, dimers)
    end
    return dimer_history
end

function extract_trajectories(state_history::StateHistory)
    n_frames = length(state_history.frames)
    n_molecules = length(state_history.frames[1].molecules)
    
    x_positions = zeros(n_frames, n_molecules)
    y_positions = zeros(n_frames, n_molecules)
    states = zeros(Int, n_frames, n_molecules)
    
    for (i, frame) in enumerate(state_history.frames)
        for (j, mol) in enumerate(frame.molecules)
            x_positions[i,j] = mol.x
            y_positions[i,j] = mol.y
            states[i,j] = mol.state
        end
    end
    
    return x_positions, y_positions, states
end

function simulate_system()
    state_history, args = smoluchowski(density=0.02, t_max=35, box_size=10, k_off=0.2, r_react=1)
    
    # Initialize with two particles
    molecule1 = Molecule(5.0, 5.0, 1)  # First particle
    molecule2 = Molecule(6.0, 6.0, 1)  # Second particle
    
    # Create initial frame
    initial_frame = Frame([molecule1, molecule2])
    push!(state_history.frames, initial_frame)
    
    # Extract trajectories
    x_position_particle1 = zeros(length(state_history.frames))
    y_position_particle1 = zeros(length(state_history.frames))
    state_particle1 = zeros(Int, length(state_history.frames))
    x_position_particle2 = zeros(length(state_history.frames))
    y_position_particle2 = zeros(length(state_history.frames))
    state_particle2 = zeros(Int, length(state_history.frames))
    
    for i in eachindex(state_history.frames)
        x_position_particle1[i] = state_history.frames[i].molecules[1].x
        y_position_particle1[i] = state_history.frames[i].molecules[1].y
        x_position_particle2[i] = state_history.frames[i].molecules[2].x
        y_position_particle2[i] = state_history.frames[i].molecules[2].y
        state_particle2[i] = state_history.frames[i].molecules[2].state
        state_particle1[i] = state_history.frames[i].molecules[1].state
    end
    
    return x_position_particle1, y_position_particle1, state_particle1,
           x_position_particle2, y_position_particle2, state_particle2,
           state_history
end

function plot_trajectories(x1, y1, x2, y2, states1, states2)
    fig = Figure(resolution=(800, 400))
    
    ax1 = Axis(fig[1, 1], 
        xlabel="X Position",
        ylabel="Y Position",
        title="Particle Trajectories")
    
    # Plot trajectories with color based on state
    for i in 1:length(x1)-1
        color1 = states1[i] == 1 ? :blue : :red
        color2 = states2[i] == 1 ? :green : :orange
        
        lines!(ax1, [x1[i], x1[i+1]], [y1[i], y1[i+1]], color=color1)
        lines!(ax1, [x2[i], x2[i+1]], [y2[i], y2[i+1]], color=color2)
    end
    
    return fig
end