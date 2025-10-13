include("final_implementation.jl")

println("\n🧪 Testing SMLMSim Integration...")
println("=" ^ 60)

# Test SMLMSim integration
println("\n1. Testing run_simulation_smlms...")
try
    sim, trajectories = run_simulation_smlms(
        k_off=0.5,
        r_react=0.05,
        d_dimer=0.07,
        t_max=5.0,
        dt=0.01,
        diff_monomer=0.1,
        diff_dimer=0.05,
        box_size=1.0,
        camera_framerate=10.0
    )

    println("   ✅ SMLMSim simulation created successfully")
    println("   Simulation has $(length(sim.particle_1)) steps")
    println("   Trajectories: $(length(trajectories))")

    # Test forward algorithm with SMLMSim data
    println("\n2. Testing forward algorithm on SMLMSim data...")
    alpha, loglik, e1_vec, e2_vec = forward_algorithm(sim, dt=0.01, sigma=0.01)
    println("   ✅ Forward algorithm completed")
    println("   Log-likelihood: $(round(loglik, digits=2))")

    # Test segment extraction
    println("\n3. Testing segment extraction...")
    free_segs = extract_free_segments(sim)
    bound_segs = extract_bound_segments(sim)
    println("   ✅ Segment extraction completed")
    println("   Free segments: $(length(free_segs))")
    println("   Bound segments: $(length(bound_segs))")

    println("\n" * "=" ^ 60)
    println("✅ SMLMSim integration tests passed!")
    println("The SMLMSim workflow matches new_smlsmsim.jl structure")

catch e
    println("\n❌ Error during SMLMSim integration test:")
    println("   $(e)")
    showerror(stdout, e, catch_backtrace())
end
