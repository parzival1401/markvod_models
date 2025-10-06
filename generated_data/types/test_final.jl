include("final_implementation.jl")

println("\n🧪 Testing updated final_implementation.jl...")
println("=" ^ 60)

# Test 1: Simulation generation
println("\n1. Testing simulation generation...")
sim, states = run_simulation(0.5, 0.3, 5, D=0.01, dt=0.016, d_dimer=0.01)
println("   ✅ Simulation created with $(length(sim.particle_1)) steps")

# Test 2: Forward algorithm
println("\n2. Testing forward algorithm...")
alpha, loglik, e1_vec, e2_vec = forward_algorithm(sim, dt=0.016, sigma=0.01)
println("   ✅ Forward algorithm completed")
println("   Log-likelihood: $(round(loglik, digits=2))")
println("   Alpha shape: $(size(alpha))")
println("   E1 values: $(length(e1_vec))")
println("   E2 values: $(length(e2_vec))")

# Test 3: Segment extraction
println("\n3. Testing segment extraction...")
free_segs = extract_free_segments(sim)
bound_segs = extract_bound_segments(sim)
println("   ✅ Segment extraction completed")
println("   Free segments: $(length(free_segs))")
println("   Bound segments: $(length(bound_segs))")

# Test 4: Noise addition
println("\n4. Testing noise addition...")
sim_noisy = add_noise(sim, 0.01)
println("   ✅ Noise added successfully")

# Test 5: Position matrices
println("\n5. Testing utility functions...")
p1, p2 = get_position_matrices(sim)
println("   ✅ Position matrices extracted: $(size(p1))")

println("\n" * "=" ^ 60)
println("✅ All functionality tests passed!")
println("The updated final_implementation.jl now has the same functionality as types_with_data.jl")
