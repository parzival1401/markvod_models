# Learning SMLMSim API - Step by Step
# This file is purely for understanding how SMLMSim works

using SMLMSim

println("🎓 Learning SMLMSim API")
println("=" ^ 40)

# Step 1: Explore what's available
println("📋 Step 1: Available components in SMLMSim")
available_components = names(SMLMSim, all=false)
for component in available_components
    if !startswith(string(component), "#") && component != :SMLMSim
        println("   • $component")
    end
end

println("\n" ^ 2)

# Step 2: Learn about the main simulation function
println("📋 Step 2: Understanding the sim() function")
println("Method signature:")
println(methods(SMLMSim.sim))

println("\n" ^ 2)

# Step 3: Learn about patterns (spatial arrangements)
println("📋 Step 3: Understanding Patterns")

println("\n🔹 Point2D pattern:")
try
    point_pattern = SMLMSim.Point2D()
    println("   ✅ Created: $(typeof(point_pattern))")
    println("   Fields: $(fieldnames(typeof(point_pattern)))")
catch e
    println("   ❌ Error: $e")
end

println("\n🔹 Nmer2D pattern (clusters):")
try
    println("   Constructor signature: $(methods(SMLMSim.Nmer2D))")
    # Create a dimer (2 molecules, 0.05 distance apart)
    nmer_pattern = SMLMSim.Nmer2D(n=2, d=0.05)
    println("   ✅ Created: $(typeof(nmer_pattern))")
    println("   Fields: $(fieldnames(typeof(nmer_pattern)))")
catch e
    println("   ❌ Error: $e")
end

println("\n" ^ 2)

# Step 4: Learn about molecules (fluorophores)
println("📋 Step 4: Understanding Fluorophores")

println("\n🔹 GenericFluor:")
try
    println("   Constructor signature: $(methods(SMLMSim.GenericFluor))")
    
    # Create a simple fluorophore
    γ = 1000.0  # photon emission rate (photons/s)
    q = [0.1, 0.9, 0.1]  # transition rate matrix for blinking
    
    fluorophore = SMLMSim.GenericFluor(γ=γ, q=q)
    println("   ✅ Created: $(typeof(fluorophore))")
    println("   Fields: $(fieldnames(typeof(fluorophore)))")
    
    # Try to access properties
    println("   Photon rate (γ): $(fluorophore.γ)")
    println("   Transition matrix shape: $(size(fluorophore.q))")
    
catch e
    println("   ❌ Error: $e")
end

println("\n" ^ 2)

# Step 5: Learn about cameras
println("📋 Step 5: Understanding Cameras")

println("\n🔹 IdealCamera:")
try
    println("   Constructor signature: $(methods(SMLMSim.IdealCamera))")
    
    camera = SMLMSim.IdealCamera(
        pixelsize=100.0,  # nm per pixel
        xpixels=64,       # image width in pixels
        ypixels=64,       # image height in pixels
        gain=1.0,         # camera gain
        offset=0.0        # camera offset
    )
    println("   ✅ Created: $(typeof(camera))")
    println("   Fields: $(fieldnames(typeof(camera)))")
    
catch e
    println("   ❌ Error: $e")
end

println("\n" ^ 2)

# Step 6: Learn about other useful functions
println("📋 Step 6: Other SMLMSim functions")

println("\n🔹 uniform2D (spatial distribution):")
try
    println("   Signature: $(methods(SMLMSim.uniform2D))")
catch e
    println("   ❌ Error: $e")
end

println("\n🔹 CTMC (Continuous Time Markov Chain):")
try
    println("   Signatures: $(methods(SMLMSim.CTMC))")
catch e
    println("   ❌ Error: $e")
end

println("\n🔹 intensitytrace:")
try
    println("   Signature: $(methods(SMLMSim.intensitytrace))")
catch e
    println("   ❌ Error: $e")
end

println("\n" ^ 2)

# Step 7: Try a minimal simulation (commented out to avoid errors)
println("📋 Step 7: Minimal simulation structure")
println("The sim() function would be called like this:")
println("""
SMLMSim.sim(
    ρ=0.1,           # density of molecules (molecules/μm²)
    σ_PSF=1.5,       # point spread function width (pixels)
    minphotons=100,  # minimum photons per localization
    ndatasets=1,     # number of datasets to generate
    nframes=100,     # number of frames
    framerate=100.0, # frames per second
    pattern=pattern, # spatial pattern (Point2D, Nmer2D, etc.)
    molecule=fluor,  # fluorophore model
    camera=camera    # camera model
)
""")

println("\n📝 Key insights:")
println("   • SMLMSim simulates MICROSCOPY experiments")
println("   • It generates IMAGES with fluorescent spots")
println("   • The output is localization data, not particle trajectories")
println("   • Designed for super-resolution microscopy analysis")

println("\n🎯 Summary:")
println("   SMLMSim = Single-Molecule Localization Microscopy Simulator")
println("   Purpose: Simulate camera images of fluorescent molecules")
println("   Output: Localization coordinates with precision/uncertainty")
println("   NOT for: Particle interaction dynamics or Markov state modeling")