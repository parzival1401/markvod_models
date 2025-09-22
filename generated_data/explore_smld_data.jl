# Exploring SMLD (Single Molecule Localization Data) structure
# Understanding what SMLMSim actually produces

using SMLMSim

println("🔍 Exploring SMLD Data Structure")
println("=" ^ 40)

function explore_smld_data()
    println("🚀 Generating and exploring SMLD data...")
    
    # Generate localization data using uniform2D
    ρ = 0.5  # density of molecules
    pattern = SMLMSim.Point2D()  # single molecules
    xsize = 5.0  # μm
    ysize = 5.0  # μm
    
    smld = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
    
    println("📋 SMLD Data Structure:")
    println("   Type: $(typeof(smld))")
    println("   Fields: $(fieldnames(typeof(smld)))")
    
    # Explore each field
    println("\n📊 Data Content:")
    
    # Basic localization data
    println("   Number of localizations: $(length(smld.x))")
    println("   X positions: $(smld.x[1:min(5, length(smld.x))])")
    println("   Y positions: $(smld.y[1:min(5, length(smld.y))])")
    
    # Precision information
    println("   X precision (σ_x): $(smld.σ_x[1:min(5, length(smld.σ_x))])")
    println("   Y precision (σ_y): $(smld.σ_y[1:min(5, length(smld.σ_y))])")
    
    # Photon information
    println("   Photons per localization: $(smld.photons[1:min(5, length(smld.photons))])")
    println("   Photon precision: $(smld.σ_photons[1:min(5, length(smld.σ_photons))])")
    
    # Background
    println("   Background: $(smld.bg[1:min(5, length(smld.bg))])")
    
    # Frame information
    println("   Frame numbers: $(smld.framenum[1:min(5, length(smld.framenum))])")
    println("   Dataset numbers: $(smld.datasetnum[1:min(5, length(smld.datasetnum))])")
    
    # Metadata
    println("   Total frames: $(smld.nframes)")
    println("   Total datasets: $(smld.ndatasets)")
    println("   Data size: $(smld.datasize)")
    
    return smld
end

function explore_nmer_data()
    println("\n🔗 Exploring Nmer (cluster) data...")
    
    # Generate dimer data
    ρ = 0.2
    pattern = SMLMSim.Nmer2D(n=2, d=0.05)  # dimers 50nm apart
    xsize = 3.0
    ysize = 3.0
    
    smld_dimers = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
    
    println("📋 Dimer SMLD Data:")
    println("   Number of localizations: $(length(smld_dimers.x))")
    println("   X positions: $(smld_dimers.x)")
    println("   Y positions: $(smld_dimers.y)")
    
    # Look for clustered positions (pairs)
    positions = [(smld_dimers.x[i], smld_dimers.y[i]) for i in 1:length(smld_dimers.x)]
    println("   Position pairs: $(positions)")
    
    return smld_dimers
end

function compare_with_our_simulation()
    println("\n🔄 Comparing with our particle simulation:")
    
    println("📊 SMLMSim (Microscopy simulation):")
    println("   • Produces: x, y coordinates with precision")
    println("   • Focus: Fluorescence microscopy")
    println("   • Static positions with measurement uncertainty")
    println("   • Frame-based data structure")
    
    println("\n📊 Our simulation (Particle dynamics):")
    println("   • Produces: time-dependent trajectories")
    println("   • Focus: Markov state transitions")
    println("   • Dynamic positions with physical movement")
    println("   • Time-series data structure")
    
    println("\n🎯 Key differences:")
    println("   • SMLMSim: STATIC positions + measurement noise")
    println("   • Our sim: DYNAMIC trajectories + state transitions")
    println("   • SMLMSim: σ_x, σ_y = localization precision")
    println("   • Our sim: σ = physical noise in movement")
end

# Run explorations
smld = explore_smld_data()
smld_dimers = explore_nmer_data()
compare_with_our_simulation()

println("\n💡 Final understanding:")
println("   SMLMSim generates LOCALIZATION coordinates for microscopy")
println("   It does NOT simulate particle trajectories or dynamics")
println("   Perfect for: Analyzing super-resolution microscopy data")
println("   NOT suitable for: Markov chain dynamics or binding kinetics")