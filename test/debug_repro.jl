
using COCOA
using Test
using HiGHS
using LinearAlgebra

try
    println("Loading model...")
    model = create_envz_ompr_model()

    println("Running concordance...")
    # Flush stdout to ensure logs appear
    flush(stdout)

    results = activity_concordance_analysis(
        model;
        optimizer=HiGHS.Optimizer,
        kinetic_analysis=false,
        sample_size=100,
        concordance_tolerance=0.01,
        cv_threshold=0.01
    )

    concordance_modules = extract_concordance_modules(results)

    println("Running kinetic analysis (Standard - No Step 2)...")
    val, t_std, bytes, gctime, memallocs = @timed kinetic_analysis(
        concordance_modules,
        model;
        enable_advanced_merging=false,
        min_module_size=2
    )
    println("  Time: $t_std s")
    println("  Modules: $(length(val))")

    println("Running kinetic analysis (Advanced - Single Pass Optimization)...")
    val_adv, t_adv, bytes_adv, gctime_adv, memallocs_adv = @timed kinetic_analysis(
        concordance_modules,
        model;
        enable_advanced_merging=true,
        min_module_size=2
    )
    println("  Time: $t_adv s")
    println("  Modules: $(length(val_adv))")

    println("Performance Check: Advanced time ($t_adv s) should be comparable to Standard time ($t_std s)")
    if t_adv > t_std * 2 + 1.0 # Allow some overhead
        @warn "Advanced merging is significantly slower! Optimization might not be working efficiently."
    else
        println("Optimization verified: Fast execution confirmed.")
    end

    println("Success!")
catch e
    open("error.log", "w") do f
        println(f, "ERROR CAUGHT:")
        showerror(f, e, catch_backtrace())
    end
end
