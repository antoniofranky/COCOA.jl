using COBREXA, SBMLFBCModels, AbstractFBCModels
using COCOA, HiGHS

# Specific models you want to process
target_models = [
    "Babjeviella_inositovora.xml",
    "Botrytis_cinerea.xml", 
    "Brettanomyces_anomalus.xml",
    "yHMPu5000034950_Citeromyces_hawaiiensis.xml"
]

# Base directory containing the models
model_dir = "/work/schaffran1/Yeast-Species-GEMs"
output_dir = "/work/schaffran1/toolbox/prpd_models"

println("Processing $(length(target_models)) specific models...")

for model_name in target_models
    model_path = joinpath(model_dir, model_name)
    
    if !isfile(model_path)
        println("❌ Model not found: $model_path")
        continue
    end
    
    println("🔄 Processing: $model_name")
    
    try
        for rdm in collect(0.1:0.1:0.9)
            println("  Random fraction: $rdm")
            
            # Load and process model
            model = load_model(model_path)
            model_splt = split_into_elementary_steps(model, seed=42, normalize_bounds=true,
                random_fraction=rdm)
            model_splt_prpd = prepare_model_for_concordance(model_splt,
                optimizer=HiGHS.Optimizer, output_type=SBMLFBCModels.SBMLFBCModel)

            # Create output directory and save
            dir_name = "random_$(Int(rdm * 100))"
            output_path = joinpath(output_dir, dir_name)
            mkpath(output_path)  # Ensure directory exists
            
            model_basename = splitext(model_name)[1]
            save_path = joinpath(output_path, "$(model_basename).xml")
            
            save_model(model_splt_prpd, save_path)
            println("    ✅ Saved: $save_path")
        end
        
        println("✅ Successfully processed: $model_name")
        
    catch e
        println("❌ Error processing $model_name: $e")
        # Continue with next model instead of exiting
    end
end

println("🎉 Custom model preparation complete!")