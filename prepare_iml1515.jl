using COCOA, HiGHS, SBMLFBCModels, AbstractFBCModels, COBREXA
@everywhere using COCOA, HiGHS
# Paths
input_model_path = "/work/schaffran1/COCOA.jl/benchmark/iML1515.xml"
output_model_path = "/work/schaffran1/COCOA.jl/benchmark/iML28654_splt_prpd.xml"

@info "Starting preparation of iML1515 model"
@info "Input:" input_model_path
@info "Output:" output_model_path

# Load the original model
@info "Loading original model..."
model = COBREXA.load_model(input_model_path)

# Get original model statistics
n_reactions_orig = length(AbstractFBCModels.reactions(model))
n_metabolites_orig = length(AbstractFBCModels.metabolites(model))
n_genes_orig = length(AbstractFBCModels.genes(model))

@info "Original model statistics:" n_reactions = n_reactions_orig n_metabolites = n_metabolites_orig n_genes = n_genes_orig

# Split into elementary steps with ordered_fraction=0.1
@info "Splitting model into elementary steps with ordered_fraction=0.1..."
split_model = COCOA.split_into_elementary_steps(model, ordered_fraction=0.1)

# Get split model statistics
n_reactions_split = length(AbstractFBCModels.reactions(split_model))
n_metabolites_split = length(AbstractFBCModels.metabolites(split_model))
n_genes_split = length(AbstractFBCModels.genes(split_model))

@info "Split model statistics:" n_reactions = n_reactions_split n_metabolites = n_metabolites_split n_genes = n_genes_split

# Prepare model for concordance analysis
@info "Preparing model for concordance analysis..."
prepared_model = COCOA.prepare_model_for_concordance(split_model, optimizer=HiGHS.Optimizer)

# Get final model statistics
n_reactions_final = length(AbstractFBCModels.reactions(prepared_model))
n_metabolites_final = length(AbstractFBCModels.metabolites(prepared_model))
n_genes_final = length(AbstractFBCModels.genes(prepared_model))

@info "Final prepared model statistics:" n_reactions = n_reactions_final n_metabolites = n_metabolites_final n_genes = n_genes_final

# Save the prepared model
@info "Saving prepared model to:" output_model_path
COBREXA.save_model(prepared_model, output_model_path)

@info "Model preparation complete!"
@info "Summary of changes:"
println("  Original -> Split -> Prepared")
println("  Reactions: $n_reactions_orig -> $n_reactions_split -> $n_reactions_final")
println("  Metabolites: $n_metabolites_orig -> $n_metabolites_split -> $n_metabolites_final")
println("  Genes: $n_genes_orig -> $n_genes_split -> $n_genes_final")

@info "Model saved and ready for benchmarking!"
