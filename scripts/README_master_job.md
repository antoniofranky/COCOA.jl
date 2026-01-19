# Master COCOA Analysis Job

## Overview

The `analyse_models_master.sh` script runs COCOA analysis across multiple combinations:

- **13 models**: Lipomyces_starkeyi, Tortispora_caseinolytica, Yarrowia_deformans, Alloascoidea_hylecoeti, Sporopachydermia_quercuum, Pachysolen_tannophilus, Komagataella_pastoris, Debaryomyces_hansenii, Saccharomycopsis_malanga, Wickerhamomyces_ciferrii, Hanseniaspora_vinae, Torulaspora_delbrueckii, Neurospora_crassa
- **4 variants**: random_25, random_50, random_75, random_100
- **9 seeds**: 44, 45, 46, 47, 48, 49, 50, 51, 52

**Total tasks**: 13 × 4 × 9 = **468 array tasks**

## Submission

```bash
cd /work/schaffran1/COCOA.jl/scripts
sbatch analyse_models_master.sh
```

The script limits concurrent tasks to 50 (`--array=1-468%50`) to avoid overloading the cluster.

## Task Mapping

Each array task ID (1-468) maps to a unique combination:

- **Seed varies slowest** (outer loop): blocks of 52 tasks per seed
- **Variant varies medium** (middle loop): blocks of 13 tasks per variant
- **Model varies fastest** (inner loop): individual tasks

### Example mapping:
- Task 1: seed=44, variant=random_25, model=Lipomyces_starkeyi
- Task 13: seed=44, variant=random_25, model=Neurospora_crassa
- Task 14: seed=44, variant=random_50, model=Lipomyces_starkeyi
- Task 52: seed=44, variant=random_100, model=Neurospora_crassa
- Task 53: seed=45, variant=random_25, model=Lipomyces_starkeyi
- Task 468: seed=52, variant=random_100, model=Neurospora_crassa

## Output Locations

- **Logs**: `/work/schaffran1/jobresults/master_logs/cocoa_<JobID>_<TaskID>.out`
- **Results**: `/work/schaffran1/jobresults/<seed>/<variant>/kinetic_results_<model>_<seed>_100000_cv0p01_samples1000_transitivitytrue_tol_10.jld2`
- **Job mapping**: `/work/schaffran1/jobresults/<seed>/<variant>/job_mapping_master_<JobID>.txt`

## Monitoring

Check job status:
```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8T %.10M %.6D %R"
```

Check specific variant results:
```bash
ls -lh /work/schaffran1/jobresults/44/random_25/*.jld2 | wc -l
ls -lh /work/schaffran1/jobresults/44/random_50/*.jld2 | wc -l
ls -lh /work/schaffran1/jobresults/44/random_75/*.jld2 | wc -l
ls -lh /work/schaffran1/jobresults/44/random_100/*.jld2 | wc -l
```

View job mapping:
```bash
cat /work/schaffran1/jobresults/44/random_25/job_mapping_master_*.txt
```

## Resource Allocation

- **CPUs**: 64 cores per task
- **Memory**: 600GB per task (sufficient for all variants)
- **Time limit**: 2 days per task
- **Concurrent tasks**: Maximum 50 simultaneous tasks

## Backward Compatibility

The updated `analyse_models_array.jl` script remains backward compatible:
- Without seed argument: defaults to seed=43
- With seed argument: uses specified seed

All existing individual variant scripts (analyse_models_array_random25.sh, etc.) continue to work unchanged.
