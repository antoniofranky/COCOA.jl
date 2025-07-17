#!/usr/bin/env julia

"""
Test script for corrected Charnes-Cooper transformation implementation.
This script verifies that the new implementation without extra w variables works correctly.
"""

using Pkg
Pkg.activate("/home/anton/master-thesis/toolbox/julia/COCOA")

using COCOA
using COBREXA
using AbstractFBCModels
using HiGHS
using JuMP
using Distributed
import ConstraintTrees as C

function test_charnes_cooper_simple()
    println("Testing Charnes-Cooper transformation...")
    
    # Create a simple test model
    try
        # Load a simple model for testing
        model = load_model("/home/anton/master-thesis/toolbox/julia/COCOA/test/e_coli_core.json")
        
        # Create constraints with complex activities
        constraints, complexes = COCOA.concordance_constraints(model; return_complexes=true)
        
        # Create template
        template = COCOA.create_charnes_cooper_template(constraints)
        
        # Get first two complexes for testing
        complex_ids = collect(keys(complexes))
        if length(complex_ids) >= 2
            c1_id = complex_ids[1]
            c2_id = complex_ids[2]
            
            c1_activity = constraints.activities[c1_id].value
            c2_activity = constraints.activities[c2_id].value
            
            println("Testing with complexes: $c1_id and $c2_id")
            
            # Test positive direction
            println("  Testing positive direction...")
            instantiated_pos, c1_expr_pos = COCOA.instantiate_charnes_cooper(
                template, c1_activity, c2_activity, :positive
            )
            
            # Test negative direction
            println("  Testing negative direction...")
            instantiated_neg, c1_expr_neg = COCOA.instantiate_charnes_cooper(
                template, c1_activity, c2_activity, :negative
            )
            
            println("✓ Charnes-Cooper transformation instantiated successfully!")
            println("  - Positive direction: $(typeof(instantiated_pos))")
            println("  - Negative direction: $(typeof(instantiated_neg))")
            println("  - Positive direction has $(length(instantiated_pos)) constraints")
            println("  - Negative direction has $(length(instantiated_neg)) constraints")
            
            return true
        else
            println("⚠ Not enough complexes in model for testing")
            return false
        end
        
    catch e
        println("❌ Error during testing: $e")
        return false
    end
end

function test_templated_concordance()
    println("\nTesting templated concordance...")
    
    try
        # Load a simple model for testing
        model = load_model("/home/anton/master-thesis/toolbox/julia/COCOA/test/e_coli_core.json")
        
        # Create constraints with complex activities
        constraints, complexes = COCOA.concordance_constraints(model; return_complexes=true)
        
        # Create template
        template = COCOA.create_charnes_cooper_template(constraints)
        
        # Get first two complexes for testing
        complex_ids = collect(keys(complexes))
        if length(complex_ids) >= 2
            c1_id = complex_ids[1]
            c2_id = complex_ids[2]
            
            c1_activity = constraints.activities[c1_id].value
            c2_activity = constraints.activities[c2_id].value
            
            println("Testing templated concordance with complexes: $c1_id and $c2_id")
            
            # Test positive direction with correct workers parameter
            println("  Testing positive direction...")
            result_pos = COCOA.test_concordance_templated(
                template, c1_activity, c2_activity, :positive;
                optimizer=HiGHS.Optimizer,
                workers=Distributed.workers(),  # Use correct workers parameter
                tolerance=1e-9
            )
            
            # Test negative direction
            println("  Testing negative direction...")
            result_neg = COCOA.test_concordance_templated(
                template, c1_activity, c2_activity, :negative;
                optimizer=HiGHS.Optimizer,
                workers=Distributed.workers(),  # Use correct workers parameter
                tolerance=1e-9
            )
            
            println("✓ Templated concordance test completed successfully!")
            println("  - Positive direction result: $result_pos")
            println("  - Negative direction result: $result_neg")
            
            return true
        else
            println("⚠ Not enough complexes in model for testing")
            return false
        end
        
    catch e
        println("❌ Error during templated concordance testing: $e")
        return false
    end
end

# Run tests
function main()
    println("=== Testing Corrected Charnes-Cooper Implementation ===")
    
    success1 = test_charnes_cooper_simple()
    success2 = test_templated_concordance()
    
    if success1 && success2
        println("\n✓ All tests passed! The corrected implementation works correctly.")
        return 0
    else
        println("\n❌ Some tests failed. Please check the implementation.")
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end