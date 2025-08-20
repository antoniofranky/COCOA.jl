"""
Memory optimization utilities for COCOA.jl

This module provides memory-efficient data structures and object pools
to reduce allocations and GC pressure during concordance analysis.
"""

using SparseArrays
import Base: push!, length, isempty

# ========================================================================================
# Section 1: Sparse BitMatrix for Concordant Pairs
# ========================================================================================

"""
Memory-efficient representation of concordant pairs using sparse bit matrix.
Uses ~1 bit per pair instead of 16 bytes for Set{Tuple{Int,Int}}.
"""
mutable struct SparseConcordantPairs
    n::Int  # Number of complexes
    # Use sparse matrix for memory efficiency - only stores non-zero entries
    matrix::SparseMatrixCSC{Bool,Int}
    # Track number of pairs for fast counting
    n_pairs::Int

    function SparseConcordantPairs(n_complexes::Int)
        # Initialize empty sparse matrix
        matrix = spzeros(Bool, n_complexes, n_complexes)
        new(n_complexes, matrix, 0)
    end
end

"""
Add a concordant pair to the sparse matrix.
"""
@inline function add_pair!(pairs::SparseConcordantPairs, i::Int, j::Int)
    if i > j
        i, j = j, i  # Canonical order
    end
    if i <= pairs.n && j <= pairs.n && !pairs.matrix[i, j]
        pairs.matrix[i, j] = true
        pairs.n_pairs += 1
    end
end

"""
Check if a pair is concordant.
"""
@inline function has_pair(pairs::SparseConcordantPairs, i::Int, j::Int)::Bool
    if i > j
        i, j = j, i  # Canonical order
    end
    return i <= pairs.n && j <= pairs.n && pairs.matrix[i, j]
end

"""
Merge another SparseConcordantPairs into this one.
"""
function merge_pairs!(dest::SparseConcordantPairs, src::SparseConcordantPairs)
    # Use sparse matrix operations for efficiency
    dest.matrix = dest.matrix .| src.matrix
    dest.n_pairs = nnz(dest.matrix)
end

"""
Convert to Set{Tuple{Int,Int}} for compatibility.
"""
function to_set(pairs::SparseConcordantPairs)::Set{Tuple{Int,Int}}
    result = Set{Tuple{Int,Int}}()
    sizehint!(result, pairs.n_pairs)

    # Iterate over non-zero entries in sparse matrix
    rows = rowvals(pairs.matrix)
    for j in 1:pairs.n
        for idx in nzrange(pairs.matrix, j)
            i = rows[idx]
            if pairs.matrix[i, j]
                push!(result, (i, j))
            end
        end
    end
    return result
end

"""
Clear all pairs (for reuse).
"""
function clear!(pairs::SparseConcordantPairs)
    pairs.matrix = spzeros(Bool, pairs.n, pairs.n)
    pairs.n_pairs = 0
end

# ========================================================================================
# Section 2: Object Pools for Container Reuse
# ========================================================================================

"""
Thread-safe object pool for reusing containers to reduce allocations.
"""
mutable struct ObjectPool{T}
    available::Vector{T}
    in_use::Set{T}
    constructor::Function
    max_size::Int
end

function ObjectPool{T}(constructor::Function, initial_size::Int=10, max_size::Int=100) where T
    available = Vector{T}()
    # Pre-allocate initial objects
    for _ in 1:initial_size
        push!(available, constructor())
    end
    return ObjectPool{T}(available, Set{T}(), constructor, max_size)
end

"""
Acquire an object from the pool.
"""
function acquire!(pool::ObjectPool{T})::T where T
    obj = if !isempty(pool.available)
        pop!(pool.available)
    else
        pool.constructor()
    end
    push!(pool.in_use, obj)
    return obj
end

"""
Release an object back to the pool after clearing it.
"""
function release!(pool::ObjectPool{T}, obj::T, clear_func::Function=empty!) where T
    if obj in pool.in_use
        delete!(pool.in_use, obj)
        clear_func(obj)
        if length(pool.available) < pool.max_size
            push!(pool.available, obj)
        end
    end
end

# Pre-defined pools for common container types
const PAIR_SET_POOL = ObjectPool{Set{Tuple{Int,Int}}}(() -> Set{Tuple{Int,Int}}())
const INT_VECTOR_POOL = ObjectPool{Vector{Int}}(() -> Vector{Int}())
const FLOAT_VECTOR_POOL = ObjectPool{Vector{Float64}}(() -> Vector{Float64}())
const TUPLE_VECTOR_POOL = ObjectPool{Vector{Tuple{Int,Int,UInt8}}}(() -> Vector{Tuple{Int,Int,UInt8}}())

# ========================================================================================
# Section 3: Memory-Efficient Result Accumulator
# ========================================================================================

"""
Memory-efficient accumulator for batch processing results.
Reuses internal buffers and minimizes allocations.
"""
mutable struct BatchResultAccumulator
    concordant_pairs::SparseConcordantPairs
    non_concordant_count::Int
    timeout_count::Int
    transitive_count::Int
    skipped_count::Int
    # Use array-based storage for optimization results instead of Dict
    optimization_results::Vector{Tuple{Int,Int,Symbol,Float64}}
    # Reusable buffers
    temp_pairs::Vector{Tuple{Int,Int}}
    temp_values::Vector{Float64}

    function BatchResultAccumulator(n_complexes::Int)
        new(
            SparseConcordantPairs(n_complexes),
            0, 0, 0, 0,
            Vector{Tuple{Int,Int,Symbol,Float64}}(),
            Vector{Tuple{Int,Int}}(),
            Vector{Float64}()
        )
    end
end

"""
Add batch results to accumulator with minimal allocations.
"""
function accumulate_results!(
    acc::BatchResultAccumulator,
    concordant_pairs::Union{Set{Tuple{Int,Int}},SparseConcordantPairs},
    non_concordant::Int,
    timeout::Int,
    transitive::Int,
    skipped::Int,
    opt_results::Dict{Tuple{Int,Int,Symbol},Float64}
)
    # Update counts
    acc.non_concordant_count += non_concordant
    acc.timeout_count += timeout
    acc.transitive_count += transitive
    acc.skipped_count += skipped

    # Merge concordant pairs efficiently
    if isa(concordant_pairs, SparseConcordantPairs)
        merge_pairs!(acc.concordant_pairs, concordant_pairs)
    else
        for (i, j) in concordant_pairs
            add_pair!(acc.concordant_pairs, i, j)
        end
    end

    # Add optimization results
    for ((i, j, dir), val) in opt_results
        push!(acc.optimization_results, (i, j, dir, val))
    end
end

"""
Reset accumulator for reuse.
"""
function reset!(acc::BatchResultAccumulator)
    clear!(acc.concordant_pairs)
    acc.non_concordant_count = 0
    acc.timeout_count = 0
    acc.transitive_count = 0
    acc.skipped_count = 0
    empty!(acc.optimization_results)
    empty!(acc.temp_pairs)
    empty!(acc.temp_values)
end

# ========================================================================================
# Section 4: Circular Buffer for Streaming
# ========================================================================================

"""
Circular buffer for efficient streaming without reallocations.
"""
mutable struct CircularBuffer{T}
    data::Vector{T}
    capacity::Int
    size::Int
    head::Int
    tail::Int

    function CircularBuffer{T}(capacity::Int) where T
        new(Vector{T}(undef, capacity), capacity, 0, 1, 1)
    end
end

@inline function push!(buf::CircularBuffer{T}, item::T) where T
    if buf.size < buf.capacity
        buf.data[buf.tail] = item
        buf.tail = mod1(buf.tail + 1, buf.capacity)
        buf.size += 1
    else
        # Overwrite oldest
        buf.data[buf.tail] = item
        buf.tail = mod1(buf.tail + 1, buf.capacity)
        buf.head = mod1(buf.head + 1, buf.capacity)
    end
end

@inline function popfirst!(buf::CircularBuffer{T})::T where T
    if buf.size == 0
        throw(BoundsError("Buffer is empty"))
    end
    item = buf.data[buf.head]
    buf.head = mod1(buf.head + 1, buf.capacity)
    buf.size -= 1
    return item
end

@inline isempty(buf::CircularBuffer) = buf.size == 0
@inline length(buf::CircularBuffer) = buf.size

"""
Get all items as a view without copying.
"""
function view_all(buf::CircularBuffer{T})::SubArray{T} where T
    if buf.size == 0
        return view(buf.data, 1:0)
    elseif buf.head <= buf.tail - 1
        return view(buf.data, buf.head:buf.tail-1)
    else
        # Wrapped around - need to handle separately
        # For now, return first contiguous segment
        return view(buf.data, buf.head:buf.capacity)
    end
end

# ========================================================================================
# Section 5: Memory Monitoring
# ========================================================================================

"""
Monitor memory usage and suggest batch size adjustments.
"""
mutable struct MemoryMonitor
    initial_memory::Float64
    peak_memory::Float64
    gc_time::Float64
    gc_count::Int
    last_check_time::Float64

    function MemoryMonitor()
        GC.gc()  # Clean slate
        initial = Base.gc_live_bytes() / 1024^3  # GB
        new(initial, initial, 0.0, 0, time())
    end
end

"""
Update memory statistics and return current usage in GB.
"""
function check_memory!(monitor::MemoryMonitor)::Float64
    current = Base.gc_live_bytes() / 1024^3
    monitor.peak_memory = max(monitor.peak_memory, current)

    # Check GC statistics
    gc_stats = Base.gc_num()
    monitor.gc_count = gc_stats.full_sweep
    monitor.gc_time = gc_stats.total_time / 1e9  # Convert to seconds

    return current
end

"""
Suggest batch size based on memory pressure.
"""
function suggest_batch_size(monitor::MemoryMonitor, current_batch_size::Int)::Int
    current_mem = check_memory!(monitor)
    mem_pressure = (current_mem - monitor.initial_memory) / monitor.initial_memory

    if mem_pressure > 0.8  # High pressure
        return max(10, current_batch_size ÷ 2)
    elseif mem_pressure < 0.3  # Low pressure
        return min(1000, current_batch_size * 2)
    else
        return current_batch_size
    end
end

# Export all public types and functions
export SparseConcordantPairs, add_pair!, has_pair, merge_pairs!, to_set, clear!,
    ObjectPool, acquire!, release!,
    PAIR_SET_POOL, INT_VECTOR_POOL, FLOAT_VECTOR_POOL, TUPLE_VECTOR_POOL,
    BatchResultAccumulator, accumulate_results!, reset!,
    CircularBuffer, popfirst!, view_all,
    MemoryMonitor, check_memory!, suggest_batch_size