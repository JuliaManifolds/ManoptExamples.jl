@inline _unwrap_extruded_arg(x) = x
@inline _unwrap_extruded_arg(x::Base.Broadcast.Extruded) = x.x

@inline function _axpy_partition!(dest::AbstractArray{T}, src::AbstractArray{T}) where {T}
    axes(dest) == axes(src) ||
        throw(DimensionMismatch("ArrayPartition blocks must have matching axes for in-place addition."))
    @inbounds for i in eachindex(dest, src)
        dest[i] += src[i]
    end
    return dest
end

@inline function _axpy_partition!(
        dest::StridedArray{T},
        src::Base.ReshapedArray{
            T,
            N,
            <:SubArray{T, 1, <:BlockNonzeroVector, Tuple{UnitRange{I}}, false},
            MI,
        },
    ) where {T, N, I <: Integer, MI}
    axes(dest) == axes(src) ||
        throw(DimensionMismatch("ArrayPartition blocks must have matching axes for in-place addition."))

    src_view = parent(src)
    src_parent = parent(src_view)
    src_range = parentindices(src_view)[1]
    src_first = first(src_range)
    src_last = last(src_range)
    dest_linear = vec(dest)

    for k in eachindex(src_parent.blocks)
        block = src_parent.blocks[k]
        block_start = src_parent.starts[k]
        block_end = block_start + length(block) - 1

        overlap_start = max(src_first, block_start)
        overlap_end = min(src_last, block_end)
        overlap_start > overlap_end && continue

        src_local =
            (overlap_start - src_first + 1):(overlap_end - src_first + 1)
        block_local =
            (overlap_start - block_start + 1):(overlap_end - block_start + 1)

        @views dest_linear[src_local] .+= block[block_local]
    end
    return dest
end


@inline function _add_arraypartition_blocks!(
        dest::ArrayPartition{T, <:Tuple{AbstractArray{T}, Vararg{AbstractArray{T}}}},
        src::ArrayPartition,
    ) where {T}
    length(dest.x) == length(src.x) ||
        throw(DimensionMismatch("ArrayPartition blocks must have matching lengths for in-place addition."))
    map(_axpy_partition!, dest.x, src.x)
    return dest
end

function Base.copyto!(
        dest::ArrayPartition{T, <:Tuple{Array{T, 3}, Matrix{T}, Matrix{T}}},
        bc::Base.Broadcast.Broadcasted{RecursiveArrayTools.ArrayPartitionStyle{Style}, Axes, typeof(+)},
    ) where {T, Style <: Union{Nothing, Base.Broadcast.BroadcastStyle}, Axes}
    bc = Base.Broadcast.instantiate(bc)
    if length(bc.args) == 2
        lhs = _unwrap_extruded_arg(bc.args[1])
        rhs = _unwrap_extruded_arg(bc.args[2])
        if lhs === dest && rhs isa ArrayPartition
            return _add_arraypartition_blocks!(dest, rhs)
        elseif rhs === dest && lhs isa ArrayPartition
            return _add_arraypartition_blocks!(dest, lhs)
        end
    end
    return invoke(Base.copyto!, Tuple{ArrayPartition, Base.Broadcast.Broadcasted}, dest, bc)
end
