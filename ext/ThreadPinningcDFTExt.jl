module ThreadPinningcDFTExt

using ClassicalDFT
using ThreadPinning
using KernelAbstractions: CPU

function ClassicalDFT.CPU(ncpu::Int, device_ids::Vector{Int})
    if length(device_ids) != ncpu
        @warn "length(device_ids)=$(length(device_ids)) does not match ncpu=$ncpu; pinning to $(length(device_ids)) cores."
    end
    if ncpu != Threads.nthreads()
        @warn "ncpu=$ncpu does not match Threads.nthreads()=$(Threads.nthreads()); Julia's thread count is fixed at process startup (`julia -t $ncpu`) and cannot be changed by this constructor — only the pinning of existing threads to cores is applied here."
    end
    ThreadPinning.pinthreads(device_ids)
    return CPU(; static=true)
end

end