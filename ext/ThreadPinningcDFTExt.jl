module ThreadPinningcDFTExt

using ClassicalDFT
using ThreadPinning

function ClassicalDFT.CPU(ncpu::Int,device_ids::Vector{Int}) 
    ThreadPinning.pinthreads(device_ids)
    return CPU(ncpu,true,device_ids)
end

end