
## wrappers for multi-gpu functionality in cusolverMg


# auxiliary functionality

# NOTE: in the cublasMg preview, which also relies on this functionality, a separate library
#       called 'cudalibmg' is introduced. factor this out when we actually ship that.

mutable struct CudaLibMGDescriptor
    desc::cudaLibMgMatrixDesc_t

    function CudaLibMGDescriptor(a, grid; rowblocks = size(a, 1), colblocks = size(a, 2), elta = eltype(a) )
        desc = Ref{cudaLibMgMatrixDesc_t}()
        try
            cudaLibMgCreateMatrixDesc(desc, size(a, 1), size(a, 2), rowblocks, colblocks, cudaDataType(elta), grid)
        catch e
            println("size(A) = $(size(a)), rowblocks = $rowblocks, colblocks = $colblocks")
            flush(stdout)
            throw(e)
        end
        return new(desc[])
    end
end

Base.cconvert(::Type{cudaLibMgMatrixDesc_t}, obj::CudaLibMGDescriptor) = obj.desc

mutable struct CudaLibMGGrid
    desc::Ref{cudaLibMgGrid_t}

    function CudaLibMGGrid(num_row_devs, num_col_devs, deviceIds, mapping)
        desc = Ref{cudaLibMgGrid_t}()
        cudaLibMgCreateDeviceGrid(desc, num_row_devs, num_col_devs, deviceIds, mapping)
        return new(desc)
    end
end

Base.cconvert(::Type{cudaLibMgGrid_t}, obj::CudaLibMGGrid) = obj.desc[]

function allocateBuffers(n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, descr, mat::Matrix)
    mat_row_block_size = div(size(mat, 1), n_row_devs)
    mat_col_block_size = div(size(mat, 2), n_col_devs)
    mat_buffers  = Vector{CuPtr{Cvoid}}(undef, num_devices)
    mat_numRows  = Vector{Int64}(undef, num_devices)
    mat_numCols  = Vector{Int64}(undef, num_devices)
    streams      = Vector{CuStream}(undef, num_devices)
    typesize = sizeof(eltype(mat))
    ldas = Vector{Int64}(undef, num_devices)
    mat_cpu_bufs = Vector{Matrix{eltype(mat)}}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        ldas[di]    = mat_col_block_size
        dev_row     = mod(di - 1, n_row_devs) + 1
        dev_col     = div(di - 1, n_row_devs) + 1

        mat_row_inds     = ((dev_row-1)*mat_row_block_size+1):min(dev_row*mat_row_block_size, size(mat, 1))
        mat_col_inds     = ((dev_col-1)*mat_col_block_size+1):min(dev_col*mat_col_block_size, size(mat, 2))
        mat_cpu_bufs[di] = Array(mat[mat_row_inds, mat_col_inds])
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        mat_gpu_buf = CuMatrix{eltype(mat)}(undef, size(mat))
        unsafe_copyto!(pointer(mat_gpu_buf), pointer(mat_cpu_bufs[di]), length(mat_cpu_bufs[di]), stream = streams[di], async = true)
        mat_buffers[di] = convert(CuPtr{Cvoid}, pointer(mat_gpu_buf))
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
    end
    device!(deviceIdsGrid[1])
    return mat_buffers
end

function returnBuffers(n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, row_block_size, col_block_size, desc, dDs, D)
    row_block_size = div(size(D, 1), n_row_devs)
    col_block_size = div(size(D, 2), n_col_devs)
    numRows  = [row_block_size for dev in 1:num_devices]
    numCols  = [col_block_size for dev in 1:num_devices]
    typesize = sizeof(eltype(D))
    current_dev = device()
    streams  = Vector{CuStream}(undef, num_devices)
    cpu_bufs = Vector{Matrix{eltype(D)}}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 2))
        cpu_bufs[di] = Matrix{eltype(D)}(undef, length(row_inds), length(col_inds))
        unsafe_copyto!(pointer(cpu_bufs[di]), convert(CuPtr{eltype(D)}, dDs[di]), length(cpu_bufs[di]), stream = streams[di], async = true)
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 2))
        D[row_inds, col_inds] = cpu_bufs[di]
    end
    device!(deviceIdsGrid[1])
    return D
end


## wrappers

function mg_syevd!(jobz::Char, uplo::Char, A; devs=[0], dev_rows=1, dev_cols=length(devs)) # one host-side array A
    ndevs   = length(devs)
    gridRef = Ref{cudaLibMgGrid_t}(C_NULL)
    cusolverMgCreateDeviceGrid(gridRef, 1, ndevs, devs, CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    m, n    = size(A)
    N       = div(size(A, 2), length(devs)) # dimension of the sub-matrix
    descRef = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    #lwork         = Vector{Csize_t}(undef, ndevs)
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    W             = Vector{real(eltype(A))}(undef, n)
    cusolverMgCreateMatrixDesc(descRef, m, n, m, N, convert(cudaDataType, eltype(A)), gridRef[]) # only 1-D column is supported for now
    A_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRef[], A)
    IA            = 1 # for now
    JA            = 1
    cusolverMgSyevd_bufferSize(mg_handle(), jobz, uplo, n, A_ref_arr, IA, JA, descRef[], W, convert(cudaDataType, real(eltype(A))), convert(cudaDataType, eltype(A)), lwork)
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        workspace_ref[di] = convert(CuPtr{Cvoid}, pointer(workspace[di]))
        synchronize()
    end
    device!(devs[1])
    info = Ref{Cint}(C_NULL)
    cusolverMgSyevd(mg_handle(), jobz, uplo, n, A_ref_arr, IA, JA, descRef[], W, convert(cudaDataType, real(eltype(A))), convert(cudaDataType, eltype(A)), workspace_ref, lwork[], info)
    if info[] < 0
        throw(ArgumentError("The $(info[])th parameter is wrong"))
    end
    A = returnBuffers(dev_rows, dev_cols, ndevs, devs, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descRef[], A_ref_arr, A)
    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

function mg_potrf!(uplo::Char, A; devs=[0], dev_rows=1, dev_cols=length(devs)) # one host-side array A
    ndevs   = length(devs)
    gridRef = Ref{cudaLibMgGrid_t}(C_NULL)
    cusolverMgCreateDeviceGrid(gridRef, 1, ndevs, devs, CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    m, n    = size(A)
    N       = div(size(A, 2), length(devs)) # dimension of the sub-matrix
    descRef = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    cusolverMgCreateMatrixDesc(descRef, m, n, m, N, convert(cudaDataType, eltype(A)), gridRef[]) # only 1-D column is supported for now
    A_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRef[], A)
    IA      = 1 # for now
    JA      = 1
    cusolverMgPotrf_bufferSize(mg_handle(), uplo, n, A_ref_arr, IA, JA, descRef[], convert(cudaDataType, eltype(A)), lwork)
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        workspace_ref[di] = convert(CuPtr{Cvoid}, pointer(workspace[di]))
        synchronize()
    end
    device!(devs[1])
    info = Ref{Cint}(C_NULL)
    cusolverMgPotrf(mg_handle(), uplo, n, A_ref_arr, IA, JA, descRef[], convert(cudaDataType, eltype(A)), workspace_ref, lwork[], info)
    if info[] < 0
        throw(ArgumentError("The $(info[])th parameter is wrong"))
    end
    A = returnBuffers(dev_rows, dev_cols, ndevs, devs, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descRef[], A_ref_arr, A)
    return A
end

function mg_potri!(uplo::Char, A; devs=[0], dev_rows=1, dev_cols=length(devs)) # one host-side array A
    ndevs   = length(devs)
    gridRef = Ref{cudaLibMgGrid_t}(C_NULL)
    cusolverMgCreateDeviceGrid(gridRef, 1, ndevs, devs, CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    m, n    = size(A)
    N       = div(size(A, 2), ndevs) # dimension of the sub-matrix
    descRef = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    cusolverMgCreateMatrixDesc(descRef, m, n, m, N, convert(cudaDataType, eltype(A)), gridRef[]) # only 1-D column is supported for now
    A_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRef[], A)
    IA      = 1 # for now
    JA      = 1
    cusolverMgPotri_bufferSize(mg_handle(), uplo, n, A_ref_arr, IA, JA, descRef[], convert(cudaDataType, eltype(A)), lwork)
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        workspace_ref[di] = convert(CuPtr{Cvoid}, pointer(workspace[di]))
        synchronize()
    end
    device!(devs[1])
    info = Ref{Cint}(C_NULL)
    cusolverMgPotri(mg_handle(), uplo, n, A_ref_arr, IA, JA, descRef[], convert(cudaDataType, eltype(A)), workspace_ref, lwork[], info)
    if info[] < 0
        throw(ArgumentError("The $(info[])th parameter is wrong"))
    end
    A = returnBuffers(dev_rows, dev_cols, ndevs, devs, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descRef[], A_ref_arr, A)
    return A
end

function mg_potrs!(uplo::Char, A, B; devs=[0], dev_rows=1, dev_cols=length(devs)) # one host-side array A
    ndevs   = length(devs)
    gridRef = Ref{cudaLibMgGrid_t}(C_NULL)
    cusolverMgCreateDeviceGrid(gridRef, 1, ndevs, devs, CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    ma, na   = size(A)
    mb, nb   = size(A)
    NA       = div(size(A, 2), ndevs) # dimension of the sub-matrix
    NB       = div(size(B, 2), ndevs) # dimension of the sub-matrix
    descRefA = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    descRefB = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    cusolverMgCreateMatrixDesc(descRefA, ma, na, ma, NA, convert(cudaDataType, eltype(A)), gridRef[]) # only 1-D column is supported for now
    cusolverMgCreateMatrixDesc(descRefB, mb, nb, mb, NB, convert(cudaDataType, eltype(B)), gridRef[]) # only 1-D column is supported for now
    A_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRefA[], A)
    B_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRefB[], B)
    IA      = 1 # for now
    JA      = 1
    IB      = 1 # for now
    JB      = 1
    cusolverMgPotrs_bufferSize(mg_handle(), uplo, na, nb, A_ref_arr, IA, JA, descRefA[], B_ref_arr, IB, JB, descRefB[], convert(cudaDataType, eltype(A)), lwork)
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        workspace_ref[di] = convert(CuPtr{Cvoid}, pointer(workspace[di]))
        synchronize()
    end
    device!(devs[1])
    info = Ref{Cint}(C_NULL)
    cusolverMgPotrs(mg_handle(), uplo, na, nb, A_ref_arr, IA, JA, descRefA[], B_ref_arr, IB, JB, descRefB[], convert(cudaDataType, eltype(A)), workspace_ref, lwork[], info)
    if info[] < 0
        throw(ArgumentError("The $(info[])th parameter is wrong"))
    end
    B = returnBuffers(dev_rows, dev_cols, ndevs, devs, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), descRefB[], B_ref_arr, B)
    return B
end

function mg_getrf!(A; devs=[0], dev_rows=1, dev_cols=length(devs)) # one host-side array A
    ndevs   = length(devs)
    gridRef = Ref{cudaLibMgGrid_t}(C_NULL)
    cusolverMgCreateDeviceGrid(gridRef, 1, ndevs, devs, CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    m, n    = size(A)
    N       = div(size(A, 2), length(devs)) # dimension of the sub-matrix
    descRef = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    lwork         = Ref{Int64}(0)
    ipivs         = Vector{CuVector{Cint}}(undef, ndevs)
    ipivs_ref     = Vector{CuPtr{Cint}}(undef, ndevs)
    workspace     = Vector{CuArray}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    cusolverMgCreateMatrixDesc(descRef, m, n, m, N, convert(cudaDataType, eltype(A)), gridRef[]) # only 1-D column is supported for now
    A_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRef[], A)
    IA      = 1 # for now
    JA      = 1
    for (di, dev) in enumerate(devs)
        device!(dev)
        ipivs[di]     = CUDA.zeros(Cint, N)
        ipivs_ref[di] = Base.unsafe_convert(CuPtr{Cint}, ipivs[di])
        synchronize()
    end
    device!(devs[1])
    cusolverMgGetrf_bufferSize(mg_handle(), m, n, A_ref_arr, IA, JA, descRef[], ipivs_ref, convert(cudaDataType, eltype(A)), lwork)
    synchronize()
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        workspace_ref[di] = convert(CuPtr{Cvoid}, pointer(workspace[di]))
        synchronize()
    end
    device!(devs[1])
    info = Ref{Cint}(C_NULL)
    cusolverMgGetrf(mg_handle(), m, n, A_ref_arr, IA, JA, descRef[], ipivs_ref, convert(cudaDataType, eltype(A)), workspace_ref, lwork[], info)
    synchronize()
    if info[] < 0
        throw(ArgumentError("The $(info[])th parameter is wrong"))
    end
    A = returnBuffers(dev_rows, dev_cols, ndevs, devs, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descRef[], A_ref_arr, A)
    ipiv = Vector{Int}(undef, n)
    for (di, dev) in enumerate(devs)
        device!(dev)
        ipiv[((di-1)*N + 1):min((di*N), n)] = collect(ipivs[di])
    end
    device!(devs[1])
    return A, ipiv
end

function mg_getrs!(trans, A, ipiv, B; devs=[0], dev_rows=1, dev_cols=length(devs)) # one host-side array A
    ndevs   = length(devs)
    gridRef = Ref{cudaLibMgGrid_t}(C_NULL)
    cusolverMgCreateDeviceGrid(gridRef, 1, ndevs, devs, CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    ma, na  = size(A)
    mb, nb  = size(B)
    NA      = div(size(A, 2), length(devs)) # dimension of the sub-matrix
    NB      = div(size(B, 2), length(devs)) # dimension of the sub-matrix
    descRefA      = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    descRefB      = Ref{cudaLibMgMatrixDesc_t}(C_NULL)
    lwork         = Ref{Int64}(0)
    ipivs         = Vector{CuVector{Cint}}(undef, ndevs)
    ipivs_ref     = Vector{CuPtr{Cint}}(undef, ndevs)
    workspace     = Vector{CuArray}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    cusolverMgCreateMatrixDesc(descRefA, ma, na, ma, NA, convert(cudaDataType, eltype(A)), gridRef[]) # only 1-D column is supported for now
    cusolverMgCreateMatrixDesc(descRefB, mb, nb, mb, NB, convert(cudaDataType, eltype(B)), gridRef[]) # only 1-D column is supported for now
    A_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRefA[], A)
    B_ref_arr     = allocateBuffers(dev_rows, dev_cols, ndevs, devs, descRefB[], B)
    IA      = 1 # for now
    JA      = 1
    IB      = 1 # for now
    JB      = 1
    for (di, dev) in enumerate(devs)
        device!(dev)
        local_ipiv    = Cint.(ipiv[(di-1)*NA+1:min(di*NA,length(ipiv))])
        ipivs[di]     = CuArray(local_ipiv)
        ipivs_ref[di] = Base.unsafe_convert(CuPtr{Cint}, ipivs[di])
        synchronize()
    end
    device!(devs[1])
    cusolverMgGetrs_bufferSize(mg_handle(), trans, na, nb, A_ref_arr, IA, JA, descRefA[], ipivs_ref, B_ref_arr, IB, JB, descRefB[], convert(cudaDataType, eltype(A)), lwork)
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        workspace_ref[di] = convert(CuPtr{Cvoid}, pointer(workspace[di]))
        synchronize()
    end
    device!(devs[1])
    info = Ref{Cint}(C_NULL)
    cusolverMgGetrs(mg_handle(), trans, na, nb, A_ref_arr, IA, JA, descRefA[], ipivs_ref, B_ref_arr, IB, JB, descRefB[], convert(cudaDataType, eltype(A)), workspace_ref, lwork[], info)
    if info[] < 0
        throw(ArgumentError("The $(info[])th parameter is wrong"))
    end
    B = returnBuffers(dev_rows, dev_cols, ndevs, devs, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), descRefB[], B_ref_arr, B)
    return B
end
