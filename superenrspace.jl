module SuperEnrSpace #not sure if I need this

using QuantumToolbox
using StaticArrays
using SparseArrays
using LinearAlgebra

export s_enr_space, s_enr_destroy, s_enr_projector, s_enr_identity, project_to_target, s_enr_liouvillian, s_enr_mesolve, SuperEnrTimeEvolution
"""
create super space:
two dictionaries are created: super2idx takes a tuple (state, tilde state) 
(so the key is the state in super number representation)
and gives the index, 
idx2super takes the index and gives the super state
"""

struct s_enr_space{N} <: QuantumToolbox.AbstractSpace #N it is a compile time constant that gives the number of sites, s_enrspace{2} and s_enrspace{3} are different types
    total_size::Int #dimension of the space including intermediate states that are not in the reduced space
    size::Int #dimension of the reduced space (the one with |q| ≤ n_excitations)
    blocks::Dict{Int, UnitRange{Int}}
    dims::SVector{N, Int}
    n_excitations::Int
    p::Int #extended cutoff, this is Hamiltonian dependent 
    state2idx::Dict{Tuple{SVector{N,Int}, SVector{N,Int}}, Int}
    idx2state::Dict{Int, Tuple{SVector{N,Int}, SVector{N,Int}}}

    function s_enr_space(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int) where {T <: Integer}
        total_size, size, blocks, state2idx, idx2state = s_enr_dictionaries(dims, n_excitations, p) #size is the dimension of the reduced (enlarged) space
        L = length(dims)
        return new{L}(total_size, size, blocks, SVector{L}(dims), n_excitations, p, state2idx, idx2state)
    end
    
end

Base.length(::s_enr_space{N}) where {N} = N #like enr_space
Base.:(==)(s1::s_enr_space, s2::s_enr_space) = (s1.total_size == s2.total_size) && (s1.dims == s2.dims) #it defines when 2 super spaces are equal

function s_enr_dictionaries(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int) where {T <: Integer}
    # argument checks like in enr
    L = length(dims)
    (L > 0) || throw(DomainError(dims, "dims must be non-empty"))
    all(>=(1), dims) || throw(DomainError(dims, "all elements of dims must be >= 1"))
    (n_excitations >= 0) || throw(DomainError(n_excitations, "n_excitations must be >= 0")) 
    (p >= 1) || throw(DomainError(p, "p must be >= 1")) 

    # extended cutoff: build basis for |k| ≤ n_excitations + p
    n_ext = n_excitations + p
    ranges = ntuple(i -> 0:dims[i]-1, L) #ranges for the occupation numbers of each site
    hilbert_states = vec(collect(Iterators.product(ranges...))) #all possible states in the Hilbert space
    m_vals = map(sum, hilbert_states) # compute m for each state, needed to sort the states by m
    m_perm = sortperm(m_vals)
    hilbert_states = hilbert_states[m_perm]
    m_vals = m_vals[m_perm]
    N = length(hilbert_states)
    m_max = maximum(m_vals)
    # create a dictionary that store the starting index and how many states are there for the specific m value: 
    #m_dict[m] = (start_index, count)
    m_dict = Dict{Int, Tuple{Int, Int}}() 
    #filling the dictionary 
    i = 1 #julia counting starts from 1
    while i <= N
        m = m_vals[i]
        start_index = i
        while i <= N && m_vals[i] == m
            i += 1
        end
        m_dict[m] = (start_index, i - start_index)
    end

    #block ordering for the super space, the ordering will be q = 0, -1, +1, -2, +2 and so on
    q_order = Int[0]
    for k in 1:n_ext
        push!(q_order, -k, +k)
    end

    state2idx = Dict{Tuple{SVector{L,Int}, SVector{L,Int}}, Int}()
    idx2state = Dict{Int, Tuple{SVector{L,Int}, SVector{L,Int}}}()

    #preallocate memory, it should optimize performances
    sizehint!(state2idx, N^2)
    sizehint!(idx2state, N^2)
    idx = 1
    size = 0 #size of the reduced space, it will be updated as the dictionaries are filled 
    dict_blocks = Dict{Int, UnitRange{Int}}() 
    for q in q_order 
        q_start = idx      
        for m in 0:m_max
            n = m-q 
            (0 <= n <= m_max) || continue
            m_start, m_count = m_dict[m]
            n_start, n_count = m_dict[n]
            for i in m_start:(m_start + m_count - 1)
                for j in n_start:(n_start + n_count - 1)
                    superstate = (SVector{L,Int}(hilbert_states[i]), SVector{L,Int}(hilbert_states[j]))
                    state2idx[superstate] = idx
                    idx2state[idx] = superstate
                    idx += 1
                    abs(q) <= n_excitations && (size += 1)
                end
            end
        end
        if idx > q_start
            dict_blocks[q] = q_start:(idx-1) #store the range of indices for the block with charge q
        end
    end
    enlarged_size = idx - 1  

    return enlarged_size, size, dict_blocks, state2idx, idx2state
end

"""
Next step: build destroy operators in the resticted super space. 
Essentially they take two dictionaries (state => idx and idx => state) and the site (to create a1, a2 ... for each site)
we create the matrix a_site_left/right
the operators are defined in the enlarged restricted space, so that if at an intermediate step the operators take the state out of the
reduced space it is fine, as long as the final output is in the reduced space. 
"""

function s_enr_destroy(s::s_enr_space, site::Int)
    D = s.total_size 
    idx2super = s.idx2state
    super2idx = s.state2idx
    I_list_left, J_list_left, V_list_left = Int[], Int[], ComplexF64[]
    I_list_right, J_list_right, V_list_right = Int[], Int[], ComplexF64[]

    for (j, (ket, bra)) in idx2super
        n_l = ket[site]
        n_r = bra[site]
        if n_l > 0                           # same check as enr_destroy: s > 0
            new_ket = setindex(ket, n_l-1, site) #returns a new vector (new_ket) with value nk-1 at index site
            new_key_l = (new_ket, bra)
            if haskey(super2idx, new_key_l)    # check output is in restricted basis
                i = super2idx[new_key_l]
                push!(I_list_left, i)
                push!(J_list_left, j)
                push!(V_list_left, sqrt(ComplexF64(n_l)))
            end
        end
        if n_r > 0
            new_bra = setindex(bra, n_r-1, site)
            new_key_r = (ket, new_bra)
            if haskey(super2idx, new_key_r)
                i = super2idx[new_key_r]
                push!(I_list_right, i)
                push!(J_list_right, j)
                push!(V_list_right, sqrt(ComplexF64(n_r)))
            end
        end
    end
    a_left = QuantumObject(sparse(I_list_left, J_list_left, V_list_left, D, D); type=Operator(), dims=(D,))
    a_right = QuantumObject(sparse(I_list_right, J_list_right, V_list_right, D, D); type=Operator(), dims=(D,))
    return (a_left, a_right)
end

function s_enr_destroy(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int, site::Int) where {T <: Integer}
    s = s_enr_space(dims, n_excitations, p)
    return s_enr_destroy(s, site)
end

"""
next step is the equivalent of enr_fock, so something like s_enr_projection that generates the projection operator for the elements 
of the reduced space, so we can generate some initial density matrices. Questions: 
- vectorization?
"""

function project_to_target(op::QuantumObject, s::s_enr_space)
    d = s.size
    new_data = op.data[1:d, 1:d]
    return QuantumObject(new_data; type = Operator(), dims = (d,))
end

function s_enr_projector(s::s_enr_space, num_list_left, num_list_right; project::Bool=true) 
    state = (SVector(num_list_left...), SVector(num_list_right...)) #... splat operators, this becomes SVector(2,1,0) for example
    haskey(s.state2idx, state) || throw(ArgumentError("state ($num_list_left, $num_list_right) not in extended super basis"))
    i = s.state2idx[state]

    if project
        d = s.size
        i in 1:d || throw(ArgumentError("state ($num_list_left, $num_list_right) is outside target blocks; use project=false to access it"))
        vec = zeros(ComplexF64, d)
        vec[i] = 1.
        return QuantumObject(vec; type=Ket(), dims=(d,))
    else
        d_tot = s.total_size
        vec = zeros(ComplexF64, d_tot)
        vec[i] = 1.
        return QuantumObject(vec; type=Ket(), dims=(d_tot,))
    end
end

function s_enr_projector(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int, num_list_left, num_list_right) where {T <: Integer}
    s = s_enr_space(dims, n_excitations, p)    
    return s_enr_projector(s, num_list_left, num_list_right)
end
"""
last step: function that creates a vectorized identity in the reduced superspace. 
This is needed as a sanity check (we can see if <I|L = 0) holds in the reduced space, 
and we can compute expectation values thanks to it 
(reminder that for a generic observable O, <O> = Tr(Oρ) = <I|Oρ|I> = <I|O|ρ>)
"""
function s_enr_identity(s::s_enr_space; project::Bool=true)
    d_tot = s.total_size 
    vec_id = zeros(ComplexF64, d_tot)
    for (j, (ket, bra)) in s.idx2state
        if ket == bra
            vec_id[j] = 1.
        end
    end
    if project
        d = s.size
        vec_r = vec_id[1:d]
        return QuantumObject(vec_r; type=Ket(), dims =(d,))
    else
        return QuantumObject(vec_id; type=Ket(), dims =(d_tot,))
    end
end


function s_enr_liouvillian(s::s_enr_space, H_left, H_right, c_ops_lr; project::Bool=true)
    L = -1im * (H_left - H_right)
    for (c_left, c_right) in c_ops_lr
        L += c_left * c_right - 0.5 * (c_left)' * c_left - 0.5 * (c_right)' * c_right
    end
    return project ? project_to_target(L, s) : L
end

struct SuperEnrTimeEvolution
    times::AbstractVector
    states::Vector{QuantumObject}   # ρ(t) at each time, always returned
    expect::Union{Matrix{ComplexF64}, Nothing}  # nothing if no observables passed
    alg::Symbol
end

function s_enr_mesolve(L::QuantumObject, ρ0::QuantumObject, tlist, vec_id::QuantumObject;
                   observables=nothing, method::Symbol=:eigen)
    L_mat = Matrix(L.data)
    ρ0_vec = ρ0.data
    id_vec = vec_id.data

    nT = length(tlist)
    states = Vector{QuantumObject}(undef, nT)

    # preallocate expectation values only if observables are passed
    if observables !== nothing
        nO = length(observables)
        results = zeros(ComplexF64, nO, nT)
    end

    if method == :eigen
        F = eigen(L_mat)
        D_eig, V = F.values, F.vectors
        Vinv = inv(V)
        c0 = Vinv * ρ0_vec

        for (it, t) in enumerate(tlist)
            ρ_t = V * (exp.(D_eig * t) .* c0)
            states[it] = QuantumObject(ρ_t; type=Ket(), dims=(length(ρ_t),))
            if observables !== nothing
                for (io, O) in enumerate(observables)
                    results[io, it] = dot(id_vec, O.data * ρ_t)
                end
            end
        end

    else  # :exp
        for (it, t) in enumerate(tlist)
            ρ_t = exp(L_mat * t) * ρ0_vec
            states[it] = QuantumObject(ρ_t; type=Ket(), dims=(length(ρ_t),))
            if observables !== nothing
                for (io, O) in enumerate(observables)
                    results[io, it] = dot(id_vec, O.data * ρ_t)
                end
            end
        end
    end

    expect = observables !== nothing ? results : nothing
    return SuperEnrTimeEvolution(tlist, states, expect, method)
end

end #end of module

