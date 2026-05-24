module SuperEnrSpace #not sure if I need this

using QuantumToolbox
#import QuantumToolbox: dimensions_to_dims, Dimensions, LiouvilleSpace, Space, get_size
using StaticArrays
using SparseArrays
#import QuantumToolbox: get_size, dimensions_to_dims 

export s_enrspace, s_destroy, s_enr_projector, s_identity
"""
create super space with k = 0,1,-1 
we create two dictionaries: super2idx takes a tuple (state, tilde state) 
(so the key is the state in super number representation)
and gives the index, 
idx2super takes the index and gives the super state
"""

struct s_enrspace{N} <: QuantumToolbox.AbstractSpace #N it is a compile time constant that gives the number of sites, s_enrspace{2} and s_enrspace{3} are different types
    total_size::Int #dimension of the space including intermediate states that are not in the reduced space
    size::Int #dimension of the reduced space (the one with |q| ≤ n_excitations)
    target_indices::Vector{Int} #indices of the states in the full super space that are in the reduced space (the one with |q| ≤ n_excitations)
    dims::SVector{N, Int}
    n_excitations::Int
    p::Int #extended cutoff, this is Hamiltonian dependent 
    state2idx::Dict{Tuple{SVector{N,Int}, SVector{N,Int}}, Int}
    idx2state::Dict{Int, Tuple{SVector{N,Int}, SVector{N,Int}}}

    function s_enrspace(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int) where {T <: Integer}
        total_size, size, target_indices, state2idx, idx2state = s_enr_dictionaries(dims, n_excitations, p) #size is the dimension of the reduced (enlarged) space
        L = length(dims)
        return new{L}(total_size, size, target_indices, SVector{L}(dims), n_excitations, p, state2idx, idx2state)
    end
    
end

#QuantumToolbox.get_size(s::s_enrspace) = s.total_size
Base.length(::s_enrspace{N}) where {N} = N #like enr_space
Base.:(==)(s1::s_enrspace, s2::s_enrspace) = (s1.total_size == s2.total_size) && (s1.dims == s2.dims) #it defines when 2 super spaces are equal
#dimensions_to_dims(s::s_enrspace) = [s.total_size]
#the following two functions were needed for OperatorKet() type
#dimensions_to_dims(s::s_enrspace) = SVector{1,Int}(s.total_size) #important to think about this!!!
#get_size(s::s_enrspace) = s.total_size
function s_enr_dictionaries(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int) where {T <: Integer}
    # argument checks like in enr
    L = length(dims)
    (L > 0) || throw(DomainError(dims, "dims must be non-empty"))
    all(>=(1), dims) || throw(DomainError(dims, "all elements of dims must be >= 1"))
    (n_excitations >= 0) || throw(DomainError(n_excitations, "n_excitations must be >= 0")) #to be checked, n_exc can be zero
    (p >= 0) || throw(DomainError(p, "p must be >= 0")) #to be checked 

    # extended cutoff: build basis for |k| ≤ n_excitations + p
    n_ext = n_excitations + p
    #question: can I optimize this? Like for example by only generating the super states that satisfy |q| ≤ n_ext? 
    #maybe there is a smarter way to do this, look at enr
    # generate all valid single-site basis states for ket and bra
    # each is a SVector of length L with n_i ∈ {0, ..., dims[i]-1}
    all_states = [SVector{L,Int}(s) for s in Iterators.product(ntuple(i -> 0:dims[i]-1, L)...)]

    # enumerate all (ket, bra) pairs satisfying the extended charge constraint
    result = Tuple{SVector{L,Int}, SVector{L,Int}}[]

    for ket in all_states
        for bra in all_states
            q = sum(ket) - sum(bra)
            if abs(q) <= n_ext #and not n_exc !!
                push!(result, (ket, bra))
            end
        end
    end

    target_indices = [i for (i, (ket, bra)) in enumerate(result) if abs(sum(ket) - sum(bra)) <= n_excitations]
    size = length(target_indices)

    enlarged_size = length(result)
    state2idx = Dict(state => i for (i, state) in enumerate(result))
    idx2state = Dict(i => state for (i, state) in enumerate(result))

    return enlarged_size, size, target_indices, state2idx, idx2state
end

"""
step 3 is to build operator in the resticted super space. I did two functions, one for left operators and one for right. -> Compact into one
Essentially they take two dictionaries (state => idx and idx => state) and the site (to create a1, a2 ... for each site)
we create the matrix a_site_left/right
the operators are defined in the enlarged restricted space, so that if at an intermediate step the operators take the state out of the
reduced space it is fine, as long as the final output is in the reduced space. 
"""

function s_destroy(s::s_enrspace, site::Int)
    D = s.total_size # dimension of reduced hilbert space
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

function s_destroy(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int, site::Int) where {T <: Integer}
    s = s_enrspace(dims, n_excitations, p)
    return s_destroy(s, site)
end

"""
next step is the equivalent of enr_fock, so something like s_enr_projection that generates the projection operator for the elements 
of the reduced space, so we can generate some initial density matrices. Questions: 
- vectorization?
"""
function s_enr_projector(s_space::s_enrspace, num_list_left, num_list_right) 
    state = (SVector(num_list_left...), SVector(num_list_right...)) #... splat operators, this becomes SVector(2,1,0) for example
    haskey(s_space.state2idx, state) || throw(ArgumentError("state ($num_list_left, $num_list_right) not in extended super basis"))
    i = s_space.state2idx[state]
    i in s_space.target_indices || @warn "state is outside the symmetry blocks" 
    d_tot = s_space.total_size
    vec = zeros(ComplexF64, d_tot)
    vec[i] = 1.
    

    return QuantumObject(vec; type=Ket(), dims=d_tot)

end

function s_enr_projector(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int, num_list_left, num_list_right) where {T <: Integer}
    s = s_enrspace(dims, n_excitations, p)    
    return s_enr_projector(s, num_list_left, num_list_right)
end
"""
last step: function that creates a vectorized identity in the reduced superspace. 
This is needed as a sanity check (we can see if <I|L = 0) holds in the reduced space, 
and we can compute expectation values thanks to it 
(reminder that for a generic observable O, <O> = Tr(Oρ) = <I|Oρ|I> = <I|O|ρ>)
"""
function s_identity(s::s_enrspace)
    d = s.total_size #can I construct it only in the blocks I care about? For now let's keep it enlarged
    vec_id = zeros(ComplexF64, d)
    for (j, (ket, bra)) in s.idx2state
        if ket == bra
            vec_id[j] = 1.
        end
    end
    return QuantumObject(vec_id; type=Ket(), dims = d)
end
end #end of module

