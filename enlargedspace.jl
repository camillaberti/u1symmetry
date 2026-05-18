module SuperEnrSpace #not sure if I need this


using QuantumToolbox
using StaticArrays
using SparseArrays

export s_enrspace, s_destroy_left, s_destroy_right
"""
create super space with k = 0,1,-1 
we create two dictionaries: super2idx takes a tuple (state, tilde state) 
(so the key is the state in super number representation)
and gives the index, 
idx2super takes the index and gives the super state
"""
#=
n_exc = 1  # max |q|

super_basis = []
for (i, ket) in enumerate(ket_basis)
    for (j, bra) in enumerate(ket_basis)
        q = sum(ket) - sum(bra)
        if abs(q) <= n_exc
            push!(super_basis, (ket, bra, q)) 
        end
    end
end

# assign indices
super2idx = Dict((s[1], s[2]) => i for (i,s) in enumerate(super_basis)) #s[1] and s[2] are the states in number repr of the ket and bra(tilde)
idx2super = Dict(i => s for (i,s) in enumerate(super_basis))
#println(super_basis)
println("Full super space: $(length(ket_basis)^2) states")
println("Restricted super space: $(length(super_basis)) states")
=#
struct s_enrspace{N} #N it is a compile time constant that gives the number of sites, s_enrspace{2} and s_enrspace{3} are different types
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

function s_destroy_right(s::s_enrspace, site::Int) #inspired by enr_destroy
    D = s.total_size # dimension of enlarged reduced hilbert space
    idx2super = s.idx2state
    super2idx = s.state2idx
    I_list, J_list, V_list = Int[], Int[], ComplexF64[]
    

    for (j, (ket, bra)) in idx2super 
        n_k = bra[site] #ask bc in julia the vectorization is column stacked, so the tilde space is on the left
        if n_k > 0                           # same check as enr_destroy: s > 0
            new_bra = setindex(bra, n_k-1, site) #returns a new vector (new_bra) with value nk-1 at index "site"
            new_key = (ket, new_bra)
            #check if the output is in the enlarged restricted basis. If so, we fill a matrix element
            # I would keep it even for the enlarged space for safety and to control the maximum dimension
            if haskey(super2idx, new_key) 
                i = super2idx[new_key]
                push!(I_list, i)
                push!(J_list, j)
                push!(V_list, sqrt(ComplexF64(n_k)))
            end
        end
    end

    return QuantumObject(sparse(I_list, J_list, V_list, D, D); type=Operator(), dims=(D,))
    
end


function s_destroy_left(s::s_enrspace, site::Int)
    D = s.total_size # dimension of reduced hilbert space
    idx2super = s.idx2state
    super2idx = s.state2idx
    I_list, J_list, V_list = Int[], Int[], ComplexF64[]
    

    for (j, (ket, bra)) in idx2super
        n_k = ket[site]
        if n_k > 0                           # same check as enr_destroy: s > 0
            new_ket = setindex(ket, n_k-1, site) #returns a new vector (new_ket) with value nk-1 at index site
            new_key = (new_ket, bra)
            if haskey(super2idx, new_key)    # check output is in restricted basis
                i = super2idx[new_key]
                push!(I_list, i)
                push!(J_list, j)
                push!(V_list, sqrt(ComplexF64(n_k)))
            end
        end
    end

    return QuantumObject(sparse(I_list, J_list, V_list, D, D); type=Operator(), dims=(D,))
end
"""
next step is the equivalent of enr_fock, so something like s_enr_projection that generates the projection operator for the elements 
of the reduced space, so we can generate some initial density matrices. Questions: 
- vectorization?
"""
function s_enr_projector(s_space::s_enrspace, num_list_left, num_list_right) 
    state = (SVector(num_list_left...), SVector(num_list_right...))
    haskey(s_space.state2idx, state) || throw(ArgumentError("state ($num_list_left, $num_list_right) not in extended super basis"))
    i = s_space.state2idx[state]
    i in s_space.target_indices || @warn "state is outside the symmetry blocks" 
    d_tot = s_space.total_size
    vec = zeros(ComplexF64, d_tot)
    vec[i] = 1.

    return QuantumObject(vec; type=OperatorKet(), dims = (d_tot,))
end

function s_enr_projector(dims::Union{AbstractVector{T}, Tuple{Vararg{T}}}, n_excitations::Int, p::Int, num_list_left, num_list_right) where {T <: Integer}
    s = s_enrspace(dims, n_excitations, p)    
    return s_enr_projector(s, num_list_left, num_list_right)
end

end #end of module

