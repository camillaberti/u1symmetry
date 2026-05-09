using QuantumToolbox
"""
write the basis in number representation, 2 sites with dims 3 (meaning each site can be in state 0,1,2)
"""
dims = (3, 3)

ket_basis = vec(collect(Iterators.product(0:dims[1]-1, 0:dims[2]-1)))
#creating a dictionary
ket2idx = Dict(s => i for (i,s) in enumerate(ket_basis))
idx2ket = Dict(i => s for (i,s) in enumerate(ket_basis))
#=
for (i, s) in idx2ket
    println("Dictionary: Index: $i, Ket: $s")
end
for k in ket_basis
    println("Ket basis: $k")
end
=#
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
super2idx = Dict((s[1], s[2]) => i for (i,s) in enumerate(super_basis))
idx2super = Dict(i => s for (i,s) in enumerate(super_basis))
println(super_basis)
println("Full super space: $(length(ket_basis)^2) states")
println("Restricted super space: $(length(super_basis)) states")
"""
step 3 is to build operator in the resticted super space. I did two functions, one for left operators and one for right. Essentially they take
two dictionaries (state => idx and idx => state) and the site (to create a1, a2 ... for each site)
we create the matrix a_site
"""

function s_destroy_right(super2idx, idx2super, site)
    D = length(super2idx) # dimension of reduced hilbert space
    I_list, J_list, V_list = Int[], Int[], ComplexF64[]
    

    for (j, (ket, bra)) in idx2super
        n_k = bra[site]
        if n_k > 0                           # same check as enr_destroy: s > 0
            new_bra = setindex(bra, n_k-1, site) #returns a new vector (new_ket) with value nk-1 at index site
            new_key = (ket, new_bra)
            if haskey(super2idx, new_key)    # check output is in restricted basis
                i = super2idx[new_key]
                push!(I_list, i)
                push!(J_list, j)
                push!(V_list, sqrt(ComplexF64(n_k)))
            end
        end
    end

    return QuantumObject(sparse(I_list, J_list, V_list, D, D), Operator())
end


function s_destroy_left(super2idx, idx2super, site)
    D = length(super2idx) # dimension of reduced hilbert space
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

    return QuantumObject(sparse(I_list, J_list, V_list, D, D), Operator())
end
#operators I defined in the reduced space
a1_left = s_destroy_left(super2idx, idx2super, 1)
a2_left = s_destroy_left(super2idx, idx2super, 2)
a1_right = s_destroy_right(super2idx, idx2super, 1)
a2_right = s_destroy_right(superidx, idx2super, 2)
"""
step4: define operators in the full hilbert space and compare. 
To compare, I create operators in the usual way in the full hilbert space and I apply them to states that belong to my reduced super space 
and check if the two results match. So I have to understand how to define the states in the vector representation that belong to the reduced super space
"""
