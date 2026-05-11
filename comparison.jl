using QuantumToolbox
using StaticArrays
using SparseArrays
"""
write the basis in number representation, 2 sites with dims 3 (meaning each site can be in state 0,1,2)
"""
dims = (3, 3)
N = length(dims) #number of sites
ket_basis = [SVector{N,Int}(s) for s in Iterators.product(ntuple(i -> 0:dims[i]-1, N)...)]
#print(ket_basis) # output: SVector{2, Int64}[[0, 0] [0, 1] [0, 2]; [1, 0] [1, 1] [1, 2]; [2, 0] [2, 1] [2, 2]]
#creating a dictionary
ket2idx = Dict(s => i for (i,s) in enumerate(ket_basis)) #the state is the key and the index is the value
idx2ket = Dict(i => s for (i,s) in enumerate(ket_basis)) #viceversa
#=
for (i, s) in idx2ket
    println("Dictionary: Index: $i, Ket: $s")
end
for k in ket_basis
    println("Ket basis: $k")
end
=#
"""
step 2 create super space with k = 0,1,-1 
we create two dictionaries: super2idx takes a tuple (state, tilde state) 
(so the key is the state in super number representation)
and gives the index, 
idx2super takes the index and gives the super state
"""
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
"""
step 3 is to build operator in the resticted super space. I did two functions, one for left operators and one for right. 
Essentially they take two dictionaries (state => idx and idx => state) and the site (to create a1, a2 ... for each site)
we create the matrix a_site
"""

function s_destroy_right(super2idx, idx2super, site) #inspired by enr_destroy
    D = length(super2idx) # dimension of reduced hilbert space
    I_list, J_list, V_list = Int[], Int[], ComplexF64[]
    

    for (j, (ket, bra)) in idx2super 
        n_k = bra[site] 
        if n_k > 0                           # same check as enr_destroy: s > 0
            new_bra = setindex(bra, n_k-1, site) #returns a new vector (new_bra) with value nk-1 at index "site"
            new_key = (ket, new_bra)
            if haskey(super2idx, new_key)    # check if the output is in restricted basis. If it is, we fill a matrix element
                i = super2idx[new_key]
                push!(I_list, i)
                push!(J_list, j)
                push!(V_list, sqrt(ComplexF64(n_k)))
            end
        end
    end

    return QuantumObject(sparse(I_list, J_list, V_list, D, D); type=Operator(), dims=(D,))
    
end


function s_destroy_left(super2idx, idx2super, site)
    D = length(super2idx) # dimension of reduced hilbert space
    I_list, J_list, V_list = Int[], Int[], ComplexF64[]
    

    for (j, (ket, bra, _)) in idx2super
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
#operators I defined in the reduced space
a1_left = s_destroy_left(super2idx, idx2super, 1)
a2_left = s_destroy_left(super2idx, idx2super, 2)
a1_right = s_destroy_right(super2idx, idx2super, 1)
a2_right = s_destroy_right(super2idx, idx2super, 2)
N1_left = a1_left' * a1_left
N2_left = a2_left' * a2_left
N1_right = a1_right' * a1_right
N2_right = a2_right' * a2_right
"""
step4: define operators in the full hilbert space and compare. 
To compare, I create operators in the usual way in the full hilbert space and I apply them to states that belong to my reduced super space 
and check if the two results match. So I have to understand how to define the states in the vector representation that belong to the reduced super space.
For comparison I define a function that given a number representation of a super state gives a quantum object. 
Then I apply the builtin operator and the full operator and compare the two results. Once this is done for each element
of the restricted super space, then I can confirm that the builtin operators are defined correctly. The full operators are called left because
they act on the ket, even if the vectorization convention in QuantumToolbox is column stacked, so the tilde space is on the left. So 
they are called left but they act on the right (on the ket). And the full right operators actually act on the tilde space, which is first on the
column stacked vectorization. 
"""

d = prod(dims)  # full ket space dimension = 9

# build a_1 in the full ket space using tensor product
a1_full = tensor(destroy(3), qeye(3))   # 9×9 operator
a2_full = tensor(qeye(3), destroy(3))   # 9×9 operator

# a_1^L in the full super space (81×81), QuantumToolbox gives this directly
a1L_full = spre(a1_full)    # (I ⊗ a_1) in super space (column stacked, it acts on the ket, the tilde space is on the left)
a2L_full = spre(a2_full)    # (I ⊗ a_2 ) in super space

a1R_full = spost(a1_full')   # (I ⊗ ā_1) in super space note it is the adjoint of a1_full (if a acts on the right, it acts on the bra, it will raise the number))
a2R_full = spost(a2_full')   # (I ⊗ ā_2) in super space

N1L_full = spre(a1_full' * a1_full)   # build n1 = a1†a1 first, then spre
N2L_full = spre(a2_full' * a2_full)
#N1L_full = a1L_full' * a1L_full #if I define n1l full in this way, then I get a mismatch
#N2L_full = a2L_full' * a2L_full

#I need to think about number operators on the right
N1R_full = spost(a1_full' * a1_full)
N2R_full = spost(a2_full' * a2_full)

#this function is needed only for full operators, the built in operators are already constructed in the reduced super space 
function s_label_to_vecdm(ket_label, bra_label, dims) #question: do I need to construct the density matrix or is there a more direct way?
    # construct ket state |n⟩ in full space
    ψ_ket = tensor([basis(dims[i], ket_label[i]) for i in 1:length(dims)]...)
    # construct bra state |m⟩ in full space  
    ψ_bra = tensor([basis(dims[i], bra_label[i]) for i in 1:length(dims)]...)
    # density matrix element |n⟩⟨m|
    return mat2vec(ψ_ket * ψ_bra')
end


for i in 1:length(super_basis), j in 1:length(super_basis) #like a nested loop
        
    # get the number representation from the dictionary
    ket_i, bra_i, q_i = idx2super[i] #ket_i and bra_i are in number representation
    ket_j, bra_j, q_j = idx2super[j]
        
        # build the vectorized density matrices in the full space, column stacked!!
    vec_ρi = s_label_to_vecdm(ket_i, bra_i, dims)
    vec_ρj = s_label_to_vecdm(ket_j, bra_j, dims)
        
    # full space matrix element
    me_full_1left  = dot(vec_ρi.data, (a1L_full * vec_ρj).data)
    me_full_2left  = dot(vec_ρi.data, (a2L_full * vec_ρj).data)
    me_full_1right = dot(vec_ρi.data, (a1R_full * vec_ρj).data)
    me_full_2right = dot(vec_ρi.data, (a2R_full * vec_ρj).data)
    me_full_n1left  = dot(vec_ρi.data, (N1L_full * vec_ρj).data)
    me_full_n2left  = dot(vec_ρi.data, (N2L_full * vec_ρj).data)

    #check tomorrow
    me_full_n1right = dot(vec_ρi.data, (N1R_full * vec_ρj).data)
    me_full_n2right = dot(vec_ρi.data, (N2R_full * vec_ρj).data)
       
        
    # compare
    if abs(me_full_1left - a1_left.data[i, j]) > 1e-10
        println("MISMATCH for a1_left (that acts on ket) at ($i,$j)")
    end
    if abs(me_full_2left - a2_left.data[i, j]) > 1e-10
        println("MISMATCH for a2_left at ($i,$j)")
    end
        
    if abs(me_full_1right - a1_right.data[i, j]) > 1e-10
        println("MISMATCH for a1_right at ($i,$j)") 
        break
    end
    
    if abs(me_full_2right - a2_right.data[i, j]) > 1e-10
        println("MISMATCH for a2_right at ($i,$j)") 
        break
    end

    if abs(me_full_n1left - N1_left.data[i, j]) > 1e-10
        println("MISMATCH for N1_left at ($i,$j)")
        println("  full space result:    $me_full_n1left")
        println("  builtin result:   $(N1_left.data[i, j])")
        println("  ket_i=$ket_i, bra_i=$bra_i")
        println("  ket_j=$ket_j, bra_j=$bra_j")
    end
    if abs(me_full_n2left - N2_left.data[i, j]) > 1e-10
        println("MISMATCH for N2_left at ($i,$j)") 
        println("  full space result:    $me_full_n2left")
        println("  builtin result:   $(N2_left.data[i, j])")
        println("  ket_i=$ket_i, bra_i=$bra_i")
        println("  ket_j=$ket_j, bra_j=$bra_j")
    end
    #=
    if abs(me_full_n1right - N1_right.data[i, j]) > 1e-10
        println("MISMATCH for N1_right at ($i,$j): full=$me_full_n1right, builtin=$N1_right.data[i, j]") 
        break
    end
    if abs(me_full_n2right - N2_right.data[i, j]) > 1e-10
        println("MISMATCH for N2_right at ($i,$j): full=$me_full_n2right, builtin=$N2_right.data[i, j]") 
        break   
    end
    =#
        
end
