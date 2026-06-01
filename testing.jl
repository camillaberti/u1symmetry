using Test
using QuantumToolbox
include("enlargedspace.jl")
using .SuperEnrSpace

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

#this function is needed only for full operators, the built in operators are already constructed in the reduced super space 
function s_label_to_vecdm(ket_label, bra_label, dims) 
    # construct ket state |n⟩ in full space
    ψ_ket = tensor([basis(dims[i], ket_label[i]) for i in 1:length(dims)]...)
    # construct bra state |m⟩ in full space  
    ψ_bra = tensor([basis(dims[i], bra_label[i]) for i in 1:length(dims)]...)
    # density matrix element |n⟩⟨m|
    return mat2vec(ψ_ket * ψ_bra')
end

@testset "Enlarged Space Operators Verification" begin
    # Setup
    dims = (3, 3, 3)
    n_exc = 1
    p = 2
    cutoff = dims[1]
    space = s_enrspace(dims, n_exc, p)

    # Build Reduced Operators

    results = [s_destroy(space, i) for i in 1:3]
    a_left = [r[1] for r in results]
    a_right = [r[2] for r in results]
    N_left = [a_left[i]' * a_left[i] for i in 1:3]
    N_right = [a_right[i]' * a_right[i] for i in 1:3]
    a2_left = [a_left[i]^2 for i in 1:3]
    a2_right = [a_right[i]^2 for i in 1:3]

    # Build Full Operators
    a_full = [tensor([i==j ? destroy(cutoff) : qeye(cutoff) for j in 1:3]...) for i in 1:3]
    a2_full = [op^2 for op in a_full]
    aL_full = [spre(op) for op in a_full]
    aR_full = [spost(op') for op in a_full]
    NL_full = [spre(op' * op) for op in a_full]
    NR_full = [spost(op' * op) for op in a_full]
    a2L_full = [spre(op)    for op in a2_full]
    a2R_full = [spost(op')  for op in a2_full]

    target_indices = space.target_indices
    idx2super = space.idx2state

    @testset "Matrix Element Comparison" begin
        for i in target_indices, j in target_indices
            ket_i, bra_i = idx2super[i]
            ket_j, bra_j = idx2super[j]
            
            vec_ρi = s_label_to_vecdm(ket_i, bra_i, dims)
            vec_ρj = s_label_to_vecdm(ket_j, bra_j, dims)

            for site in 1:3
                # Test Annihilation Left
                me_full = dot(vec_ρi.data, (aL_full[site] * vec_ρj).data)
                @test me_full ≈ a_left[site].data[i, j] atol=1e-10

                # Test Annihilation Right
                me_full_r = dot(vec_ρi.data, (aR_full[site] * vec_ρj).data)
                @test me_full_r ≈ a_right[site].data[i, j] atol=1e-10

                # Test Number Left
                me_n_full = dot(vec_ρi.data, (NL_full[site] * vec_ρj).data)
                @test me_n_full ≈ N_left[site].data[i, j] atol=1e-10

                # Test Number Right
                me_nr_full = dot(vec_ρi.data, (NR_full[site] * vec_ρj).data)
                @test me_nr_full ≈ N_right[site].data[i, j] atol=1e-10

                 # Test a²_left
                me_a2L = dot(vec_ρi.data, (a2L_full[site] * vec_ρj).data)
                @test me_a2L ≈ a2_left[site].data[i, j] atol=1e-10

                # Test a²_right
                me_a2R = dot(vec_ρi.data, (a2R_full[site] * vec_ρj).data)
                @test me_a2R ≈ a2_right[site].data[i, j] atol=1e-10

                # Test (a²)†a² left
                LdagL2_full = spre(a2_full[site]' * a2_full[site])
                me_LdagL2L = dot(vec_ρi.data, (LdagL2_full * vec_ρj).data)

                # composition in restricted space: (a²_left)' * a²_left
                LdagL2_left_comp = a2_left[site]' * a2_left[site]
                @test me_LdagL2L ≈ LdagL2_left_comp.data[i, j] atol=1e-10

                # Test (a²)†a² right
                LdagL2_full_R = spost(a2_full[site]' * a2_full[site])
                me_LdagL2R = dot(vec_ρi.data, (LdagL2_full_R * vec_ρj).data)

                LdagL2_right_comp = a2_right[site]' * a2_right[site]
                @test me_LdagL2R ≈ LdagL2_right_comp.data[i, j] atol=1e-10

            end
        end
    end
end