# Abstract
The numerical simulation of the Lindblad master equation is numerical demanding, with the cost scaling exponentially with the system size. 
The things are different when the system possesses a U(1) symmetry: in this case the vectorized Liouvillian is block-diagonal in the number basis. 
This allows to restrict computations to a subset of those sectors, significantly reducing the effective dimension of the problem. 
This report describes the design and implementation of `SuperEnrSpace`, a Julia module extending `QuantumToolbox.jl` that exploits this symmetry. 
Working in the Fock-Liouville space, the module provides a block-ordered restricted superoperator basis, that is sufficient to compute the spectrum and analyse dynamics of Liouvillians with U(1) symmetry. 
In the end, the implementation is successfully validated on a Kerr non-linear oscillator.
