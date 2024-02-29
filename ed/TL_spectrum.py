from quspin.operators import hamiltonian, quantum_operator, exp_op # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
np.object = np.object_
import sys, os
import matplotlib.pyplot as plt

def set_hamiltonian(L, J, Delta, H, basis):
    ### define operators with PBC using site-coupling lists
    Jzz = [[J*Delta, i, (i+1)%L] for i in range(L)]
    Jxx_yy = [[J/2, i, (i+1)%L] for i in range(L)]
    Hz = [[-H, i] for i in range(L)]

    ### static and dynamic lists
    static = [
        ["+-", Jxx_yy], ["-+", Jxx_yy], ["zz", Jzz], # Heisenberg term
        ["z", Hz] # Zeeman term
    ]
    dynamic = []

    ### compute the Hamiltonian
    return hamiltonian(static, dynamic, basis=basis, dtype=np.complex128, check_symm=False, check_herm=False, check_pcon=False)

def spectrum(twoS, L, J, Delta, H, bases, nE):
    print("2S =", twoS, "| L =", L, "| J =", J, "| Δ =", Delta, "| H =", H)

    ### solve eigenvalue problem for each crystal momentum
    E_array = np.full((twoS*L+1, L, nE), 1E16)
    for Nup in range(twoS*L+1):
        print("########## Nup =", Nup, "##########")
        for ik in range(L):
            print("##### ik =", ik, "#####")

            # compute Hamiltonian
            hamil = set_hamiltonian(L, J, Delta, H, bases[L*Nup+ik])

            # calculate nE lowest minimum energies
            nB = bases[L*Nup+ik].Ns
            if nB < 1:
                continue
            elif nB < 3:
                E, ψ = hamil.eigh()
                print(E)
                E_array[Nup, ik, 0:nB] = E
            else:
                E, ψ = hamil.eigsh(k=min(nE, nB-2), which="SA", maxiter=1E4) # tol=1E-12
                ind = np.argsort(E)
                print(E[ind])
                E_array[Nup, ik, 0:min(nE, nB-2)] = E[ind]

            print("")

    return E_array

if __name__ == "__main__":
    ##### define model parameters #####
    twoS = int(sys.argv[1]) # spin
    L = int(sys.argv[2]) # system size
    J = float(sys.argv[3]) # interaction
    Delta = float(sys.argv[4]) # anisotropy for XXZ
    H = float(sys.argv[5]) # z external field
    nE = int(sys.argv[6]) # number of eigenvalues

    if twoS % 2 == 0:
        S = str(twoS//2)
    else:
        S = str(twoS) + "/2"

    # compute spin-S basis
    bases = [
        spin_basis_1d(L, S=S, Nup=Nup, pauli=0, kblock=ik)
        for Nup in range(twoS*L+1) for ik in range(L)
    ]

    # spectrum
    E_array = spectrum(twoS, L, J, Delta, H, bases, nE)
    E_min_ind = np.unravel_index(np.argmin(E_array), E_array.shape)
    E_min = E_array[E_min_ind]
    E_array = np.roll(E_array, -E_min_ind[1], axis=1)

    ### create data folders
    if not os.path.exists("images_PBC"):
        os.mkdir("images_PBC")
    if not os.path.exists("images_PBC/twoS" + str(twoS)):
        os.mkdir("images_PBC/twoS" + str(twoS))

    ### output figure
    filename = "twoS{}_L{}_J{:.3f}_Delta{:.3f}_H{:.4f}".format(twoS, L, J, Delta, H)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xlim((-0.2, 2*np.pi+0.2))
    ax.set_xticklabels(["0", "$\pi$", "$2\pi$"])
    ax.set_ylim((-0.05, 2.0*twoS))

    plt.tick_params(labelsize=14)
    ax.set_xlabel("$Δk$", fontsize=20)
    ax.set_ylabel("$ΔE / J$", fontsize=20)
    ax.grid(linestyle="dotted", linewidth=0.5)

    k_list = np.linspace(0, 2*np.pi, L, endpoint=False)
    for iE in range(nE):
        cplt = ax.scatter(
            k_list, E_array[E_min_ind[0], :, iE] - E_min, c="blue", s=20
        )

    if E_min_ind[0] < twoS*L:
        for iE in range(nE):
            cplt = ax.scatter(
                k_list, E_array[E_min_ind[0]+1, :, iE] - E_min, c="red", s=12
            )

    # if E_min_ind[0] % 2 == 0:
    #     color=["blue", "red"]
    # else:
    #     color=["red", "blue"]
    # for Nup in range(twoS*L+1):
    #     p_Nup = Nup % 2
    #     for iE in range(nE):
    #         cplt = ax.scatter(
    #             k_list, E_array[Nup, :, iE] - E_min, c=color[p_Nup], s=4
    #         )

    plt.savefig("images_PBC/twoS" + str(twoS) + "/spectrum_" + filename + ".png")
