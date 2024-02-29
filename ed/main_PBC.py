from quspin.operators import hamiltonian, quantum_operator, exp_op # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
np.object = np.object_
import sys, os
from concurrent import futures
import matplotlib.pyplot as plt

def set_hamiltonian(L, J, D, K, H, basis):
    ### define operators with PBC using site-coupling lists
    Jzz = [[J, i, (i+1)%L] for i in range(L)]
    Jxx_yy = [[J/2, i, (i+1)%L] for i in range(L)]
    Dzx = [[D/2, i, (i+1)%L] for i in range(L)]
    Dxz = [[-D/2, i, (i+1)%L] for i in range(L)]
    Kpp_mm = [[-K/4, i, i] for i in range(L)]
    Kpm_mp = [[K/4, i, i] for i in range(L)]
    Hz = [[-H, i] for i in range(L)]

    ### static and dynamic lists
    static = [
        ["+-", Jxx_yy], ["-+", Jxx_yy], ["zz", Jzz], # Heisenberg term
        ["z+", Dzx], ["z-", Dzx], ["+z", Dxz], ["-z", Dxz], # DM term
        ["++", Kpp_mm], ["--", Kpp_mm], ["+-", Kpm_mp], ["-+", Kpm_mp], # anisotropic term
        ["z", Hz] # Zeeman term
    ]
    dynamic = []

    ### compute the Hamiltonian
    return hamiltonian(static, dynamic, basis=basis, dtype=np.complex128, check_symm=False, check_herm=False)

def calc_magnetization(L, basis, ψ, fluc=True):
    Sz = [[1 / L, i] for i in range(L)]
    Sz_op = quantum_operator(
        dict(Sz=[["z", Sz]]),
        basis=basis, dtype=np.complex128, check_symm=False, check_herm=False
    )
    Mz = Sz_op.matrix_ele(ψ, ψ, pars=dict(Sz=1.0))

    if not fluc:
        if np.abs(np.imag(Mz)) > 1E-10:
            print("!!! imaginary part is too large: Im(Mz) =", np.imag(Mz), "!!!")
        return np.real(Mz)

    ΔMz = np.sqrt(Sz_op.quant_fluct(ψ, pars=dict(Sz=1.0)))

    if np.abs(np.imag(Mz)) > 1E-10 or np.abs(np.imag(ΔMz)) > 1E-10:
        print("!!! imaginary part is too large: Im(Mz) =", np.imag(Mz), ", Im(ΔMz) =", np.imag(ΔMz), "!!!")

    return np.real(Mz), np.real(ΔMz)

def calc_magnetization_AFM(L, basis, ψ, fluc=True):
    SzAFM = [[(-1)**i / L, i] for i in range(L)]
    SzAFM_op = quantum_operator(
        dict(SzAFM=[["z", SzAFM]]),
        basis=basis, dtype=np.complex128, check_symm=False, check_herm=False
    )
    MzAFM = SzAFM_op.matrix_ele(ψ, ψ, pars=dict(SzAFM=1.0))

    if not fluc:
        if np.abs(np.imag(MzAFM)) > 1E-10:
            print("!!! imaginary part is too large: Im(MzAFM) =", np.imag(MzAFM), "!!!")
        return np.real(MzAFM)

    ΔMzAFM = np.sqrt(SzAFM_op.quant_fluct(ψ, pars=dict(SzAFM=1.0)))

    if np.abs(np.imag(MzAFM)) > 1E-10 or np.abs(np.imag(ΔMzAFM)) > 1E-10:
        print("!!! imaginary part is too large: Im(MzAFM) =", np.imag(MzAFM), ", Im(ΔMzAFM) =", np.imag(ΔMzAFM), "!!!")

    return np.real(MzAFM), np.real(ΔMzAFM)

def calc_soliton_num(twoS, L, J, basis, ψ, fluc=True):
    ### soliton number
    factor = (2*np.pi * (twoS/2)**2)
    C = [[0.5/factor, i, (i+1)%L] for i in range(L)]
    D = [[0.5/factor, i, (i+2)%L] for i in range(L)]
    Ns_op = quantum_operator(
        dict(
            Czp=[["z+", C]], Czm=[["z-", C]], Cpz=[["+z", C]], Cmz=[["-z", C]],
            Dzp=[["z+", D]], Dzm=[["z-", D]], Dpz=[["+z", D]], Dmz=[["-z", D]]
        ),
        basis=basis, dtype=np.complex128, check_symm=False, check_herm=False
    )

    if J < 0:
        pars = dict(
            Czp=-1, Czm=-1, Cpz=1, Cmz=1,
            Dzp=0, Dzm=0, Dpz=0, Dmz=0
        )
    elif J >= 0:
        pars = dict(
            Czp=-0.5, Czm=-0.5, Cpz=0.5, Cmz=0.5,
            Dzp=0.25, Dzm=0.25, Dpz=-0.25, Dmz=-0.25
        )

    Ns = Ns_op.matrix_ele(ψ, ψ, pars=pars)

    if not fluc:
        if np.abs(np.imag(Ns)) > 1E-10:
            print("!!! imaginary part is too large: Im(Ns) =", np.imag(Ns), "!!!")
        return np.real(Ns)

    ### fluctuation of soliton number
    ΔNs = np.sqrt(Ns_op.quant_fluct(ψ, pars=pars))

    if np.abs(np.imag(Ns)) > 1E-10 or np.abs(np.imag(ΔNs)) > 1E-10:
        print("!!! imaginary part is too large: Im(Ns) =", np.imag(Ns), ", Im(ΔNs) =", np.imag(ΔNs), "!!!")

    return np.real(Ns), np.real(ΔNs)

def oneshot(twoS, L, J, D, K, H, bases, verbose=False):
    ### solve eigenvalue problem for each crystal momentum
    E_list = []
    ψ_list = []
    E_min = 1E10
    for ik in range(L):
        # compute Hamiltonian
        hamil = set_hamiltonian(L, J, D, K, H, bases[ik])

        # calculate two lowest minimum energies
        E, ψ = hamil.eigsh(k=2, which="SA") #, maxiter=1E4, tol=1E-12
        ind = np.argsort(E)
        E = E[ind].tolist(); ψ = ψ[:, ind]
        E_list.append(E)
        ψ_list.append(ψ)

        # save the minimum energy for all crystal momentum
        if E[0] < E_min:
            E_min = E[0]
            ik_min = ik
            basis_min = bases[ik]

    if verbose:
        # print(E_list)
        print("GS energy:", E_min, "with k = 2π *", ik_min, "/", L)

    ### calculate magnetization and its fluctuation for GS
    Mz, ΔMz = calc_magnetization(L, basis_min, ψ_list[ik_min][:, 0])

    ### calculate AFM magnetization and its fluctuation for GS
    MzAFM, ΔMzAFM = calc_magnetization_AFM(L, basis_min, ψ_list[ik_min][:, 0])

    ### calculate number of solitons and its fluctuation for GS
    Ns, ΔNs = calc_soliton_num(twoS, L, J, basis_min, ψ_list[ik_min][:, 0])

    ### calculate expectation value of exp(iπSz)
    Sz_op = quantum_operator(
        dict(Sz=[["z", [[1, i] for i in range(L)]]]),
        basis=basis_min, dtype=np.complex128, check_symm=False, check_herm=False
    )
    cisπSz_op = exp_op(Sz_op, a=1j*np.pi)
    cisπSz_ψ = cisπSz_op.dot(ψ_list[ik_min][:, 0])
    ex_cisπSz = np.vdot(ψ_list[ik_min][:, 0], cisπSz_ψ)

    if np.abs(np.imag(ex_cisπSz)) > 1E-10:
        print("!!! imaginary part is too large: Im(exp(iπSz)) =", np.imag(ex_cisπSz), "!!!")
    print("2S =", twoS, "| L =", L, "| J =", J, "| D =", D, "| K =", K, "| H =", H, flush=True)
    print(ik_min, Mz, ΔMz, MzAFM, ΔMzAFM, Ns, ΔNs, flush=True)

    return ik_min, Mz, ΔMz, MzAFM, ΔMzAFM, Ns, ΔNs, ex_cisπSz, E_list, ψ_list

def wrapper_oneshot(args):
    return oneshot(*args)

def kresolve(twoS, L, J, D, K, H, bases, nE):
    print("2S =", twoS, "| L =", L, "| J =", J, "| D =", D, "| K =", K, "| H =", H)

    ### solve eigenvalue problem for each crystal momentum
    E_list = []
    Mz_list = np.zeros((L, nE))
    Ns_list = np.zeros((L, nE))
    for ik in range(L):
        print("########## ik =", ik, "##########")
        # compute Hamiltonian
        hamil = set_hamiltonian(L, J, D, K, H, bases[ik])

        # calculate nE lowest minimum energies
        E, ψ = hamil.eigsh(k=nE, which="SA", maxiter=1E4) # tol=1E-12
        ind = np.argsort(E)
        E = E[ind].tolist(); ψ = ψ[:, ind]
        E_list.append(E)

        for iE in range(nE):
            ### calculate expectation value of Sz
            Mz = calc_magnetization(L, bases[ik], ψ[:, iE], fluc=False)
            Mz_list[ik, iE] = Mz

            ### calculate soliton number
            Ns = calc_soliton_num(twoS, L, J, bases[ik], ψ[:, iE], fluc=False)
            Ns_list[ik, iE] = Ns

            print("iE =", iE, ": Mz =", Mz, ", Ns =", Ns)
        print("")

    E_list = np.roll(np.array(E_list), L//2, axis=0)
    Mz_list = np.roll(Mz_list, L//2, axis=0)
    Ns_list = np.roll(Ns_list, L//2, axis=0)

    return E_list, Mz_list, Ns_list

if __name__ == "__main__":
    ##### define model parameters #####
    mode = sys.argv[1]
    twoS = int(sys.argv[2]) # spin
    L = int(sys.argv[3]) # system size
    J = float(sys.argv[4]) # interaction
    D = float(sys.argv[5]) # DM
    K = float(sys.argv[6]) # anisotropy

    if twoS % 2 == 0:
        S = str(twoS//2)
    else:
        S = str(twoS) + "/2"

    # compute spin-S basis
    bases = [spin_basis_1d(L, S=S, pauli=0, kblock=ik) for ik in range(L)]

    if mode == "oneshot":
        H = float(sys.argv[7]) # z external field

        ik_min, Mz, ΔMz, MzAFM, ΔMzAFM, Ns, ΔNs, ex_cisπSz, E_list, ψ_list = oneshot(twoS, L, J, D, K, H, bases, verbose=True)
        print("Mz =", Mz); print("ΔMz =", ΔMz)
        print("MzAFM =", MzAFM); print("ΔMzAFM =", ΔMzAFM)
        print("Ns =", Ns); print("ΔNs =", ΔNs)
        print("exp(iπSz) =", ex_cisπSz)
        print("energy:", np.sort(np.ravel(np.array(E_list))))
        print("energy:", E_list)

    elif mode == "sweep":
        ### z external field
        H1 = float(sys.argv[7])
        dH = float(sys.argv[8])
        H2 = float(sys.argv[9])
        nH = round((H2-H1)/dH)+1
        print("sweep magnetic field from", H1, "to", H2, "with", nH, "points")

        ### sweep magnetic field using multithreading
        print("calculation start")
        params = [[twoS, L, J, D, K, H1+dH*i, bases] for i in range(nH)]
        with futures.ThreadPoolExecutor(max_workers=int(sys.argv[10])) as executor:
            results = executor.map(wrapper_oneshot, params)
        print("calculation stop")
        results = list(results)

        ### create data folders
        if not os.path.exists("data_PBC"):
            os.mkdir("data_PBC")
        if not os.path.exists("data_PBC/twoS{}".format(twoS)):
            os.mkdir("data_PBC/twoS{}".format(twoS))
        if not os.path.exists("images_PBC"):
            os.mkdir("images_PBC")

        ### output data
        filename = "twoS{}_L{}_J{:.3f}_D{:.3f}_K{:.3f}".format(twoS, L, J, D, K)
        fout1 = open("data_PBC/twoS{}/quant_".format(twoS) + filename + ".txt", "wt")
        fout2 = open("data_PBC/twoS{}/energy_".format(twoS) + filename + ".txt", "wt")

        k_list = [2*np.pi*j / L for j in range(L)]
        H_list = [H1+dH*i for i in range(nH)]
        Mzs = []; ΔMzs = []
        MzAFMs = []; ΔMzAFMs = []
        Nss = []; ΔNss = []
        E_array = []
        min_k_list = []
        for i in range(nH):
            ik_min, Mz, ΔMz, MzAFM, ΔMzAFM, Ns, ΔNs, ex_cisπSz, E_list, ψ_list = results[i]
            Mzs.append(Mz); ΔMzs.append(ΔMz)
            MzAFMs.append(MzAFM); ΔMzAFMs.append(ΔMzAFM)
            Nss.append(Ns); ΔNss.append(ΔNs)
            E_array.append(E_list)
            min_k_list.append(k_list[ik_min])

            print("{:.4f}".format(H_list[i]), ik_min, Mz, ΔMz, MzAFM, ΔMzAFM, Ns, ΔNs, ex_cisπSz, file=fout1, end="\n")

            print("{:.4f}".format(H_list[i]), file=fout2, end="")
            for j in range(L):
                print("", j, E_list[j][0], E_list[j][1], file=fout2, end="")
            print("", file=fout2, end="\n")
        E_array = np.array(E_array)

        fout1.close()
        fout2.close()

        # ### output figure
        # fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(6, 9), constrained_layout=True)
        # # fig.subplots_adjust(left=0.2, right=0.85, top=0.85, bottom=0.1)

        # # magnetization
        # ax0.set_ylabel("$M_z$")
        # ax0.grid(linestyle='dotted', linewidth=0.5)
        # cplt = ax0.scatter(
        #     H_list, Mzs, c=min_k_list,
        #     cmap="rainbow", vmin=0, vmax=2*np.pi, s=5
        # )
        # cbar = fig.colorbar(cplt, ax=ax0, ticks=[0, np.pi, 2*np.pi])
        # cbar.ax.set_yticklabels(["0", "$\pi$", "$2\pi$"])

        # # magnetic fluctuation
        # ax1.set_ylabel("$\Delta M_z$")
        # ax1.grid(linestyle='dotted', linewidth=0.5)
        # ax1.scatter(
        #     H_list, ΔMzs, c=min_k_list,
        #     cmap="rainbow", vmin=0, vmax=2*np.pi, s=5
        # )

        # # AFM magnetization
        # ax2.set_ylabel(r"$\widetilde{M}_z$")
        # ax2.grid(linestyle="dotted", linewidth=0.5)
        # ax2.scatter(
        #     H_list, Nss, c=min_k_list,
        #     cmap="rainbow", vmin=0, vmax=2*np.pi, s=5
        # )

        # # AFM magnetic fluctuation
        # ax3.set_xlabel("$H$")
        # ax3.set_ylabel(r"$\Delta\widetilde{M}_z$")
        # ax3.grid(linestyle="dotted", linewidth=0.5)
        # ax3.scatter(
        #     H_list, ΔNss, c=min_k_list,
        #     cmap="rainbow", vmin=0, vmax=2*np.pi, s=5
        # )

        # # # energy eigenvalues
        # # ax2.set_xlabel("$H$")
        # # ax2.set_ylabel("$E$")
        # # ax2.grid(linestyle="dotted", linewidth=0.5)
        # # for j in range(L):
        # #     for k in range(2):
        # #         ax2.scatter(
        # #             H_list, E_array[:, j, k], c=np.full(nH, k_list[j]),
        # #             cmap="rainbow", vmin=0, vmax=2*np.pi, s=0.5
        # #         )

        # fig.savefig("images_PBC/quant_" + filename + ".png")

    elif mode == "kresolve":
        H = float(sys.argv[7]) # z external field
        nE = int(sys.argv[8]) # number of eigenvalues

        E_list, Mz_list, Ns_list = kresolve(twoS, L, J, D, K, H, bases, nE)
        Ns_max = max(np.abs([np.min(Ns_list), np.max(Ns_list)]))
        if L % 2 == 0:
            k_list = np.linspace(-np.pi, np.pi, L, endpoint=False)
        else:
            k_list = np.linspace(-np.pi * (1 - 1/L), np.pi * (1 - 1/L), L, endpoint=True)

        # test: gap from the GS energy
        test = np.partition(E_list.ravel(), 1)
        print("gap:", test[1] - test[0])

        ### create data folders
        if not os.path.exists("images_PBC"):
            os.mkdir("images_PBC")

        ### output figure
        # determine marker size
        with open("data_PBC/soliton_num_norm.txt", "r") as f:
            if J < 0:
                col = 2
            elif J > 0:
                col = 3
            lines = f.read().splitlines()
            Ns1 = float(lines[4*(L-8)+1].split()[col])
        sigma = 0.05
        marker_size = 20 * np.exp(- (Ns_list - Ns1)**2 / (2*sigma**2)) + 1

        filename = "twoS{}_L{}_J{:.3f}_D{:.3f}_K{:.3f}_H{:.4f}".format(twoS, L, J, D, K, H)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ax.set_xlabel("$k$")
        ax.set_ylabel("$E$")
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(["$-\pi$", "0", "$\pi$"])
        ax.set_xlim((-np.pi-0.2, np.pi+0.2))
        ax.grid(linestyle="dotted", linewidth=0.5)
        for iE in range(nE):
            cplt = ax.scatter(
                k_list, E_list[:, iE], c=Ns_list[:, iE],
                cmap="jet", vmin=0, vmax=Ns_max, s=marker_size[:, iE]
            )
            if L % 2 == 0:
                ax.scatter(
                    [np.pi], [E_list[0, iE]], c=[Ns_list[0, iE]],
                    cmap="jet", vmin=0, vmax=Ns_max, s=marker_size[0, iE]
                )
        cbar = fig.colorbar(cplt, ax=ax, label="$N_{\mathrm{s}}$")
        # cbar.ax.set_yticklabels(["0", "$\pi$", "$2\pi$"])

        plt.savefig("images_PBC/twoS" + str(twoS) + "/kresolve_" + filename + ".png")

    else:
        print('mode is "oneshot", "sweep", or "kresolve"')
