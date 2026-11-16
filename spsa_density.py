import pennylane as qp
from pennylane import qchem
from pennylane import numpy as pnp
import time
import matplotlib.pyplot as plt
from zne_density import run_zne_vqe
from zne_density import calcThetaOpt
from monitor import CentrallyMonitoredCost
import numpy as np
from symmetry import SymmetryCF


#######
# SPECIFYING OPTIMIZER & INITIAL ANSATZ PARAMS
#######


if __name__ == "__main__":
    checkpoint0 = time.perf_counter()

    #######
    # MOLECULE CONFIG
    #######

    chain_length = 2
    spacing = 1.3228
    basis_set = "sto-3g"
    electrons = chain_length
    noise = 0.001
    symmetry = True

    coordinates = pnp.array([], requires_grad=True)
    symbols = []

    for i in range(chain_length):
        symbols.append("H")
        coordinates = pnp.append(coordinates, pnp.array([[0.0, 0.0, i*spacing]]))



    #######
    # GENERATE PENNYLANE MOLECULE OBJECT
    #######

    molecule = qchem.Molecule(symbols, np.array(coordinates, dtype=float).reshape(-1, 3), basis_name=basis_set)


    #######
    # GENERATE HAMILTONIAN & HARTREE-FOCK STATE USING 1- AND 2- ELECTRON INTEGRALS
    #######

    H, qubits = qchem.molecular_hamiltonian(molecule)
    hf = qchem.hf_state(electrons, qubits)

    hf_index = int("".join(str(int(b)) for b in hf), 2)

    # 4. Create the 2^N state vector
    hf_statevector = np.zeros(2**qubits, dtype=complex)
    hf_statevector[hf_index] = 1.0


    #######
    # GET EXACT GROUND STATE
    #######


    H_matrix = qp.matrix(H)

    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

    fci_energy = eigenvalues[0]
    fci_qubit_state = eigenvectors[:, 0]

    print(f"Exact Qubit FCI Energy: {fci_energy:.8f} Ha")


    #####
    # CALC HAMILTONIAN SCALE
    #####

    coeffs, ops = H.terms()

    # Sum absolute weights of all non-identity terms
    scale = sum(abs(c) for c, op in zip(coeffs, ops) if not isinstance(op, qp.Identity))

    print(f"Observable scale: {float(scale):.4f}")


    #######
    # GENERATE ALL SINGLE AND DOUBLE EXCITATIONS
    #######

    singles, doubles = qchem.excitations(electrons=electrons, orbitals=qubits)
    avg_wires_per_gate = (len(singles) + 2*len(doubles)) / len(singles + doubles)

    num_theta = len(singles) + len(doubles)
    print(f"Singles: {len(singles)} Doubles: {len(doubles)}")


    #######
    # ANSATZ CIRCUIT
    #######
    dev = qp.device("default.mixed", wires=qubits)
    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit_VQE_decomposed(theta, l=1.0):
        # 1. State preparation (HF state)
        qp.BasisState(hf, wires=range(qubits))

        # Under the hood, AllSinglesDoubles applies doubles first, then singles
        num_doubles = len(doubles)

        # 2. Apply Double Excitations
        for i, d_gate in enumerate(doubles):
            qp.DoubleExcitation(theta[i], wires=d_gate)
            # Apply stochastic bit flip on each wire involved in the gate
            for wire in d_gate:
                qp.BitFlip(noise*l, wires=wire)

        # 3. Apply Single Excitations
        for i, s_gate in enumerate(singles):
            theta_index = num_doubles + i
            qp.SingleExcitation(theta[theta_index], wires=s_gate)
            # Apply stochastic bit flip on each wire involved in the gate
            for wire in s_gate:
                qp.BitFlip(noise*l, wires=wire)

        return qp.expval(H)


    dev1 = qp.device("lightning.qubit", wires=qubits)
    @qp.qnode(dev1, diff_method="parameter-shift")
    def circuit_VQE_decomposed_nonoise(theta, l=1.0):
        # 1. State preparation (HF state)
        qp.BasisState(hf, wires=range(qubits))

        # Under the hood, AllSinglesDoubles applies doubles first, then singles
        num_doubles = len(doubles)

        # 2. Apply Double Excitations
        for i, d_gate in enumerate(doubles):
            qp.DoubleExcitation(theta[i], wires=d_gate)
    

        # 3. Apply Single Excitations
        for i, s_gate in enumerate(singles):
            theta_index = num_doubles + i
            qp.SingleExcitation(theta[theta_index], wires=s_gate)

        return qp.expval(H)


    #######
    # STATE RETRIEVER
    #######
    dev_state = qp.device("lightning.qubit", wires=qubits)
    @qp.qnode(dev_state)
    def get_vqe_state(theta):
        # 1. State preparation (HF state)
        qp.BasisState(hf, wires=range(qubits))

        # Under the hood, AllSinglesDoubles applies doubles first, then singles
        num_doubles = len(doubles)

        # 2. Apply Double Excitations
        for i, d_gate in enumerate(doubles):
            qp.DoubleExcitation(theta[i], wires=d_gate)
    

        # 3. Apply Single Excitations
        for i, s_gate in enumerate(singles):
            theta_index = num_doubles + i
            qp.SingleExcitation(theta[theta_index], wires=s_gate)

        return qp.state()  # Returns the exact 16-dimensional state vector

    

    print("\n🔥 Running Final VQE Parameter Optimization...")
    params = pnp.zeros(num_theta, requires_grad=True)

    if noise != 0:
        if symmetry:
            cost = SymmetryCF(molecule, hf, qubits, doubles, singles, noise, electrons)
        else:
            cost = CentrallyMonitoredCost(circuit_VQE_decomposed)
        # Run ZNE when noise is active
        # Watch out: ensure max(scale_factors) * noise <= 1.0 so probability stays valid!
        scale_factors = [1.0, 2.0, 3.0]
    
        theta_opt, energy_VQE, zne_data = run_zne_vqe(
            cost,
            params,
            scale,
            noise,
            avg_wires_per_gate,
            scale_factors=scale_factors,
            method="poly", 
            poly_degree=1,            # 1 = linear fit (most resilient to shot/trajectory variance)
        )
        
        # Extract the base l=1.0 trajectory history for plotting your standard VQE curve
        base_hist = zne_data["histories"][scale_factors[0]]
        energies = base_hist["energies"]
        polyak_energies = base_hist["polyak_energies"]
        energy_unmitigated = base_hist["energy_VQE"]
        shots_unmitigated = base_hist["shots"]
        
        #print(f"\n📊 ZNE Summary across scales {scale_factors}: {[round(e, 5) for e in zne_data['energies_per_scale']]}")
        #print(f"Unmitigated VQE Energy (l=1): {energy_unmitigated:.5f} Ha")
        #print(f"ZNE Mitigated Energy (l->0):  {energy_VQE:.5f} Ha")
    else:
        # Standard noiseless VQE when noise == 0
        if symmetry:
            cost = SymmetryCF(molecule, hf, qubits, doubles, singles, noise, electrons)
        else:
            cost = CentrallyMonitoredCost(circuit_VQE_decomposed_nonoise)
        theta_opt, energies, polyak_energies, energy_VQE = calcThetaOpt(cost, params, scale, noise, avg_wires_per_gate)


    #######
    # STATE FIDELITY CALCULATION
    #######

    vqe_state = get_vqe_state(theta_opt)

    state_fidelity = np.abs(np.dot(np.conj(vqe_state), fci_qubit_state)) ** 2

    hf_state_fidelity = np.abs(np.dot(np.conj(hf_statevector), fci_qubit_state)) ** 2


    #######
    # RESULTS
    #######
    print("\n")

    print("---Relevant stuff for excel---")
    print(f"Chain length = {chain_length}, Noise = {noise}\n")

    GROUND_TRUTH = [0, 0, -1.1361891624019838, 0, -2.106992740303056, 0, -3.077987174976868]
    HF = [0, 0, -1.1120756981592153, 0, -1.9943386943769075]
    print(f"Ground truth: {GROUND_TRUTH[chain_length]:.5f}")
    print(f"HF energy: {HF[chain_length]:.5f}  |  Diff = {abs(GROUND_TRUTH[chain_length] - HF[chain_length]):.5f}  |  Accurate? = {bool(0.0016 > abs(GROUND_TRUTH[chain_length] - HF[chain_length]))}")

    print("\n**Final energy (Ha) and Final energy  absolute error (Ha) tables:")
    if noise != 0:
        print(f"ZNE VQE energy: {energy_VQE:.5f}  |  Diff = {abs(GROUND_TRUTH[chain_length] - energy_VQE):.5f}  |  Accurate? = {bool(0.0016 > abs(GROUND_TRUTH[chain_length] - energy_VQE))}")
        print(f"Unmitgated VQE energy: {energy_unmitigated:.5f}  |  Diff = {abs(GROUND_TRUTH[chain_length] - energy_unmitigated):.5f}  |  Accurate? = {bool(0.0016 > abs(GROUND_TRUTH[chain_length] - energy_unmitigated))}")
    else:
        print(f"Unmitgated VQE energy: {energy_VQE:.5f}  |  Diff = {abs(GROUND_TRUTH[chain_length] - energy_VQE):.5f}  |  Accurate? = {bool(0.0016 > abs(GROUND_TRUTH[chain_length] - energy_VQE))}")

    print("\n**Final state ground truth fidelity table:")
    print(f"HF state fidelity: {hf_state_fidelity:.6f}")
    print(f"VQE state fidelity: {state_fidelity:.6f}")

    print("\n**Total shots table:")
    if noise != 0:
        print(f"Shots without ZNE = {shots_unmitigated}")
    if symmetry:
        print(f"Symmetry rejected shots = {cost.rejected_shots}")
    print(f"Total shots = {cost.total_shots_taken}")

   
    print()


    #######
    # GRAPH
    #######
    plt.plot(polyak_energies, marker='o', linestyle='-', color='r', label='Polyak')
    plt.plot(energies, marker='o', linestyle='-', color='b', label='My Data')

    plt.axhline(y=GROUND_TRUTH[chain_length], color='r', linestyle='-', label=f'Ground truth (y={GROUND_TRUTH[chain_length]})')
    plt.axhline(y=GROUND_TRUTH[chain_length] + 0.0016, color='r', linestyle='--')
    plt.axhline(y=GROUND_TRUTH[chain_length] - 0.0016, color='r', linestyle='--')

    plt.xlabel('Iteration')
    plt.ylabel('Ha')
    plt.title(f'Chain length = {chain_length}, Noise = {noise}, Post-selection? = False')
    plt.legend() # Displays the labels we defined in the plot calls
    plt.grid(True, alpha=0.3) # Optional: Adds a subtle background grid

    # 6. Display the graph
    plt.show()
