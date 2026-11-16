# symmetry.py
import pennylane as qp
import numpy as np
from pennylane import numpy as pnp  # Import PL's numpy wrapper
from monitor import CentrallyMonitoredCost


class SymmetryCF:
    def __init__(self, molecule, hf, qubits, doubles, singles, noise, electrons):
        self.core_const, self.one_electron, self.two_electron = qp.qchem.electron_integrals(molecule)()     

        print(qp.__version__)   

        self.grouped_coeffs, self.grouped_ops, self.unitaries = (
            qp.qchem.basis_rotation(
                self.one_electron,
                self.two_electron,
                cholesky=True,
                tol_factor=1e-5,
                tol_eigval=0.0,
            )
        )

        self.hf = hf
        self.qubits = qubits
        self.doubles = doubles
        self.singles = singles
        self.noise = noise
        self.electrons = electrons

        self.rejected_shots = 0
        self.total_shots_taken = 0
        self.shots = 30

        # Build QNodes wrapped by CentrallyMonitoredCost
        if noise != 0:
            dev = qp.device("default.mixed", wires=qubits, shots=30)
            self.sample = CentrallyMonitoredCost(
                qp.QNode(self.sample_rotated_basis, dev, diff_method="parameter-shift")
            )
        else:
            dev = qp.device("lightning.qubit", wires=qubits, shots=30)
            self.sample = CentrallyMonitoredCost(
                qp.QNode(self.sample_rotated_basis_nonoise, dev, diff_method="parameter-shift")
            )
            
        # FIX 1: Provide the device property so SPSA can read cost.device.shots
        self.device = dev

    @staticmethod
    def evaluate_diagonal_fragment(bitstring, coeffs, ops):
        """
        Evaluates a diagonal Hamiltonian fragment on a single accepted bitstring.
        Works in PennyLane 0.42.3 for Identity, PauliZ(i), and Prod(PauliZ(i), PauliZ(j)).
        """
        energy = 0.0
        # Map bit b in {0, 1} to Pauli-Z eigenvalue in {+1, -1}
        z_values = [1.0 - 2.0 * int(b) for b in bitstring]

        for coeff, op in zip(coeffs, ops):
            # In basis_rotation, Identity has 0 wires (or op.name == "Identity").
            # All other ops are PauliZ(i) or Prod(PauliZ(i), PauliZ(j)), where
            # the eigenvalue is simply the product of z_values[w] over op.wires!
            term_val = 1.0
            if getattr(op, "name", "") != "Identity":
                for w in op.wires:
                    term_val *= z_values[int(w)]

            energy += float(coeff) * term_val
        return energy


    def sample_rotated_basis(self, theta, basis_idx, l=1.0):
        qp.BasisState(self.hf, wires=range(self.qubits))
        num_doubles = len(self.doubles)

        for i, d_gate in enumerate(self.doubles):
            qp.DoubleExcitation(theta[i], wires=d_gate)
            for wire in d_gate:
                qp.BitFlip(self.noise * l, wires=wire)

        for i, s_gate in enumerate(self.singles):
            qp.SingleExcitation(theta[num_doubles + i], wires=s_gate)
            for wire in s_gate:
                qp.BitFlip(self.noise * l, wires=wire)

        # Apply single-particle basis rotation U_l
        qp.BasisRotation(wires=range(self.qubits), unitary_matrix=self.unitaries[basis_idx])
        return qp.sample(wires=range(self.qubits))

    def sample_rotated_basis_nonoise(self, theta, basis_idx, l=1.0):
        qp.BasisState(self.hf, wires=range(self.qubits))
        num_doubles = len(self.doubles)

        for i, d_gate in enumerate(self.doubles):
            qp.DoubleExcitation(theta[i], wires=d_gate)

        for i, s_gate in enumerate(self.singles):
            qp.SingleExcitation(theta[num_doubles + i], wires=s_gate)

        qp.BasisRotation(wires=range(self.qubits), unitary_matrix=self.unitaries[basis_idx])
        return qp.sample(wires=range(self.qubits))

    def __call__(self, theta, l=1.0, **kwargs):
        # Start with the nuclear repulsion core constant
        total_energy = float(np.squeeze(self.core_const))

        for l_idx in range(len(self.unitaries)):
            valid_samples = []
            shots_to_request = self.shots

            while len(valid_samples) < self.shots:                
                self.sample._set_shots(shots_to_request)
                raw_samples = self.sample(theta, l_idx, l=l)
                raw_samples = np.atleast_2d(raw_samples)

                for sample in raw_samples:
                    n_alpha = sum(sample[0::2])
                    n_beta = sum(sample[1::2])
                    n_e = n_alpha + n_beta
                    s_z = 0.5 * (n_alpha - n_beta)

                    if n_e == self.electrons and s_z == 0.0:
                        valid_samples.append(sample)
                        if len(valid_samples) == self.shots:
                            break
                    else:
                        self.rejected_shots += 1

                if len(valid_samples) < self.shots:
                    acc_rate = max(0.01, len(valid_samples) / shots_to_request)
                    shots_to_request = max(1, int((self.shots - len(valid_samples)) / acc_rate))

            # Restore base shots
            self.sample._set_shots(self.shots)

            # Evaluate fragment on accepted samples
            fragment_energy = 0.0
            for sample in valid_samples:
                fragment_energy += self.evaluate_diagonal_fragment(
                    sample, self.grouped_coeffs[l_idx], self.grouped_ops[l_idx]
                )
            total_energy += fragment_energy / self.shots

        self.total_shots_taken = self.sample.total_shots_taken
        
        # FIX 2: Wrap the output in a PennyLane NumPy array so .size exists!
        return pnp.array(total_energy, requires_grad=True)

    def _set_shots(self, shots):
        self.shots = shots
        self.sample._set_shots(shots)
