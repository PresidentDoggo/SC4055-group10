import numpy as np
from pennylane import numpy as pnp
import time
import math
import pennylane as qp

H2_shots = 30
H2_shots_eval_nonoise = 100000
H2_iters = 200
H2_scale = 1.9480
H2_params = 3
H2_avg_gate_wires = 4/3

def richardson_extrapolate(scale_factors, values):
    """
    Exact Richardson extrapolation to l = 0 using Lagrange polynomials.
    Works for both scalar energies and 1D parameter arrays.
    """
    scale_factors = np.asarray(scale_factors, dtype=float)
    values = np.asarray(values, dtype=float)
    
    n = len(scale_factors)
    weights = np.zeros(n)
    
    for i in range(n):
        # Compute Lagrange basis polynomial L_i(0)
        num = 1.0
        den = 1.0
        for j in range(n):
            if i != j:
                num *= (0.0 - scale_factors[j])
                den *= (scale_factors[i] - scale_factors[j])
        weights[i] = num / den

    # Weighted sum across scale factors (axis 0)
    return np.tensordot(weights, values, axes=(0, 0))


def poly_extrapolate(scale_factors, values, degree=1):
    """
    Least-squares polynomial extrapolation to l = 0.
    degree=1 is linear ZNE; degree=len(scale_factors)-1 equals Richardson extrapolation.
    """
    coeffs = np.polyfit(scale_factors, values, deg=degree)
    # In np.polyfit, the last coefficient coeffs[-1] is the intercept at l = 0
    return coeffs[-1]

def calcThetaOpt(cost, init_params, scale, noise, avg_wire_per_gate, **kwargs):

    shots = int(max(H2_shots, H2_shots * ((scale / H2_scale)**2)))

    shots_eval = int(max(H2_shots_eval_nonoise, H2_shots_eval_nonoise * ((scale / H2_scale)**2)))

    iterations = int(H2_iters * (len(init_params) / H2_params) * (scale / H2_scale) * (1 / (1 - 2 * noise))**(2*(len(init_params)*avg_wire_per_gate - H2_params*H2_avg_gate_wires)))


    if (iterations > 200):
        print(f"Num iterations = {iterations} is too high. Clipping at 200.")
        iterations = 200

    print(f"Scale = {scale}, #Params = {len(init_params)}, Avg wires per gate = {avg_wire_per_gate}")

    print(f"Num shots per iter = {shots}, Eval shots = {shots_eval}, Num iter = {iterations}, Noise = {noise}")

    cost._set_shots(shots)

    total_error = (math.sqrt((1/shots) + len(init_params) * avg_wire_per_gate * noise)) / ((1 - 2*noise)**(len(init_params * avg_wire_per_gate)))
    
    opt = qp.SPSAOptimizer(maxiter=iterations, c=((total_error**(1/3))))

    params = init_params.copy()
    


    #######
    # OPTIMIZER LOOP
    #######

    printInterval = 1 if iterations <= 20 else 10 # Print frequency

    param_history = []
    energies = []
    polyak_energies = []

    

    start = time.perf_counter()

    for i in range(iterations):

        params = pnp.array(params, requires_grad=True)
        params, energy = opt.step_and_cost(cost, params, **kwargs)
        energy = float(energy)
        energies.append(energy)

        if i >= int(0.90 * iterations):
            param_history.append(np.array(params))
            polyak_energies.append(cost(np.mean(param_history, axis=0)))

        else:
            polyak_energies.append(energy)
        
        if (i % printInterval == 0):
            print(f"Iteration: {i} | Energy: {energy:.5f}")



    #######
    # Polyak-Ruppert average
    #######
    #last = energies[-len(param_history):]
    #lowest = max(last)
    #theta_opt = param_history[last.index(lowest)]
    theta_opt = np.mean(param_history, axis=0)
    print(f"Parameter optimization time: {(time.perf_counter() - start):.6f} seconds\n")

    cost._set_shots(shots_eval)

    energy_VQE = float(cost(theta_opt, **kwargs))


    return theta_opt, energies, polyak_energies, energy_VQE


def run_zne_vqe(
    cost_fn,
    init_params,
    scale,
    noise,
    avg_wires_per_gate,
    scale_factors=(1.0, 2.0, 3.0),
    method="richardson",
    poly_degree=1,
    **kwargs
):
    np.random.seed(42)
    shots_eval = int(max(100000, 100000 * (scale / 1.9480)**2))

    scale_factors = list(scale_factors)
    thetas_per_scale = []
    energies_per_scale = []
    all_opt_histories = {}

    base_l = scale_factors[0]
    print(f"\n--- [ZNE] Optimizing Parameters at Base Noise Scale l = {base_l} ---")
    theta_base, energies_base, polyak_base, energy_VQE_base = calcThetaOpt(
        cost_fn,
        pnp.array(init_params, requires_grad=True),
        scale,
        noise,
        avg_wires_per_gate,
        l=base_l,
        **kwargs
    )
    
    theta_zne = np.array(theta_base)
    thetas_per_scale.append(theta_zne)
    energies_per_scale.append(float(energy_VQE_base))
    all_opt_histories[base_l] = {
        "energies": energies_base,
        "polyak_energies": polyak_base,
        "energy_VQE": energy_VQE_base,
        "theta_opt": theta_base,
        "shots": cost_fn.total_shots_taken
    }

    # Evaluate at higher noise scale factors with fixed theta_base
    cost_fn._set_shots(shots_eval)
    for l_val in scale_factors[1:]:
        print(f"--- [ZNE] Evaluating fixed circuit at noise scale l = {l_val} (Noise = {l_val * noise})")

        e_l = float(cost_fn(theta_zne, l=l_val, **kwargs))
        energies_per_scale.append(e_l)
        print(f"--- [ZNE] Scale l = {l_val} | Noisy Energy = {e_l:.5f} Ha ---")

    # Extrapolate energy to l = 0
    if method == "richardson":
        print("Using richardson extrapolation")
        energy_zne = float(richardson_extrapolate(scale_factors, energies_per_scale))
    else:
        print("Using polynomial extrapolation")
        energy_zne = float(poly_extrapolate(scale_factors, energies_per_scale, degree=poly_degree))

    zne_data = {
        "scale_factors": scale_factors,
        "energies_per_scale": energies_per_scale,
        "thetas_per_scale": thetas_per_scale,
        "histories": all_opt_histories,
    }

    return theta_zne, energy_zne, zne_data


# Guard so importing from zne.py never executes standalone code
if __name__ == "__main__":
    pass
