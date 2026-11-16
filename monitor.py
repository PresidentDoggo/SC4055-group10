class CentrallyMonitoredCost:
    def __init__(self, qnode):
        self.qnode = qnode
        self.total_shots_taken = 0

    def __call__(self, *args, **kwargs):
        # 1. Check if shots are overridden in the call (e.g., cost_fn(params, shots=100))
        if "shots" in kwargs and kwargs["shots"] is not None:
            current_shots = kwargs["shots"]
        else:
            # 2. Otherwise, read the active shots directly on the device
            # This handles cost._set_shots(...) calls made inside calcThetaOpt!
            current_shots = self.qnode._shots.total_shots

        # 3. Add to our central counter
        self.total_shots_taken += current_shots

        # 4. Execute the actual QNode and return the result
        return self.qnode(*args, **kwargs)

    def _set_shots(self, shots):
        """
        Passes through any direct shot configuration changes (like cost._set_shots)
        to the underlying QNode and its device.
        """
        self.qnode._set_shots(shots)

    def reset(self):
        """Resets the shot counter back to zero (e.g., between different runs)."""
        self.total_shots_taken = 0
