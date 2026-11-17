import os
import sys
import tempfile

import numpy as np

# Import physics_engine from the repo root, one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_engine import ParticlePusher                # noqa: E402

def test_maxwellian_sampling():
    print("Testing Maxwell-Boltzmann velocity sampling and diagnostic plotting...")
    
    pusher = ParticlePusher(equilibrium=None)

    num_particles = 10000
    T_keV = 1.0

    # 1. JIT-compiled sampling
    velocities = pusher.initialize_velocities(num_particles, T_keV=T_keV)
    
    assert velocities.shape == (num_particles, 3), f"Expected shape {(num_particles, 3)}, got {velocities.shape}"
    print(f"Successfully sampled {num_particles} velocity vectors.")
    
    # 2. Check the standard deviation
    k_B = 1.380649e-23
    T_K = T_keV * 1.16045e7
    v_th_expected = np.sqrt(k_B * T_K / pusher.m)
    
    vx_std = np.std(velocities[:, 0])
    vy_std = np.std(velocities[:, 1])
    vz_std = np.std(velocities[:, 2])
    
    print(f"Expected thermal velocity (std): {v_th_expected:.4e} m/s")
    print(f"Sampled standard deviations: vx={vx_std:.4e}, vy={vy_std:.4e}, vz={vz_std:.4e}")
    
    # 3% tolerance -- reliable at this N
    tolerance = 0.03
    assert abs(vx_std - v_th_expected) / v_th_expected < tolerance, "vx deviation out of range"
    assert abs(vy_std - v_th_expected) / v_th_expected < tolerance, "vy deviation out of range"
    assert abs(vz_std - v_th_expected) / v_th_expected < tolerance, "vz deviation out of range"
    print("Statistical verification passed!")
    
    # 3. Diagnostic plot, written to a temp directory so the test leaves nothing
    # behind in the repo.
    with tempfile.TemporaryDirectory(prefix="fusion_maxwellian_") as out_dir:
        plot_filename = os.path.join(out_dir, "test_maxwell_boltzmann_diagnostic.png")
        pusher.plot_velocity_diagnostic(velocities, T_keV=T_keV, filename=plot_filename)

        assert os.path.exists(plot_filename), "Diagnostic plot was not saved!"
        print("Visual diagnostic plotting passed (temp file written and removed).")
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_maxwellian_sampling()
