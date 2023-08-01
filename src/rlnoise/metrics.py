import numpy as np
from qibo import quantum_info
from scipy.linalg import sqrtm

def trace_distance(rho1,rho2):
    return quantum_info.trace_distance(rho1,rho2)

def compute_fidelity(density_matrix0, density_matrix1):
    """Compute the fidelity for two density matrices (pure or mixed states).

    .. math::
            F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2
    """
    sqrt_mat1_mat2 = sqrtm(density_matrix0 @ density_matrix1)
    trace = np.real(np.trace(sqrt_mat1_mat2)**2)
    if trace > 1:
        trace=1 #TODO: problem the trace can be sligtlhy > 1! This problem appeared only on the hardware test, so probably the dm matrices are not perfect
    return trace

def bures_distance(density_matrix0, density_matrix1):
    """ Compute the Bures distance between density matrices
    .. math::
        B( \rho , \sigma ) = -\sqrt{2*(1-sqrt(F(\sigma,\rho)))} where F is the fidelity
    """
    return np.sqrt(2*(1-np.sqrt(compute_fidelity(density_matrix0, density_matrix1))))
