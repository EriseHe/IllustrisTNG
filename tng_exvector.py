"""
Compute ex_vector for a given subhalo and desiredinclination angle.

----------------
get_ex_vector(basePath, snap, subhalo_id, inclination_deg)

    basePath        path to TNG output (e.g. '../sims.TNG/TNG50-1/output')
    snap            snapshot number (e.g. 98 or 99)
    subhalo_id      integer Subfind ID within that snapshot
    inclination_deg inclination angle in degrees:
                    0  -> face-on (along disk normal)
                    90 -> edge-on (in the disk plane)

    returns: numpy array of shape 3 unit vector in our simulation coords! Yayyy!
"""

import numpy as np
import illustris_python as il


# ----adapted from Dylan Nelson's public forum code: inertia tensor -----------------

def _compute_inertia_tensor(basePath, snap, subhalo_id):
    """
    Compute the 3x3 moment of inertia tensor for the star-forming gas of a single subhalo
    """
    # --- subhalo position + R_half(stars) from group catalog ---
    sub = il.groupcat.loadSubhalos(
        basePath,
        snap,
        fields=['SubhaloPos', 'SubhaloHalfmassRadType']
    )

    shPos = sub['SubhaloPos'][subhalo_id]            # (3,)
    STARS_PT = 4                                     # partType for stars
    rHalf = sub['SubhaloHalfmassRadType'][subhalo_id, STARS_PT]
    
    #load only gas bound to this subhalo
    gas = il.snapshot.loadSubhalo(
        basePath,
        snap,
        subhalo_id,
        'gas',
        fields=['Coordinates', 'Masses', 'StarFormationRate']
    )

    coords = gas['Coordinates']          # (N_sub, 3)
    masses = gas['Masses']               # (N_sub,)
    sfr    = gas['StarFormationRate']    # (N_sub,)

    if coords.size == 0:
        raise RuntimeError(f"No gas loaded for subhalo {subhalo_id}")

    # --- periodic distances relative to the subhalo center ------
    header = il.groupcat.loadHeader(basePath, snap)
    box = header['BoxSize']

    dxyz = coords - shPos          # naive offsets
    dxyz -= box * np.round(dxyz / box)   # fix wrapping

    r = np.linalg.norm(dxyz, axis=1)

    # --- select star-forming gas within 2 * rHalf ---------------
    sel = (r <= 2.0 * rHalf) & (sfr > 0.0)
    idx = np.where(sel)[0]

    if idx.size < 10:
        raise RuntimeError(
            f"Not enough star-forming gas cells for subhalo {subhalo_id} "
            f"(found {idx.size}). You may want a stars-based fallback."
        )

    x = dxyz[idx, 0]
    y = dxyz[idx, 1]
    z = dxyz[idx, 2]
    m = masses[idx]

    # --- build inertia tensor I_ij = sum m (r^2 δ_ij - x_i x_j) --
    I = np.zeros((3, 3), dtype=float)

    I[0, 0] = np.sum(m * (y*y + z*z))
    I[1, 1] = np.sum(m * (x*x + z*z))
    I[2, 2] = np.sum(m * (x*x + y*y))

    I[0, 1] = I[1, 0] = -np.sum(m * x*y)
    I[0, 2] = I[2, 0] = -np.sum(m * x*z)
    I[1, 2] = I[2, 1] = -np.sum(m * y*z)

    return I



# ---- principal axes and inclination → ex_vector ------------------------

def _principal_axes(I):
    evals, evecs = np.linalg.eigh(I)
    order = np.argsort(evals)
    evecs = evecs[:, order]

    # CORRECT CHOICE:
    n_hat = evecs[:, 2]   # largest eigenvalue -> disk normal
    m_hat = evecs[:, 0]   # one in-plane axis

    n_hat /= np.linalg.norm(n_hat)
    m_hat /= np.linalg.norm(m_hat)
    return n_hat, m_hat

def _ex_for_inclination(I, inclination_deg):
    n_hat, m_hat = _principal_axes(I)
    i = np.deg2rad(inclination_deg)
    ex = np.cos(i)*n_hat + np.sin(i)*m_hat
    return ex/np.linalg.norm(ex)

# ---- THIS IS THE FUNCTION YOU SHOULD USE DIRECTLY --------------------------------------------------------

def get_ex_vector(basePath, snap, subhalo_id, inclination_deg):
    """
    Returns
    -------
    ex_vec : (3,) ndarray
        Unit ex vector in simulation coordinates.
        Use this directly in e.g. sifu.mk_particle_files(..., ex=ex_vec, ...).
    """
    I = _compute_inertia_tensor(basePath, snap, subhalo_id)
    ex_vec = _ex_for_inclination(I, inclination_deg)
    return ex_vec
