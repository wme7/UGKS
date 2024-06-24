import numpy as np

def discrete_maxwell(prim, uspace, ck):
    h = prim[0] * (prim[2] / np.pi)**(1.0 / 2.0) * np.exp(-prim[2] * (uspace - prim[1])**2)
    b = h * ck / (2.0 * prim[2])
    return h, b

def shakhov_part(H, B, qf, prim, uspace, ck, pr):
    H_plus = 0.8 * (1 - pr) * prim[2]**2 / prim[0] * (uspace - prim[1]) * qf * (2 * prim[2] * (uspace - prim[1])**2 + ck - 5) * H
    B_plus = 0.8 * (1 - pr) * prim[2]**2 / prim[0] * (uspace - prim[1]) * qf * (2 * prim[2] * (uspace - prim[1])**2 + ck - 3) * B
    return H_plus, B_plus

def get_conserved(gamma, prim):
    conserved = np.zeros(3)
    conserved[0] = prim[0]
    conserved[1] = prim[0] * prim[1]
    conserved[2] = 0.5 * prim[0] / prim[2] / (gamma - 1.0) + 0.5 * prim[0] * prim[1]**2
    return conserved

def get_primary(gamma, w):
    primary = np.zeros(3)
    primary[0] = w[0]
    primary[1] = w[1] / w[0]
    primary[2] = 0.5 * w[0] / (gamma - 1.0) / (w[2] - 0.5 * w[1]**2 / w[0])
    return primary

def get_gamma(ck):
    return float(ck + 3) / float(ck + 1)

def get_sos(gamma, prim):
    return np.sqrt(0.5 * gamma / prim[2])

def get_tau(mu_ref, omega, prim):
    return mu_ref * 2 * prim[2]**(1 - omega) / prim[0]

def get_heat_flux(h, b, prim, uspace, weight):
    heat_flux = 0.5 * (np.sum(weight * (uspace - prim[1]) * (uspace - prim[1])**2 * h) + np.sum(weight * (uspace - prim[1]) * b))
    return heat_flux

def get_temperature(h, b, prim, uspace, weight, ck):
    temperature = 2.0 * (np.sum(weight * (uspace - prim[1])**2 * h) + np.sum(weight * b)) / (ck + 1) / prim[0]
    return temperature

def get_mu(kn, alpha, omega):
    mu = 5 * (alpha + 1) * (alpha + 2) * np.sqrt(np.pi) / (4 * alpha * (5 - 2 * omega) * (7 - 2 * omega)) * kn
    return mu

def init_geometry(xlength, xscale):
    xnum = int(xlength / xscale)
    xlength = xnum * xscale
    dx = xlength / xnum
    return xnum, dx

def init_velocity_newton(num_u, min_u, max_u):
    unum = (num_u // 4) * 4 + 1
    du = (max_u - min_u) / (unum - 1)
    uspace = np.linspace(min_u, max_u, unum)
    weight = np.zeros_like(uspace)

    for i in range(1, unum + 1):
        weight[i - 1] = newton_coeff(i, unum) * du
    return uspace, weight

def newton_coeff(idx, num):
    if idx == 1 or idx == num:
        return 14.0 / 45.0
    elif (idx - 5) % 4 == 0:
        return 28.0 / 45.0
    elif (idx - 3) % 4 == 0:
        return 24.0 / 45.0
    else:
        return 64.0 / 45.0

def init_allocation(ixmin, ixmax, dx, unum, RKD = np.float64):
    ctr = {}
    for i in range(ixmin - 1, ixmax + 2):
        ctr[i] = {
            'x': (i + 0.5)*dx, # cell center coordinates
            'length': dx, # cell length
            'w': np.zeros(3, dtype=RKD), # density, x-momentum, total energy
            'h': np.zeros(unum, dtype=RKD),
            'b': np.zeros(unum, dtype=RKD),
            'sh': np.zeros(unum, dtype=RKD),
            'sb': np.zeros(unum, dtype=RKD)
        }
    vface = {}
    for i in range(ixmin, ixmax + 2):
        vface[i] = {
            'flux': np.zeros(3, dtype=RKD), # mass flux, x momentum flux, energy flux
            'flux_h': np.zeros(unum, dtype=RKD),
            'flux_b': np.zeros(unum, dtype=RKD)
        }
    return ctr, vface

def interp_inner(cell_L, cell_N, cell_R, EPS = np.finfo(float).eps):
    sL_h = (cell_N['h'] - cell_L['h']) / (0.5 * cell_N['length'] + 0.5 * cell_L['length'])
    sR_h = (cell_R['h'] - cell_N['h']) / (0.5 * cell_R['length'] + 0.5 * cell_N['length'])
    cell_N['sh'] = (np.sign(sR_h) + np.sign(sL_h)) * np.abs(sR_h) * np.abs(sL_h) / (np.abs(sR_h) + np.abs(sL_h) + EPS)
    
    sL_b = (cell_N['b']- cell_L['b']) / (0.5 * cell_N['length'] + 0.5 * cell_L['length'])
    sR_b = (cell_R['b']- cell_N['b']) / (0.5 * cell_R['length'] + 0.5 * cell_N['length'])
    cell_N['sb'] = (np.sign(sR_b) + np.sign(sL_b)) * np.abs(sR_b) * np.abs(sL_b) / (np.abs(sR_b) + np.abs(sL_b) + EPS)

def interp_boundary(cell_N, cell_L, cell_R):
    cell_N['sh'] = (cell_R['h'] - cell_L['h']) / (0.5 * cell_R['length'] + 0.5 * cell_L['length'])
    cell_N['sb'] = (cell_R['b'] - cell_L['b']) / (0.5 * cell_R['length'] + 0.5 * cell_L['length'])

def output(sim_time, RKD = np.float64):
    solution = np.zeros((2, ixmax - ixmin + 3), dtype=RKD)

    for i in range(ixmin - 1, ixmax + 2):
        solution[0, i] = ctr[i]['w'][0]
        solution[1, i] = get_temperature(ctr[i]['h'], ctr[i]['b'], get_primary(ctr[i]['w']))

    rmid = 0.5 * (solution[0, ixmin - 1] + solution[0, ixmax + 1])

    xmid = None
    for i in range(ixmin, ixmax + 1):
        if (solution[0, i] - rmid) * (solution[0, i + 1] - rmid) <= 0:
            xmid = ctr[i]['x'] + (ctr[i + 1]['x'] - ctr[i]['x']) / (solution[0, i + 1] - solution[0, i]) * (rmid - solution[0, i])
            break

    if method_output == NORMALIZE:
        solution[0, :] = (solution[0, :] - solution[0, ixmin - 1]) / (solution[0, ixmax + 1] - solution[0, ixmin - 1])
        solution[1, :] = (solution[1, :] - solution[1, ixmin - 1]) / (solution[1, ixmax + 1] - solution[1, ixmin - 1])

    # Write to file (assuming file operations are handled externally in Python)
    write_to_file(solution, sim_time)


def write_to_file(solution, sim_time):
    # Simulating file operations (replace with actual file writing code)
    print("VARIABLES = X, RHO, T")
    print(f'ZONE  T="Time: {sim_time}", I = {ixmax - ixmin + 1}, DATAPACKING=BLOCK')
    print(" ".join([f'{ctr[i]["x"]}' for i in range(ixmin, ixmax + 1)]))
    for i in range(2):
        print(" ".join([f'{solution[i, j]}' for j in range(ixmin, ixmax + 1)]))