"""
Polar SOCP for planar gradient coils with curvature regularization as SOC
- Objective: sum of 3D field error norms + (optional) power + (optional) TV + (NEW) curvature SOC
- Constraints: pitch (SOC), sign-in-center (linear), Dirichlet at aperture
- Symmetry: theta reduction (x-even & y-odd) via linear expansion E
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, issparse

# ---------- Physical constant (kept for compatibility with earlier code) ----------
mu0 = 4 * np.pi * 1e-7 * 1e3

# ---------------- Parameters ----------------
# Geometry
R_AP = 150e-3
GAP = 180e-3

# Polar grid
NR = 32
NT = 32  # even
dr = R_AP / NR
dth = 2 * np.pi / NT
r_c = (np.arange(NR) + 0.5) * dr
th_c = (np.arange(NT) + 0.5) * dth

# ROI (sphere)
ROI_RADIUS = 50e-3
N_ROI_BASE = 200

# Target: Y-gradient
G_TARGET = 0.02

# TV (optional)
USE_TV = False
LAMBDA_TV = 0.0

# Pitch constraint
USE_PITCH_CONSTRAINT = True
DELTA_S_A_TURN = 0.010 * 0.08
PITCH_MIN = 0.0010 * 0.08
J_MAX = DELTA_S_A_TURN / PITCH_MIN

# Power regularization
USE_POWER_L2 = True
LAMBDA_POWER_L2 = 1 * 1e-3

USE_POWER_QP = False
LAMBDA_POWER_QP = 10.0
R_SHEET = 0.000492

# -------- NEW: Curvature regularization as SOC (L1-of-L2 of gradient differences) --------
USE_CURV_SOC = True
LAMBDA_CURV_SOC = 4e-7  # tune 0.01~0.3
R_CURV = 150e-3  # apply inside this radius
# (譌ｧ) QP 迚医・辟｡蜉ｹ蛹・
USE_CURV_QP = False
LAMBDA_CURV_QP = 1e-4

# Sign constraint inside radius (suppress polarity flips near center)
USE_SIGN_REGION = True
R_SIGN = 150e-3
SIGN_Y_FACTOR = +1  # +1 => y>0 S>=0 / y<0 S<=0

# Solver & save/plot
SOLVER = "MOSEK"
VERBOSE = True
N_CONTOURS = 36
SAVE_PATH = "polar_SOCP_sum3D_TV_PWR_CURV_SOC_Ygrad.npz"


# ---------------- Utilities ----------------
def _vdc_base2(n: int) -> np.ndarray:
    out = np.empty(n, dtype=float)
    for k in range(n):
        x = 0.0
        denom = 1.0
        m = k
        while m:
            denom *= 2.0
            x += (m & 1) / denom
            m >>= 1
        out[k] = x
    return out


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def hammersley_sphere(samples: int, radius: float, *, rotate=True, seed=42) -> np.ndarray:
    N = int(samples)
    if N <= 0:
        return np.zeros((0, 3), float)
    k = np.arange(N, dtype=float)
    u = (k + 0.5) / N
    z = 1.0 - 2.0 * u
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = 2 * np.pi * _vdc_base2(N)
    x = r * np.cos(th)
    y = r * np.sin(th)
    pts = np.column_stack([x, y, z])
    if rotate:
        rng = np.random.default_rng(seed)
        R = _random_rotation_matrix(rng)
        pts = pts @ R.T
    return radius * pts


def emdm_components(points, centers, normals, areas, chunk=4096):
    P = points.shape[0]
    M = centers.shape[0]
    Ax = np.zeros((P, M))
    Ay = np.zeros((P, M))
    Az = np.zeros((P, M))
    c = mu0 / (4 * np.pi)
    for j0 in range(0, M, chunk):
        j1 = min(M, j0 + chunk)
        C = centers[j0:j1]
        N = normals[j0:j1]
        A = areas[j0:j1][:, None]
        R = points[:, None, :] - C[None, :, :]
        R2 = np.sum(R * R, axis=2)
        Rn = np.sqrt(R2)
        Rh = R / np.maximum(Rn[..., None], 1e-30)
        m = A * N
        md = np.sum(m[None, :, :] * Rh, axis=2)
        B = c * ((3.0 * md)[..., None] * Rh - m[None, :, :]) / np.maximum(Rn[..., None] ** 3, 1e-30)
        Ax[:, j0:j1] = B[..., 0]
        Ay[:, j0:j1] = B[..., 1]
        Az[:, j0:j1] = B[..., 2]
    return Ax, Ay, Az


def closed_contour_levels(S, n_levels):
    smax = float(np.nanmax(np.abs(S)))
    lev = np.linspace(-smax, smax, n_levels + 2)
    return lev[np.abs(lev) > 1e-12 * smax]


# ---------------- Polar grid & masks ----------------
Rc, Th = np.meshgrid(r_c, th_c, indexing="ij")
Xp = Rc * np.cos(Th)
Yp = Rc * np.sin(Th)
Areas_grid = Rc * dr * dth

interior = np.zeros((NR, NT), dtype=bool)
interior[: NR - 1, :] = True
boundary = np.zeros((NR, NT), dtype=bool)
boundary[NR - 1, :] = True

coords_int = np.argwhere(interior)
idx_map = -np.ones((NR, NT), dtype=int)
for k, (ir, jt) in enumerate(coords_int):
    idx_map[ir, jt] = k
Nint = coords_int.shape[0]
print(f"[Polar] interior unknowns: {Nint}")

# ---------------- Dipole sheets (two planes) ----------------
z_top = +GAP / 2.0
z_bot = -GAP / 2.0
centers_top = []
centers_bot = []
normals_top = []
normals_bot = []
areas_top = []
areas_bot = []
for ir, jt in coords_int:
    x = Xp[ir, jt]
    y = Yp[ir, jt]
    a = Areas_grid[ir, jt]
    centers_top.append([x, y, z_top])
    centers_bot.append([x, y, z_bot])
    normals_top.append([0, 0, 1.0])
    normals_bot.append([0, 0, 1.0])
    areas_top.append(a)
    areas_bot.append(a)
centers_top = np.array(centers_top)
centers_bot = np.array(centers_bot)
normals_top = np.array(normals_top)
normals_bot = np.array(normals_bot)
areas_top = np.array(areas_top, float)
areas_bot = np.array(areas_bot, float)

# ---------------- ROI & target (3D vector) ----------------
P0 = hammersley_sphere(N_ROI_BASE, ROI_RADIUS, rotate=True, seed=42)
P = np.vstack(
    [
        np.column_stack([sx * P0[:, 0], sy * P0[:, 1], sz * P0[:, 2]])
        for sx in (+1, -1)
        for sy in (+1, -1)
        for sz in (+1, -1)
    ]
)
B_target = np.column_stack([np.zeros(P.shape[0]), np.zeros(P.shape[0]), G_TARGET * P[:, 1]])


# ---------------- Forward matrices ----------------
def build_A_xyz_polar(P, chunk=512):
    Pn = P.shape[0]
    Ax = np.zeros((Pn, Nint))
    Ay = np.zeros((Pn, Nint))
    Az = np.zeros((Pn, Nint))
    for j0 in range(0, Nint, chunk):
        j1 = min(Nint, j0 + chunk)
        Ct = centers_top[j0:j1]
        Cb = centers_bot[j0:j1]
        Nt = normals_top[j0:j1]
        Nb = normals_bot[j0:j1]
        At = areas_top[j0:j1]
        Ab = areas_bot[j0:j1]
        Ax_t, Ay_t, Az_t = emdm_components(P, Ct, Nt, At, chunk=8192)
        Ax_b, Ay_b, Az_b = emdm_components(P, Cb, Nb, Ab, chunk=8192)
        Ax[:, j0:j1] = Ax_t + Ax_b
        Ay[:, j0:j1] = Ay_t + Ay_b
        Az[:, j0:j1] = Az_t + Az_b
    return Ax, Ay, Az


A_x_int, A_y_int, A_z_int = build_A_xyz_polar(P)

# ---------------- Polar gradient operator D_int: S -> [Sr ; Sﾎｸ/r] ----------------
rows = []
cols = []
data = []


def add_entry(row, k, val):
    rows.append(row)
    cols.append(k)
    data.append(val)


for ir, jt in coords_int:
    k0 = idx_map[ir, jt]
    # Sr row (forward in r)
    row_r = 2 * k0
    if ir + 1 <= NR - 1:
        if interior[ir + 1, jt]:
            kR = idx_map[ir + 1, jt]
            add_entry(row_r, kR, +1.0 / dr)
            add_entry(row_r, k0, -1.0 / dr)
        else:
            add_entry(row_r, k0, -1.0 / dr)
    else:
        add_entry(row_r, k0, 0.0)
    # Sﾎｸ/r row (forward in ﾎｸ, periodic)
    row_t = 2 * k0 + 1
    jt_p = (jt + 1) % NT
    kT = idx_map[ir, jt_p] if interior[ir, jt_p] else -1
    scale = max(1e-12, r_c[ir] * dth)
    if kT >= 0:
        add_entry(row_t, kT, +1.0 / scale)
        add_entry(row_t, k0, -1.0 / scale)
    else:
        add_entry(row_t, k0, -1.0 / scale)

D_int = coo_matrix((data, (rows, cols)), shape=(2 * Nint, Nint)).tocsr()


# ---------------- Symmetry reduction (theta): x-even & y-odd ----------------
def build_E_xeven_yodd_polar(interior_mask):
    visited = np.zeros_like(interior_mask, dtype=bool)
    rowsE = []
    colsE = []
    dataE = []
    group_id = 0
    for ir, jt in np.argwhere(interior_mask):
        if visited[ir, jt]:
            continue
        jx = (NT // 2 - jt - 1) % NT
        jy = (NT - jt - 1) % NT
        jxy = (jt - NT // 2) % NT
        members = []
        for jj, sgn in [(jt, +1.0), (jx, +1.0), (jy, -1.0), (jxy, -1.0)]:
            if interior_mask[ir, jj] and (not visited[ir, jj]):
                k = idx_map[ir, jj]
                rowsE.append(k)
                colsE.append(group_id)
                dataE.append(sgn)
                members.append(jj)
        for jj in members:
            visited[ir, jj] = True
        if len(members) > 0:
            group_id += 1
    return coo_matrix((dataE, (rowsE, colsE)), shape=(Nint, group_id)).tocsr()


E = build_E_xeven_yodd_polar(interior)
A_x = A_x_int @ E
A_y = A_y_int @ E
A_z = A_z_int @ E
D = D_int @ E
Nsym = E.shape[1]
print(f"[Symmetry] reduced vars: {Nsym}/{Nint}")

# ---------------- Variables & base constraints ----------------
Ih = cp.Variable(Nsym)
t = cp.Variable(P.shape[0], nonneg=True)
s = cp.Variable(Nint, nonneg=True)

constraints = []

# 3D error SOC per ROI point
for p in range(P.shape[0]):
    ax = A_x[p, :] @ Ih
    ay = A_y[p, :] @ Ih
    az = A_z[p, :] @ Ih
    bx, by, bz = B_target[p, 0], B_target[p, 1], B_target[p, 2]
    constraints += [cp.norm(cp.hstack([ax - bx, ay - by, az - bz]), 2) <= t[p]]

# Gradient stack: G_vec = [ Sr_0, (Sﾎｸ/r)_0, Sr_1, (Sﾎｸ/r)_1, ... ]
G_vec = D @ Ih
Sr_all = G_vec[0 : 2 * Nint : 2]
Sto_all = G_vec[1 : 2 * Nint : 2]

# TV (optional)
if USE_TV and LAMBDA_TV > 0:
    for k in range(Nint):
        constraints += [cp.norm(cp.hstack([Sr_all[k], Sto_all[k]]), 2) <= s[k]]

# Pitch constraint
if USE_PITCH_CONSTRAINT:
    for k in range(Nint):
        constraints += [cp.norm(cp.hstack([Sr_all[k], Sto_all[k]]), 2) <= J_MAX]

# Sign constraint (inside R_SIGN)
if USE_SIGN_REGION:
    S_int_expr = E @ Ih
    th_vec = np.array([th_c[jt] for (ir, jt) in coords_int], dtype=float)
    r_vec = np.array([r_c[ir] for (ir, jt) in coords_int], dtype=float)
    idx_in = np.where(r_vec <= R_SIGN + 1e-12)[0]
    if idx_in.size > 0:
        sgn_y = np.where(np.sin(th_vec) >= 0.0, 1.0, -1.0)
        constraints += [cp.multiply(SIGN_Y_FACTOR * sgn_y[idx_in], S_int_expr[idx_in]) >= 0]

# -------- NEW: Curvature SOC (neighbor pairs inside R_CURV) --------
curv_soc_term = None
if USE_CURV_SOC and (LAMBDA_CURV_SOC > 0):
    # Cartesian gradient components at centers: Sx,Sy = R(ﾎｸ) * [Sr; Sﾎｸ/r]
    cth = np.cos(th_vec)
    sth = np.sin(th_vec)
    Sx_all = cp.multiply(cth, Sr_all) - cp.multiply(sth, Sto_all)
    Sy_all = cp.multiply(sth, Sr_all) + cp.multiply(cth, Sto_all)

    # Build neighbor pairs (radial and angular)
    x_c = Xp[interior]
    y_c = Yp[interior]  # flatten in same order as coords_int
    pair_i = []
    pair_j = []
    w_list = []
    for ir, jt in coords_int:
        i = idx_map[ir, jt]
        if r_c[ir] > R_CURV + 1e-12:
            continue
        # radial neighbor
        ir2 = ir + 1
        if (ir2 <= NR - 2) and interior[ir2, jt] and (r_c[ir2] <= R_CURV + 1e-12):
            j = idx_map[ir2, jt]
            r_mid = 0.5 * (r_c[ir] + r_c[ir2])
            d_loc = min(dr, max(1e-12, r_mid * dth))
            dij = float(np.hypot(Xp[ir2, jt] - Xp[ir, jt], Yp[ir2, jt] - Yp[ir, jt]))
            d = max(dij, 0.5 * d_loc)
            pair_i.append(i)
            pair_j.append(j)
            w_list.append(1.0 / (d * d))
        # angular neighbor
        jt2 = (jt + 1) % NT
        if interior[ir, jt2] and (r_c[ir] <= R_CURV + 1e-12):
            j = idx_map[ir, jt2]
            r_mid = r_c[ir]
            d_loc = max(1e-12, r_mid * dth)
            dij = float(np.hypot(Xp[ir, jt2] - Xp[ir, jt], Yp[ir, jt2] - Yp[ir, jt]))
            d = max(dij, 0.5 * d_loc)
            pair_i.append(i)
            pair_j.append(j)
            w_list.append(1.0 / (d * d))

    if len(pair_i) > 0:
        idx_i = np.array(pair_i, dtype=int)
        idx_j = np.array(pair_j, dtype=int)
        W = np.array(w_list, dtype=float)
        W = W / np.maximum(W.mean(), 1e-12)  # stabilize scale

        # Per-pair SOC: ||[dSx, dSy]||_2 <= u_m,  objective += ﾎｻ * sum( w_m * u_m )
        dSx = Sx_all[idx_j] - Sx_all[idx_i]
        dSy = Sy_all[idx_j] - Sy_all[idx_i]
        u = cp.Variable(len(idx_i), nonneg=True)
        for m in range(len(idx_i)):
            constraints += [cp.norm(cp.hstack([dSx[m], dSy[m]]), 2) <= u[m]]
        curv_soc_term = LAMBDA_CURV_SOC * cp.sum(cp.multiply(W, u))

# Power regularization
areas_vec = np.array([Areas_grid[ir, jt] for (ir, jt) in coords_int], dtype=float)
Wsqrt = np.sqrt(2.0 * R_SHEET * np.repeat(areas_vec, 2))  # length 2*Nint
weighted_grad = cp.multiply(Wsqrt, G_vec)

# Objective
obj_terms = [cp.sum(t)]
if USE_TV and LAMBDA_TV > 0:
    obj_terms.append(LAMBDA_TV * cp.sum(s))
if USE_POWER_L2 and LAMBDA_POWER_L2 > 0:
    obj_terms.append(LAMBDA_POWER_L2 * cp.norm(weighted_grad, 2))
if USE_POWER_QP and LAMBDA_POWER_QP > 0:
    obj_terms.append(LAMBDA_POWER_QP * cp.sum_squares(weighted_grad))
if curv_soc_term is not None:
    obj_terms.append(curv_soc_term)

objective = cp.Minimize(cp.sum(cp.hstack(obj_terms)))
prob = cp.Problem(objective, constraints)
try:
    prob.solve(solver=SOLVER, verbose=VERBOSE)
except Exception as e:
    print("Primary solver failed, switching to SCS:", e)
    prob.solve(solver="SCS", verbose=VERBOSE, max_iters=50000)

print("SOCP status:", prob.status, "objective:", prob.value)
Ih_val = Ih.value
if Ih_val is None:
    raise RuntimeError("Solver returned no solution.")

# ---------------- Reconstruct S on (r,ﾎｸ) grid ----------------
S_grid = np.zeros((NR, NT), dtype=float)
I_int_full = (E @ Ih_val).ravel() if issparse(E) else np.asarray(E @ Ih_val).ravel()
for k, (ir, jt) in enumerate(coords_int):
    S_grid[ir, jt] = I_int_full[k]  # outer ring remains 0 (Dirichlet)

r_nodes = np.linspace(0.0, R_AP, NR + 1)
th_nodes = np.linspace(0.0, 2 * np.pi, NT + 1)
R_node, TH_node = np.meshgrid(r_nodes, th_nodes, indexing="ij")
X_node = R_node * np.cos(TH_node)
Y_node = R_node * np.sin(TH_node)

# ・亥盾閠・ｼ臥ｭ蛾ｫ倡ｷ壹ｒ驥阪・縺溘＞蝣ｴ蜷医・縲∝捉譛滓僑蠑ｵ縺ｧ seam 繧帝哩縺倥ｋ・・
# theta 縺ｮ 1 蛻励ｒ隍・｣ｽ縺励※諡｡蠑ｵ縺励…ontour/contourf 縺ｫ貂｡縺・
S_ext = np.concatenate([S_grid, S_grid[:, :1]], axis=1)  # (NR, NT+1)
TH_ext = np.concatenate([Th, Th[:, :1] + 2 * np.pi], axis=1)  # (NR, NT+1)
RC_ext = np.concatenate([Rc, Rc[:, :1]], axis=1)  # (NR, NT+1)
X_ext = RC_ext * np.cos(TH_ext)
Y_ext = RC_ext * np.sin(TH_ext)


# ---------------- Robust filled plot (seamless) ----------------
def circle_boundary(ax, R):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R * np.sin(th), "k--", lw=1, alpha=0.6)


levels = closed_contour_levels(S_grid, N_CONTOURS)
plt.figure(figsize=(7, 6))
cf = plt.contourf(X_ext, Y_ext, S_ext, levels=levels, antialiased=True, cmap="viridis")
plt.contour(X_ext, Y_ext, S_ext, levels=levels, colors="k", linewidths=0.8, alpha=0.6)
circle_boundary(plt.gca(), R_AP)
plt.gca().set_aspect("equal")
plt.title("Stream-function (POLAR, contourf with periodic ﾎｸ-extension)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.colorbar(cf, shrink=0.9, label="S [A]")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()


# ---------------- Verification: |B| on axis (x=0,z=0) ----------------
def B_from_planes(points, chunk=512):
    Pn = points.shape[0]
    Bx = np.zeros(Pn)
    By = np.zeros(Pn)
    Bz = np.zeros(Pn)
    for j0 in range(0, Nint, 512):
        j1 = min(Nint, j0 + 512)
        Ct = centers_top[j0:j1]
        Cb = centers_bot[j0:j1]
        Nt = normals_top[j0:j1]
        Nb = normals_bot[j0:j1]
        At = areas_top[j0:j1]
        Ab = areas_bot[j0:j1]
        Bx_t, By_t, Bz_t = emdm_components(points, Ct, Nt, At, chunk=8192)
        Bx_b, By_b, Bz_b = emdm_components(points, Cb, Nb, Ab, chunk=8192)
        Bx += (Bx_t + Bx_b) @ I_int_full[j0:j1]
        By += (By_t + By_b) @ I_int_full[j0:j1]
        Bz += (Bz_t + Bz_b) @ I_int_full[j0:j1]
    return Bx, By, Bz


ny = 251
y_line = np.linspace(-ROI_RADIUS, ROI_RADIUS, ny)
pts_line = np.stack([np.zeros_like(y_line), y_line, np.zeros_like(y_line)], axis=1)
Bx_line, By_line, Bz_line = B_from_planes(pts_line)
Bnorm_line = np.sqrt(Bx_line**2 + By_line**2 + Bz_line**2)
Bnorm_ideal = np.abs(G_TARGET * y_line)

mask_pos = y_line >= 0
alpha_fit_pos, intercept_pos = np.polyfit(y_line[mask_pos], Bnorm_line[mask_pos], 1)
print(f"On-axis |B| slope (y>=0): {alpha_fit_pos:.6e} T/m  (ratio={alpha_fit_pos / G_TARGET:.3f})")

plt.figure(figsize=(6, 4))
plt.plot(y_line * 1e3, Bnorm_line * 1e6 * 2, label="|B| (computed) [ﾂｵT]")
plt.plot(y_line * 1e3, Bnorm_ideal * 1e6, "--", label="|Gﾂｷy| (ideal) [ﾂｵT]")
plt.xlabel("y at z=0 [mm]")
plt.ylabel("|B| [ﾂｵT]")
plt.title("Mid-plane line: |B| vs |Gﾂｷy| (curvature SOC)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- Save ----------------
np.savez(
    SAVE_PATH,
    Ih=Ih_val,
    I_int_full=I_int_full,
    NR=NR,
    NT=NT,
    dr=dr,
    dth=dth,
    r_c=r_c,
    th_c=th_c,
    Xp=Xp,
    Yp=Yp,
    Areas=Areas_grid,
    S_polar=S_grid,
    interior=interior,
    boundary=boundary,
    R_AP=R_AP,
    GAP=GAP,
    centers_top=centers_top,
    centers_bot=centers_bot,
    normals_top=normals_top,
    normals_bot=normals_bot,
    areas_top=areas_top,
    areas_bot=areas_bot,
    ROI_POINTS=P,
    ROI_RADIUS=ROI_RADIUS,
    A_x_int=A_x_int,
    A_y_int=A_y_int,
    A_z_int=A_z_int,
    A_x_red=A_x,
    A_y_red=A_y,
    A_z_red=A_z,
    D_int=D_int,
    D_red=D,
    G_TARGET=G_TARGET,
    y_line=y_line,
    Bx_line=Bx_line,
    By_line=By_line,
    Bz_line=Bz_line,
    Bnorm_line=Bnorm_line,
    Bnorm_ideal=Bnorm_ideal,
    # regularizers / constraints
    USE_TV=USE_TV,
    LAMBDA_TV=LAMBDA_TV,
    DELTA_S_A_TURN=DELTA_S_A_TURN,
    PITCH_MIN=PITCH_MIN,
    J_MAX=J_MAX,
    USE_PITCH_CONSTRAINT=USE_PITCH_CONSTRAINT,
    USE_POWER_L2=USE_POWER_L2,
    LAMBDA_POWER_L2=LAMBDA_POWER_L2,
    USE_POWER_QP=USE_POWER_QP,
    LAMBDA_POWER_QP=LAMBDA_POWER_QP,
    R_SHEET=R_SHEET,
    USE_SIGN_REGION=USE_SIGN_REGION,
    R_SIGN=R_SIGN,
    SIGN_Y_FACTOR=SIGN_Y_FACTOR,
    USE_CURV_SOC=USE_CURV_SOC,
    LAMBDA_CURV_SOC=LAMBDA_CURV_SOC,
    R_CURV=R_CURV,
    symmetry="theta-reduction: x-even & y-odd; 3D target fit + curvature SOC",
    solver=(prob.solver_stats.solver_name if prob.solver_stats is not None else "unknown"),
)
print("Saved:", SAVE_PATH)
