import os

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 物理定数（Biot–Savart 用は SI の μ0 を使用）
mu0 = 4 * np.pi * 1e-7 * 1e3 * 2

# ========= User params =========
# ここを極座標 SOCP（または ADMM）で保存した npz に合わせてください
NPZ_PATH = "polar_SOCP_sumOf3DNorms_TV_PWR_circular_Ygrad_xEven_yOdd.npz"
FALLBACK_ADMM = None  # 例: "polar_ADMM_xxx.npz"

# 等高線抽出の健全性フィルタ
MIN_VERTICES = 12  # 短い偽ループを除外
AREA_EPS = 1e-8  # [m^2] 極小スリバーを除外
MAX_TURNS = None  # ループ本数の上限（None=制限なし）

# Bz プロファイル（x固定）の本数と解像度
N_X_SLICES = 9
N_Y_SAMPLES = 401

# 表示倍率（z軸を Bz に使うので、見やすさのため倍率可）
BZ_SCALE = 1.0
# ===============================


# ---------- 小物ユーティリティ ----------
def load_npz(path, fallback=None):
    if os.path.exists(path):
        print("[Load] Using:", path)
        return np.load(path, allow_pickle=True)
    if fallback and os.path.exists(fallback):
        print("[Load] Fallback:", fallback)
        return np.load(fallback, allow_pickle=True)
    raise FileNotFoundError("npz not found. Set NPZ_PATH to your saved file.")


def pick_key(D, *candidates, default=None):
    for k in candidates:
        if k in D.files:
            return D[k]
    if default is not None:
        return default
    raise KeyError(f"None of keys {candidates} found in NPZ.")


def polygon_signed_area(V):
    x = V[:, 0]
    y = V[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def ensure_ccw(V):
    return V if polygon_signed_area(V) >= 0 else V[::-1].copy()


def split_path_to_loops(path_obj):
    """
    Path.codes を用いてサブパスごとに分解し、個別の閉曲線配列を返す。
    - MOVETO で開始、LINETO で追加、CLOSEPOLY でループ完結。
    - 最終ループも明示的にクローズ、微小スリバー/未閉曲線を除去。
    """
    V = path_obj.vertices
    C = path_obj.codes
    loops = []
    if C is None:
        W = V.copy()
        if not np.allclose(W[0], W[-1]):
            W = np.vstack([W, W[0]])
        loops.append(W)
    else:
        cur = []
        for v, code in zip(V, C, strict=False):
            if code == mpath.Path.MOVETO:
                if len(cur) > 0:
                    if not np.allclose(cur[0], cur[-1]):
                        cur.append(cur[0])
                    loops.append(np.array(cur))
                    cur = []
                cur = [v]
            elif code == mpath.Path.LINETO:
                cur.append(v)
            elif code == mpath.Path.CLOSEPOLY:
                if len(cur) > 0:
                    if not np.allclose(cur[0], cur[-1]):
                        cur.append(cur[0])
                    loops.append(np.array(cur))
                    cur = []
            else:
                cur.append(v)
        if len(cur) > 0:
            if not np.allclose(cur[0], cur[-1]):
                cur.append(cur[0])
            loops.append(np.array(cur))

    # 健全性フィルタ
    cleaned = []
    for W in loops:
        if W.shape[0] < max(4, MIN_VERTICES):  # 頂点数が少なすぎる
            continue
        if np.linalg.norm(W[0] - W[-1]) > 1e-10:  # 未閉曲線
            continue
        if abs(polygon_signed_area(W)) < AREA_EPS:  # 面積ほぼゼロ
            continue
        cleaned.append(W)
    return cleaned


def grid_contour_loops(X, Y, Z, levels):
    """格子データ X,Y,Z から contour を作り、レベルごとに Path を閉曲線へ分解して返す。"""
    fig = plt.figure()
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.close(fig)
    out = []
    for lev, coll in zip(levels, cs.collections, strict=False):
        level_loops = []
        for pth in coll.get_paths():
            level_loops.extend(split_path_to_loops(pth))
        out.append((lev, level_loops))
    return out


def build_loops_by_level_grid(X, Y, Z, delta_S, max_turns=None):
    """
    ΔS 間隔の等高線を抽出し、値ごと・ループごとに分離。
    sign(level) に基づき、赤（S>0）/ 青（S<0）で色分けできるように準備。
    """
    Smin, Smax = float(np.nanmin(Z)), float(np.nanmax(Z))
    if not np.isfinite(Smin) or not np.isfinite(Smax) or Smax - Smin < 1e-15:
        raise RuntimeError("Invalid S range for contour extraction.")
    k0 = int(np.ceil((Smin - 0.5 * delta_S) / delta_S))
    k1 = int(np.floor((Smax - 0.5 * delta_S) / delta_S))
    ks = np.arange(k0, k1 + 1, dtype=int)
    if max_turns is not None and len(ks) > max_turns:
        mid = len(ks) // 2
        half = max_turns // 2
        ks = ks[mid - half : mid - half + max_turns]
    if ks.size == 0:
        # 範囲が狭すぎる場合のフォールバック
        ks = np.array([0], dtype=int)
    levels = (0.5 * delta_S + ks * delta_S).astype(float)

    raw = grid_contour_loops(X, Y, Z, levels)
    groups = []
    for c, loops in raw:
        if abs(c) < 1e-14:
            continue  # S=0 近傍は除外
        sgn = 1 if c > 0 else -1
        loops_ccw = [ensure_ccw(W) for W in loops]
        groups.append({"level": c, "sign": sgn, "loops": loops_ccw})
    return groups, levels


def biot_savart_Bz_at_points(groups, gap, P, I_per_turn=1.0, chunk_pts=5000, eps=1e-12):
    """任意点群 P(N,3) における Bz（上下 2 平面の配線）を中点則で評価。"""
    Bz = np.zeros(P.shape[0], dtype=float)

    def accum(z_plane):
        nonlocal Bz
        for g in groups:
            I_loop = I_per_turn * g["sign"]  # CCW 基準の向き。符号は等高線レベルの符号で決定。
            for W in g["loops"]:
                P3 = np.column_stack([W, np.full(len(W), z_plane)])
                Pn = np.roll(P3, -1, axis=0)
                dl = Pn - P3
                mid = 0.5 * (P3 + Pn)
                for s0 in range(0, P.shape[0], chunk_pts):
                    s1 = min(P.shape[0], s0 + chunk_pts)
                    R = P[s0:s1, None, :] - mid[None, :, :]
                    R2 = np.sum(R * R, axis=2) + eps
                    R32 = R2 * np.sqrt(R2)
                    dB = (
                        (mu0 / (4 * np.pi))
                        * I_loop
                        * np.cross(dl[None, :, :], R, axis=2)
                        / R32[..., None]
                    )
                    Bz[s0:s1] += np.sum(dB[..., 2], axis=1)

    accum(+gap / 2.0)
    accum(-gap / 2.0)
    return Bz


# ---------- メイン ----------
def main():
    # --- NPZ 読み込み ---
    D = load_npz(NPZ_PATH, FALLBACK_ADMM)

    # 必須（位置・形状）
    R_AP = float(pick_key(D, "R_AP"))
    GAP = float(pick_key(D, "GAP"))
    ROI_RADIUS = float(pick_key(D, "ROI_RADIUS", default=0.15))

    # グリッド（優先：保存済み Xp,Yp；無ければ r_c,th_c から再構成）
    if "Xp" in D.files and "Yp" in D.files:
        Xp = D["Xp"]
        Yp = D["Yp"]
    else:
        r_c = pick_key(D, "r_c")
        th_c = pick_key(D, "th_c")
        Rc, Th = np.meshgrid(r_c, th_c, indexing="ij")  # (NR,NT)
        Xp = Rc * np.cos(Th)
        Yp = Rc * np.sin(Th)

    # ストリーム関数 S（極座標格子）
    if "S_grid" in D.files:
        S_grid = D["S_grid"]
    elif "S_polar" in D.files:
        S_grid = D["S_polar"]
    else:
        raise KeyError(
            "S_grid / S_polar が見つかりません。極座標 SOCP の保存ファイルを指定してください。"
        )

    # 等高線間隔（1ターンあたりの ΔS）
    DELTA_S = float(pick_key(D, "DELTA_S_A_TURN", default=1.0))
    # 目標（比較用）
    G_TARGET = float(pick_key(D, "G_TARGET", default=0.01))

    # --- 等高線（値ごと・ループごと）抽出（サブパス分解+閉曲線検査） ---
    groups, levels = build_loops_by_level_grid(Xp, Yp, S_grid, DELTA_S, max_turns=MAX_TURNS)
    n_loops = sum(len(g["loops"]) for g in groups)
    print(f"[Extract] levels={len(groups)} loops={n_loops}  ΔS={DELTA_S} A  R_AP={R_AP} m")

    if n_loops == 0:
        raise RuntimeError(
            "抽出された閉じた等高線がありません。S のレンジと ΔS を確認してください。"
        )

    # --- 3D：配線（z=±GAP/2） + Bz(y) プロファイル（複数 x スライス） ---
    fig = plt.figure(figsize=(11, 8.5))
    ax3d = fig.add_subplot(111, projection="3d")

    # アパーチャ円（破線）と ROI 円（点線）を z=0 に描画
    th = np.linspace(0, 2 * np.pi, 600)
    ax3d.plot(
        R_AP * np.cos(th), R_AP * np.sin(th), 0 * th, "k--", lw=1.2, alpha=0.9, label="Aperture"
    )
    ax3d.plot(
        ROI_RADIUS * np.cos(th),
        ROI_RADIUS * np.sin(th),
        0 * th,
        "k:",
        lw=1.4,
        alpha=0.9,
        label="ROI",
    )

    # 配線（level ごと・loop ごとに完全分離して plot：誤接続なし）
    for g in groups:
        col = "r" if g["sign"] > 0 else "b"
        for W in g["loops"]:
            ax3d.plot(W[:, 0], W[:, 1], np.full(len(W), +GAP / 2.0), color=col, lw=1.8, alpha=0.95)
            ax3d.plot(W[:, 0], W[:, 1], np.full(len(W), -GAP / 2.0), color=col, lw=1.8, alpha=0.95)

    # Bz 断面（x 固定）：x=0 を含む N_X_SLICES 本
    xs = np.linspace(-0.9 * ROI_RADIUS, +0.9 * ROI_RADIUS, N_X_SLICES)
    xs = np.unique(np.concatenate(([0.0], xs)))
    xs = xs[np.argsort(np.abs(xs))]  # 0, ±… の順
    colors = cm.tab10(np.linspace(0, 1, len(xs)))
    y_line = np.linspace(-ROI_RADIUS * 2.5, ROI_RADIUS * 2.5, N_Y_SAMPLES)

    for ii, x0 in enumerate(xs):
        P = np.column_stack([np.full_like(y_line, x0), y_line, np.zeros_like(y_line)])
        Bz = biot_savart_Bz_at_points(groups, GAP, P, I_per_turn=DELTA_S)
        ax3d.plot(
            np.full_like(y_line, x0),
            y_line,
            BZ_SCALE * Bz * 1e2,
            color=colors[ii],
            lw=2.0 if np.isclose(x0, 0) else 1.5,
            alpha=1.0 if np.isclose(x0, 0) else 0.85,
            label="x=0" if np.isclose(x0, 0) else None,
        )

    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel(f"z ≡ Bz [T] × {BZ_SCALE:g}")
    ax3d.set_title("Disjoint closed coils (red S>0 / blue S<0) + Bz profiles at z=0")
    ax3d.set_box_aspect((1, 1, 0.5))
    plt.tight_layout()
    plt.show()

    # --- 参考：x=0 の理想直線との比較 ---
    P0 = np.column_stack([np.zeros_like(y_line), y_line, np.zeros_like(y_line)])
    Bz0 = biot_savart_Bz_at_points(groups, GAP, P0, I_per_turn=DELTA_S)
    Bz_ideal = G_TARGET * y_line
    plt.figure(figsize=(6.6, 4.6))
    plt.plot(y_line * 1e3, Bz0 * 1e6, label="Biot–Savart Bz [µT]")
    plt.plot(y_line * 1e3, Bz_ideal * 1e6, "--", label="Ideal G·y [µT]")
    plt.xlabel("y at z=0 [mm]")
    plt.ylabel("Bz [µT]")
    plt.title("On-axis (x=0) Bz")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
