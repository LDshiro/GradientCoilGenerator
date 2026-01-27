# ROADMAP.md
このロードマップは **10章以内**で、極座標（円盤）だけでなく直交座標・円筒展開まで拡張できる構成を想定しています。  
各章は「人間がやるべき作業」と「Codexに出す指示文（プロンプト）」を必ず含みます。

---

## Chapter 0（準備）：ガイドラインとワークフロー確立
**成果物**
- `AGENTS.md`：設計方針・ルール
- `WORKFLOW.md`：Git+Codex運用
- `ROADMAP.md`：章構成

**人間**
- 章1〜の優先度確認
- OS/solver（ECOS/SCS）インストール確認

**Codex**
- リポジトリの雛形作成、`.gitignore`、`pyproject.toml`（任意）

---

## Chapter 1：Surface抽象（ParametricSurfaceGrid）を確立
**ゴール**
- 極座標/直交/円筒を同じ「2Dパラメータ面」として扱う最小I/Fを決める

**主な成果物**
- `src/gradientcoil/surfaces/base.py`
- `tests/test_surfaces_base.py`

**人間がやる**
- 最小I/Fの設計判断（以下を確定）
  - `centers, normals, areas`（unknownのみ）
  - 可視化用の `grid_u, grid_v` と `grid_x, grid_y`（必要なら）
  - 境界：Dirichletの取り扱い
  - 周期：`is_periodic_v` 等のフラグ
  - metric：`h_u(u,v), h_v(u,v)` の返し方（配列でも可）

**Codexへの指示（例）**
```
Goal:
- ParametricSurfaceGrid の抽象クラス（Protocolでも可）を追加し、最小限の属性/メソッドを定義する。
Context:
- AGENTS.mdの設計方針に従う。将来 disk_polar / plane_cart / cylinder_unwrap が実装される。
Files:
- src/gradientcoil/surfaces/base.py
- tests/test_surfaces_base.py
Acceptance criteria:
- surface 実装が満たすべきshape仕様が明文化される
- pytestが通る
Constraints:
- 余計な機能を入れない（最小I/F）
```

---

## Chapter 2：Surface実装（円盤・極座標）disk_polar
**ゴール**
- 現在の極座標コードを surface として切り出し、同等の格子・面積・境界を生成

**成果物**
- `surfaces/disk_polar.py`
- `tests/test_disk_polar.py`

**人間**
- 「outer ring Dirichlet」仕様の確認
- θシーム（0/2π）の可視化欠け対策を後章で扱うか決める

**Codexプロンプト**
（WORKFLOW.mdのテンプレ参照。Acceptanceにshapeとboundary一致を入れる）

---

## Chapter 3：Surface実装（直交平面）plane_cart
**ゴール**
- 直交平面（正方格子＋円形マスク＋Dirichlet境界）を surface として実装

**成果物**
- `surfaces/plane_cart.py`
- `tests/test_plane_cart.py`

**人間**
- 円形マスクの境界定義（strict interior / boundary）を確定

**Codex**
- マスク生成とidx_map、centers/normals/areasの実装、テスト

---

## Chapter 4：Surface実装（円筒展開）cylinder_unwrap
**ゴール**
- 円筒表面を `(u=z, v=θ)` の展開座標として扱う surface を実装

**成果物**
- `surfaces/cylinder_unwrap.py`
- `tests/test_cylinder_unwrap.py`

**人間**
- 円筒の境界条件（z端をDirichletにするか、自由端にするか）を決める
- 法線方向（半径方向）を確認

**Codex**
- centers/normals/areas と v周期の実装

---

## Chapter 5：Operators（surface共通）離散勾配 D の構築
**ゴール**
- disk/plane/cylinder 共通で `D` を作れる（勾配成分は 2Nint の偶奇スタック）

**成果物**
- `operators/gradient.py`
- `tests/test_gradient_operator.py`

**人間**
- 勾配の定義を確定（forward差分、Dirichlet 0 の織り込み、周期方向）

**Codex**
- `build_D(surface) -> csr_matrix(2Nint,Nint)`
- テスト：周期・境界の期待非零パターン

> 注意：CVXPY reshape バグを避けるため **偶奇スライス**設計を仕様に明記する。

---

## Chapter 6：Physics（EMDM）A行列とキャッシュ
**ゴール**
- 任意 surface から EMDM 前進行列 `A_x,A_y,A_z` を生成

**成果物**
- `physics/emdm.py`
- `physics/roi_sampling.py`
- `tests/test_emdm_shapes.py`

**人間**
- mu0スケールの扱い（既存互換を維持する）確認
- ROI点の分布（Hammersley）を採用

**Codex**
- チャンク計算実装
- 形状テスト

---

## Chapter 6.5：Targets（Bzスカラー）基底＋係数（NMR慣例）
**ゴール**
- 目標磁場は **Bzのみ**（スカラー）で入力する
- ターゲットは **基底＋係数**で表現し、GUIで係数を入力できる形にする
- 将来拡張として「測定fieldmapからBzターゲット」も追加可能なI/Fにする

**成果物**
- `targets/base.py`：`TargetBz` interface（`evaluate_bz(P_world)->(P,)`）
- `targets/shim_basis.py`：NMR慣例の shim 名と基底多項式（少なくとも2次まで）
- `targets/composite.py`：係数辞書で合成
- `targets/fieldmap.py`（スタブ）：後で補間を実装できる枠だけ用意

**人間（判断が必要）**
- 基底の最低セット（例：Z, X, Y, Z2, XZ, YZ, X2-Y2, XY）
- 「スケール/正規化」は**判断保留**：ただし config に `target_scale_policy` を必ず持たせて忘れない  
  - 例：`none / roi_radius / custom_length`
- 係数の単位は当面 **T/m** で扱う想定（高次は後でスケール方針で整合させる）

**Codexプロンプト**
```
Goal:
- Bzターゲットを「shim基底＋係数」で表すtargetsモジュールを追加したい（NMR慣例の名前）。
Context:
- 目標はBzスカラーのみ。将来fieldmap入力も追加したいのでI/Fを拡張可能に。
- スケール/正規化は判断保留だが、configで指定できる設計にして忘れない。
Files:
- src/gradientcoil/targets/{base.py, shim_basis.py, composite.py, fieldmap.py}
- tests/test_targets_shim.py
Acceptance:
- `CompositeShimTarget(coeffs).evaluate_bz(P)` が (P,) を返す
- shim名の辞書と多項式定義がドキュメント化される
- fieldmapはスタブでOK（NotImplementedErrorでも可）
```

---

## Chapter 7：Optimization（SOCP/QCQP）Problem Builder（Bzターゲット対応）
**ゴール**
- `TargetBz` を受けて、誤差を `|| [Bx-0, By-0, Bz-Bz_target] ||_2` で評価（デフォルト）
- 目的：`sum t_p`（L1-of-L2）、制約：ピッチ（SOC）、TV/電力/曲率（任意）

**成果物**
- `optimize/problem_builder.py`
- `optimize/solve.py`
- `tests/test_smoke_smallsolve.py`

**人間**
- 誤差の定義（3DベクトルでBzターゲットのみ）を正式仕様化
- solver fallback 方針（ECOS→SCS）

**Codex**
- CVXPY定式化の実装・テスト

---

## Chapter 8：runs管理（保存・一覧・再読み込み）
**ゴール**
- 設定と結果をrun単位で保存し、GUI/CLIから再表示できる

**成果物**
- `runs/store.py`, `runs/summary.py`

**人間**
- 保存フォーマット（config.json / result.npz / summary.json）確定

**Codex**
- save/load/list とテスト

---

## Chapter 9：Streamlit GUI（surface×target×solver）
**ゴール**
- GUIから surface（disk/plane/cylinder）選択
- GUIから shim 係数（Bzターゲット）入力
- 実行→保存→結果選択→可視化（S面、等高線、軸上|B|）

**成果物**
- `app/` 一式

**人間**
- UX設計（入力項目、既定値、比較ビュー）

**Codex**
- Streamlit画面の実装、runs連携、可視化関数呼び出し

---

## 未決事項（必ず残すメモ）
- [ ] shim基底の **スケール/正規化**（係数単位をT/mに統一する設計）  
- [ ] fieldmap入力（測定データ→Bzターゲット）  
- [ ] 単位系の整理（mu0*1e3互換からSIへ移行するか）  
- [ ] 大規模化に向けたキャッシュ（A行列、D/E）とメモリ戦略

---
