# AGENTS.md
本リポジトリは「EMDM（等価磁気双極子法）＋凸最適化（SOCP/QCQP）」により、MRI/NMR向け勾配コイル・シムコイルを設計する研究用ツールです。  
開発では **人間（設計判断）** と **Codex（実装・リファクタ・テスト補助）** を役割分担します。

---

## 0. 目的と非目的

### 目的
- 複数の設計面（平面・円盤・円筒展開）に対して、同一の最適化フレームワークでコイル設計ができること
- 目標磁場を **Bz のスカラー場**として、**基底＋係数（NMR慣例）**で入力できること（将来は測定データも）
- 計算結果（設定・行列・S面・検証プロット）を **run単位**で保存し、GUIから再表示できること
- バグりやすい箇所（離散勾配 D、対称縮約 E、周期境界、シーム処理）をテストで守ること

### 非目的（現段階でやらない）
- 実配線（厚み・リターン・ビア・多層配線）の完全電磁界解析
- FEMやMaxwellソルバー連携（将来の拡張候補）
- 超巨大問題（数十万変数）を高速に解く最適化エンジン化（キャッシュ・近似は検討）

---

## 1. コーディング原則（全体）

### 1.1 コア計算は「副作用なし」
- `src/gradientcoil/` は **ファイルI/OやGUIに依存しない**
- 乱数は `seed` を config に持ち、**再現性**を担保する

### 1.2 形状（shape）と単位（unit）を明確に
- 点群: `P_world.shape == (P,3)`
- 変数: `S_unknown.shape == (N,)`
- 行列: `A_x.shape == (P,N)`、`D.shape == (2N,N)`
- 角度は rad、長さは m（ただしCAD出力は mm 変換）

### 1.3 疎行列優先・チャンク計算
- `scipy.sparse` を優先（COOで組んでCSRへ）
- EMDM の `A` 行列構築はセル方向にチャンク分割（メモリ節約）

### 1.4 CVXPY注意事項
- **CVXPY reshape は列優先**：`G_vec = D@Ih` は必ず  
  `Sr = G_vec[0::2]`, `Sv = G_vec[1::2]` の **偶奇スライス**で扱う
- `if curv_term != 0:` のように **制約・式をbool評価しない**（例外の原因）
- SOCP（ECOS）失敗時は SCS fallback を用意（ただし精度は要チェック）

### 1.5 互換性（既存成果物との整合）
- `.npz` 保存キーは **なるべく互換**を維持
- 追加キーはOK、削除・改名は避ける（やむを得ない場合は version を上げる）

---

## 2. 主要コンポーネントと責務

### 2.1 `surfaces/`（設計面）
- 設計セルの  
  `centers[k]`, `normals[k]`, `areas[k]`（3D）と、  
  可視化・等高線用の2D格子（u,v / x,y）を提供
- 境界条件（Dirichlet）、周期境界（θ, 円筒周方向）を surface が責務として持つ

### 2.2 `operators/`（D, E, 正則化用構造）
- `gradient.py`: surface から隣接関係を取得して `D` を構築
- `reduction.py`: 対称縮約 `E` の構築（surface別でもよいがI/F統一）
- `curvature.py`: `∇S` の隣接差（QP/SOC）に必要なペア集合の構築

### 2.3 `physics/`（EMDM）
- `emdm.py`: 双極子磁場カーネル＋ `build_A_xyz(points, surface, planes=...)`
- `roi_sampling.py`: 球面点群生成（Hammersley等）
- ※mu0のスケーリングは **既存コード互換**を守る。単位系の整理は別タスク。

### 2.4 `targets/`（目標磁場 Bz）
- 目標は **Bzスカラー**（shape `(P,)`）を返す設計とし、
  最適化側で `B_target = [0,0,Bz_target]` に埋め込む（必要なら重み付け可）
- 入力は「基底＋係数」（NMR慣例の shim 名）を第一優先
- 将来拡張：測定データ（field map）から Bz を補間してターゲット化

> **未決事項（忘れない）**  
> 高次基底の「スケール/正規化（係数単位を T/m に統一するか）」は判断保留。  
> ただし config に `target_scale_policy` を必ず持たせ、後で変更可能にする。

### 2.5 `optimize/`（CVXPY Problem Builder）
- 「surface + operators + A + target」を受けて、CVXPY Problem を構築する
- 目的：`sum ||B(p)-B_target(p)||_2`（L1-of-L2）を基本形
- 制約：ピッチ（SOC）、符号制約（線形）、TV（SOC）、電力（SOCP/QP）、曲率（SOC/QP）

### 2.6 `runs/`（保存・再表示）
- run_id で `runs/<id>/` に保存
- `config.json`, `summary.json`, `result.npz`, `logs.txt` 等を標準化

---

## 3. Codexへの共通ルール（実装補助用）

Codexに依頼する際は、必ず以下を含む指示文にすること。

- **目的（Goal）**
- **対象ファイル（Files to edit / create）**
- **前提（Context）**：このリポジトリの設計方針、既存API、互換性
- **受け入れ条件（Acceptance criteria）**
- **禁止事項（Constraints）**：互換破壊禁止、テスト必須、計算仕様の保持

Codexは次を守ること：
- 破壊的リファクタを一度にやらない（小さく分割）
- 例外時に黙って握りつぶさない（明示的にraise + message）
- 疎行列のshape・index整合性チェックを追加する
- テストを同時に追加し、再現性を壊さない

---

## 4. テスト方針（最低限）

- `test_gradient_operator.py`  
  - Dのshape、非零数、境界処理（Dirichlet）、周期処理（θ/周方向）のチェック  
- `test_reduction_symmetry.py`  
  - Eによる再構成後の対称性誤差が十分小さいこと  
- `test_targets_shim.py`  
  - shim基底の対称性（偶奇）が期待通りであること  
- `test_smoke_smallsolve.py`  
  - 小格子（例 NR=12, NT=16）でECOSが解けること（タイムアウト短め）

---

## 5. 既知の落とし穴（常に警戒）

- θ方向シーム（0と2π）：可視化と等高線抽出で欠けやゴミ線が出やすい  
- r→0：`r*dθ` が小さく差分係数が巨大化 → eps処理必須  
- CVXPYの `reshape` による勾配成分の取り違え  
- スケール問題：正則化重みの絶対値が目的関数に対して不適切だと解が硬直する  
- mu0×1e3 などのスケール：既存互換のまま進め、後で「単位系整理」章を立てる

---
