# WORKFLOW.md
本リポジトリは「Git + VSCode + Codex」を前提に、段階的に実装していきます。  
Git初心者向けに、**毎章＝1つの機能を完成**させ、**小さくコミット**する運用を推奨します。

---

## 0. 用語（最小限）
- **Working tree**：作業中のファイル
- **Staging**：コミットに含めるファイル集合
- **Commit**：作業のスナップショット
- **Branch**：コミットの流れ（機能ごとに切る）
- **main**：安定版ブランチ（常に動く状態）

---

## 1. 環境セットアップ（Chapter 1に入る前）

### 1.1 Python環境
推奨：`venv` か `conda`（Windowsでcvxpyが入らない場合はcondaが無難）

例（venv）：
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# mac/linux:
source .venv/bin/activate
pip install -U pip
```

依存（最低限）：
- numpy, scipy, matplotlib
- cvxpy
- ecos, scs（solver）
- streamlit（GUI段階で）

```bash
pip install numpy scipy matplotlib cvxpy ecos scs streamlit
```

> 注意：OS/環境により `pip install cvxpy` が失敗する場合があります。  
> その場合は `conda install -c conda-forge cvxpy ecos scs` を推奨。

---

## 2. Git開始（初心者向け：最短ルート）

### 2.1 初期化
```bash
git init
git branch -M main
```

### 2.2 `.gitignore`（runsやcacheは管理外）
最低限：
```
.venv/
__pycache__/
runs/
*.npz
*.log
```

### 2.3 初回コミット
```bash
git add .
git commit -m "ch0: initialize repository skeleton"
```

---

## 3. ブランチ運用（毎章＝1ブランチ）
Chapter 1 に着手する前にブランチを切る：
```bash
git checkout -b ch1-surface-base
```

作業が終わったら main にマージ（初心者は fast-forward でもOK）：
```bash
git checkout main
git merge ch1-surface-base
```

---

## 4. コミットの粒度（重要）
**1コミット＝1つの意味**にする。  
例：
- `ch1: add ParametricSurfaceGrid interface`
- `ch1: implement DiskPolarSurface`
- `tests: add D operator tests for polar disk`

コミットメッセージの推奨形式：
- `chX: ...`（章番号）
- `tests: ...`
- `docs: ...`
- `refactor: ...`
- `fix: ...`

---

## 5. Codexを最大限活用するための「指示テンプレ」

### 5.1 Codexへの依頼テンプレ（そのまま貼れる）
以下を VSCode の Codex に貼る想定です：

```
Goal:
- （例）disk_polar surface を ParametricSurfaceGrid として実装し、既存コード同等の Xp,Yp,Areas,interior/boundary を生成したい。

Context:
- 本リポジトリはAGENTS.mdの方針に従う（副作用なし、shape厳密、疎行列優先）。
- surface は u-v パラメータ面として扱い、metric_hu/hv を提供する。
- 外周リングは Dirichlet S=0（unknownから除外）、θ方向は周期。
- 既存の極座標SOCPコードのロジックを再利用してよい。

Files to edit/create:
- src/gradientcoil/surfaces/base.py
- src/gradientcoil/surfaces/disk_polar.py
- tests/test_surfaces.py

Acceptance criteria:
- `DiskPolarSurface(NR,NT,R_AP)` が生成できる
- `surface.centers.shape == (Nint,3)`、`surface.areas.shape == (Nint,)`
- `surface.boundary_mask` と `surface.interior_mask` が正しい（outer ring）
- tests が通る（pytest）
- ドキュメントstringと型注釈を追加する

Constraints:
- 既存の数式・差分定義を変えない
- 互換破壊（キー削除/改名）をしない
```

### 5.2 人間がやるべきこと / Codexに任せるべきこと
**人間がやる（判断が必要）**
- 仕様決定（API名、保存フォーマット、スケール方針）
- 「この結果が正しいか」の物理的判断（プロットの妥当性）
- 重み（λ）の探索戦略

**Codexに任せる（手数が多い・機械的）**
- ファイル分割・移植
- 疎行列構築の実装
- テスト雛形の作成
- 型注釈やdocstring付け

---

## 6. 開発の回し方（毎日ルーチン）
1. `git checkout -b chX-...`
2. 変更前に `python -m pytest -q`（落ちるのが正常でもOK。現状把握）
3. Codexに小さなタスクを依頼（1回で大改造させない）
4. 途中でプロットやshapeログで確認
5. テスト追加
6. `git status` → `git add` → `git commit`
7. mainにマージ

---

## 7. 章1に入る前のチェックリスト
- [ ] `python -c "import cvxpy; import ecos; import scs"` が通る
- [ ] `pytest` が動く（まだテスト0件でもOK）
- [ ] `.gitignore` に `runs/` と `*.npz` が入っている
- [ ] `AGENTS.md / WORKFLOW.md / ROADMAP.md` がmainにコミット済み
- [ ] `runs/` の保存先を決めた（例：`repo/runs/`）

---
