# SourceCodeScope モデル可視化 仕様書 (v1.0)

状態: Fix (2026-10-07)。

## 1. 目的

任意の PyTorch `nn.Module` を、サンプル入力で実際に forward した結果をもとに可視化する。
対象は LLM や Transformer に限らない。CNN、RNN、音声・画像モデル、複数入出力のモデルを同じ仕組みで扱い、特定のアーキテクチャを前提にした作り込みはしない。
torchvista の課題だった 4 点を解消する。

| 課題 | 対応 |
|---|---|
| 見た目のデザイン | ブロック種別ごとの配色、角丸ノード、統一したタイポグラフィ |
| データの流れが見えない | テンソルを「帯」で描き、流れのアニメーションを付ける |
| 中身の値が見えない | 活性値・Attention・統計量を図の中に直接描く |
| 階層の見せ方 | 展開/折りたたみをアニメーション付きで行い、抽象度を連続的に切り替える |

## 2. 2 つの表示モード

同じ解析結果 (IR) から 2 種類の図を出し分ける。

### 2.1 `flow` モード (シンプル)

torchvista 相当の、構造と入出力サイズの確認用。

- ノード = モジュールまたは演算、エッジ = テンソル
- 各ノードに 名前 / 型 / 入力形状 → 出力形状 / パラメータ数 を表示
- エッジに形状ラベル
- 階層の展開/折りたたみ (クリック)、パン/ズーム
- 実データは表示しない (ホバーで統計量のみ)
- 上から下へのレイアウト

### 2.2 `explainer` モード (リッチ)

transformer-explainer の表現手法 (帯、その場展開、行列表示) を借りた、任意モデル向けの 2D 表示。左から右へ流す。
専用表示は「モジュールの型」と「テンソルの次元数」だけで決まり、モデル全体の種類には依存しない。

共通の表現:
- **帯 (リボン)**: ブロック間のテンソルを Sankey 風の帯で描く。太さは要素数の対数、色は流れ元のブロック種別
- **ベクトル/テンソルのストリップ**: 値をヒートマップ化した細長いセルとして帯の端に描く
- **多段階の抽象度**: モデル全体 → ブロック → ブロック内部の演算 を、クリックでその場に展開。展開時は周囲が滑らかに押し広げられる
- **流れのアニメーション**: 帯に沿って入力から出力へ光が走る。再生/停止可
- **ホバー連動**: ノードにホバーすると、その入出力の帯と関連ノードだけを強調
- **詳細パネル**: 選択ノードの形状、dtype、パラメータ数、統計量 (平均/標準偏差/最小/最大)、ヒストグラム
- **繰り返しブロック**: 同じ構造が N 個並ぶ場合 (`ModuleList` など) は「×N」の束として畳み、選んだ 1 個だけ展開できる
- **複数入出力**: 入力・出力がそれぞれ複数あっても、別々の帯として左端・右端に並べる

テンソルの次元数による値の表示 (全ノード共通):

| テンソル | 表示 |
|---|---|
| ベクトル相当 `[B, C]` | ストリップ (1 行のヒートマップ) |
| 系列 `[B, T, C]` / `[B, C, T]` | 2D ヒートマップ (系列 × チャネル) |
| マップ `[B, C, H, W]` | チャネルごとのサムネイルグリッド (チャネル選択可)。スペクトログラムや画像の特徴マップはこれで見る |
| それ以上 | 先頭の次元をスライダーで選んで 2D に落とす |

バッチは先頭 1 件を表示。複素数は振幅と位相に分けて表示。

モジュールの型による専用表示:

| ブロック | 展開時の表示 |
|---|---|
| Attention | ヘッド一覧のサムネイル。ヘッド選択で 位置×位置 の重み行列と、位置間を結ぶ曲線 (太さ = 重み)。Q / K / V → 内積 → スケール → マスク → softmax → 出力 の各段を行列で表示。系列の軸が時間でも周波数でもトークンでも同じ扱い |
| Conv / ConvTranspose | 入力マップ → カーネル (縮小表示、kernel / stride / dilation を図示) → 出力マップ |
| RNN (GRU / LSTM) | 系列方向の隠れ状態ヒートマップ。双方向は 2 段 |
| Linear / MLP | 入力ストリップ → 重み行列 (縮小ヒートマップ) → 出力ストリップ |
| Norm | 正規化前後の分布ヒストグラム |
| 活性化関数 | 関数曲線と、その上の入力値の分布。学習可能なパラメータ (PReLU など) は現在値で描く |
| Embedding | インデックス → 埋め込みベクトル |
| 形状操作 (permute / view / cat / stack / squeeze) | 小さな演算ノード。前後の形状と、どの軸がどこへ移ったかを表示 |
| 残差・スキップ接続 | 本流をまたぐバイパスの帯として描く |
| 上記以外 (自作モジュール含む) | 汎用ノード (形状 + 次元数に応じた入出力表示)。未知のモデルでも必ず全体が描ける |

軸の名前 (`dim_names`) や位置のラベル (`axis_labels`) を渡された場合は、形状表示や行列の軸にそれを使う。

## 3. アーキテクチャ

```
model + inputs ─▶ Tracer ─▶ IR (JSON) ─▶ Renderer ─▶ 単一 HTML
                                                  └▶ SVG
```

### 3.1 Tracer (Python)

- forward を 1 回実行して記録する動的トレース (`torch.no_grad()`, `model.eval()`)
- `nn.Module` は forward hook で入出力を捕捉
- モジュール外の演算 (`+`, `torch.cat`, `matmul`, `softmax`, `F.*` など) は関数ラップで捕捉し、残差接続や自作 Attention の内部も辺として残す
- テンソルの同一性 (id) で辺を張る
- Attention の検出:
  1. `nn.MultiheadAttention` と `F.scaled_dot_product_attention` は専用処理で重みを取得
  2. 自作実装は「softmax の出力が `[..., T, T]` 形状で、直後に matmul される」パターンで検出
  3. 検出できない場合はユーザーが `attention_modules=[...]` で明示
- 実データは容量を抑えて保存: 統計量とヒストグラムは全ノード、値そのものは上限サイズまで縮小 (上限は設定可)

### 3.2 IR (JSON)

- `nodes`: id, 名前 (階層パス), 型, 種別 (attention / linear / norm / ...), 親 id, パラメータ数, 入出力形状, 統計量, 値データへの参照
- `edges`: 元 id, 先 id, 形状, dtype
- `tensors`: 縮小済みの値 (最小値・最大値で 8 bit に量子化し base64)
- `meta`: モデル名, 入力情報, 軸名・ラベル, torch バージョン

IR のスキーマとトレーサは自前で実装し、torchvista には依存しない。`flow` モードも同じ IR から自前で描く。

IR は単独で保存/読み込み可能にし、将来の Data Flow / Call Graph 可視化と共通の土台にする。

### 3.3 Renderer

- HTML: IR と JS/CSS をすべて 1 ファイルに埋め込む。外部 CDN に依存せず、オフラインで開ける
- 描画は SVG (D3)、値の多いヒートマップのみ Canvas を SVG 内に埋め込む
- レイアウトはブラウザ側で計算 (展開/折りたたみのたびに再計算するため)
- ライト/ダーク両対応

### 3.4 SVG 保存

- HTML 上の「SVG 保存」ボタンで、**現在の表示状態** (展開状態・選択ヘッド含む) を 1 枚の SVG として保存
- コマンドや Python から直接 SVG を出す機能は持たない
- フォントはアウトライン化せず、スタイルはインライン化して単体で開けるようにする
- Canvas 部分は PNG として SVG に埋め込む

## 4. インターフェース

```python
import sourcecodescope as scs

scs.visualize(
    model,
    inputs,  # Tensor / tuple / dict (複数入力可)
    mode="explainer",  # "flow" | "explainer"
    out="model.html",
    depth=2,  # 初期の展開深さ
    dim_names=None,  # 例: {"noisy_amp": ("B", "F", "T")}
    axis_labels=None,  # 例: {"T": ["The", "cat", ...]}
    attention_modules=None,  # 自動検出できない場合の明示指定
    max_tensor_elems=4096,  # 値を保存する上限
)
```

`scs.trace(model, inputs)` で IR だけを取得し、`scs.render(ir, mode=...)` で後から描画することもできる。

## 5. 動作環境

- OS: Windows
- GPU: RTX 5070 (CUDA 12.8 版の torch 2.7 以上が必要)
- Python: 3.10 (`requires-python = ">=3.10,<3.11"`、`.python-version` で固定)
- 仮想環境とパッケージ管理: `uv` (`uv sync` で環境を再現、`uv.lock` をコミット)
- CPU / CUDA どちらのモデルでも可 (記録時に CPU へ移す)
- 実行時依存は torch のみ。JS ライブラリはパッケージに同梱

### 5.1 依存関係

依存はすべて `pyproject.toml` に集約する。`requirements.txt` は置かない。

- 実行時依存: `[project.dependencies]`
- 開発ツール (ty / ruff / pylint / pytest): `[dependency-groups].dev`
- 検証モデル用の依存 (torchvision、MP-SENet が import する joblib / matplotlib): `[dependency-groups].verify`
- MP-SENet 本体はパッケージではないため `external/MP-SENet` に clone して使う
- torch は `[tool.uv.sources]` と `[[tool.uv.index]]` で CUDA 12.8 用インデックス (`https://download.pytorch.org/whl/cu128`) を指定

### 5.2 コード品質

以下の 3 つがすべて警告ゼロで通ることを完了条件とする。設定は `pyproject.toml` の `[tool.ruff]` / `[tool.pylint]` / `[tool.ty]` に置く。

```
uv run ruff check .
uv run ruff format --check .
uv run pylint src
uv run ty check
```

- 全関数に型注釈を付ける (Python 3.10 の構文の範囲で書く)
- 警告の抑制コメント (`# noqa`、`# pylint: disable`、`# type: ignore`) は原則使わない。torch の動的な API でやむを得ない場合のみ、行単位で理由を添えて使う
- ルールをファイル全体や設定で一括無効化して警告を消すことはしない

## 6. 非機能要件

- 出力 HTML は既定で 20 MB 以下 (超える場合は値データを自動で縮小し警告)
- 1000 ノード規模でも折りたたみ状態なら滑らかに操作できる
- トレースはモデルの重みと挙動を変更しない (hook とラップは終了時に必ず解除)

## 7. v1 の対象外

- 入力を変えてのブラウザ内再推論 (HTML は 1 回の forward のスナップショット)
- 複数ステップにわたるループ (自己回帰生成、拡散のサンプリング過程) の時間方向の可視化
- 学習中の変化、勾配の可視化
- Call Graph / Dependency Map (別仕様)

## 8. 検証モデルと受け入れ基準

検証モデルは [MP-SENet](https://github.com/yxlu-0102/MP-SENet) の `MPNet` (音声強調)。
入力は `noisy_amp` `[B, F, T]` と `noisy_pha` `[B, F, T]`、出力は `denoised_amp`、`denoised_pha`、`denoised_com` `[B, F, T, 2]`。

MP-SENet 専用のコードを一切書かずに、以下を満たすこと。

1. `scs.visualize(model, (noisy_amp, noisy_pha))` がモデルのコードを変更せずに完走し、両モードの HTML が開ける
2. 2 入力・3 出力がそれぞれ別の帯として描かれる
3. `DenseEncoder` → `TSTransformerBlock` ×N → `MaskDecoder` / `PhaseDecoder` の分岐が最上位の図で読み取れ、繰り返しブロックは「×N」に畳まれている
4. `DenseBlock` 内の `cat` による密結合、`stack` / `permute` / `view` による軸の入れ替え、`atan2` による位相の算出が演算ノードとして辺でつながっている
5. `TSTransformerBlock` 内で時間方向と周波数方向の Attention がそれぞれ検出され、重み行列が表示される
6. Conv 系の入出力がスペクトログラム状のマップとして表示される
7. 自作モジュール (`LearnableSigmoid2d`、`SPConvTranspose2d`) が汎用ノードとして欠けずに描かれ、展開すれば内部の演算が見える
8. 表示中の状態を SVG として保存でき、その SVG が単体で開ける

汎用性の確認として、`torchvision` の ResNet-18 と `nn.TransformerEncoder` でも基準 1 を満たすこと。

## 9. 決定事項の記録

- SVG 保存は HTML 上のボタンのみ。コマンドからの出力は不要
- IR とトレーサは自前実装。torchvista には依存しない
- 対象は任意のモデル。LLM / Transformer 前提にしない
- import 名は `sourcecodescope`
- 環境は Windows / RTX 5070 / Python 3.10 / uv。依存は `pyproject.toml` に集約
- ty / ruff / pylint の警告ゼロを完了条件とする

## 10. v0.1.0 の実装状況

受け入れ基準 1〜8 は MP-SENet で確認済み (torch 2.11.0 / 2.14.1 の CPU 版、Linux)。Windows + CUDA での実行は未確認。

仕様との差分:

| 項目 | 仕様 | v0.1.0 |
|---|---|---|
| 型ごとの専用表示の場所 | 図の中でその場に展開 | 図のカードには出力のサムネイルを表示し、専用表示 (重み、分布、関数曲線、Attention 行列) は右の詳細パネルに出す。コンテナの展開は図の中 |
| Attention の Q / K / V 各段 | 各段を行列で表示 | `nn.MultiheadAttention` と `scaled_dot_product_attention` は重みのみ。自作 Attention は展開すれば matmul / softmax が演算ノードとして見える |
| 4 次元超のテンソル | スライダーで選択 | 先頭のスライスを固定で表示 |
| チャネル選択 | 選択可 | 等間隔に選んだ最大 8 チャネルを一覧表示 |
| 帯の端のストリップ | あり | なし (ノードのサムネイルで代替) |
| RNN の双方向 2 段表示 | あり | なし (出力のヒートマップのみ) |
| モード | 引数で選択 | 引数は初期表示。ページ上でも切り替え可能 |
| 1000 ノード規模の操作性 | 滑らか | 未計測 (検証した最大は 283 ノード) |

静的解析の設定で調整した点 (チェックの無効化はしていない):

- pylint / ruff の引数数の上限を 10 に変更 (`visualize()` が仕様どおり 9 引数のため)
- pylint の属性数の上限を 14 に変更 (IR のデータクラスとトレーサ)
- pylint が torch の C 拡張を解決できるよう `extension-pkg-allow-list` と `generated-members` を設定
