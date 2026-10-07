# SourceCodeScope

任意の PyTorch モデルを、サンプル入力で 1 回 forward した結果から可視化し、オフラインで開ける 1 枚の HTML にします。

- **Explainer モード**: テンソルの流れを太さのある帯で描き、各ブロックの出力・Attention・重み・分布を図と詳細パネルに表示
- **Flow モード**: 構造と入出力の形状だけを確認するシンプルな図
- 階層の展開 / 折りたたみ、繰り返しブロックの「×N」表示、表示中の状態の SVG 保存

仕様は [docs/model-viz-spec.md](docs/model-viz-spec.md) を参照してください。

## セットアップ

Windows / Python 3.10 / uv を前提にしています。torch は CUDA 12.8 版 (RTX 50 シリーズ対応) を取得します。

```
uv sync
```

## 使い方

```python
import torch
import sourcecodescope as scs

path = scs.visualize(
    model,  # 任意の nn.Module
    (noisy_amp, noisy_pha),  # Tensor / tuple / dict
    mode="explainer",  # "explainer" | "flow"
    out="out/model.html",
    depth=2,  # 初期の展開深さ
    dim_names={"noisy_amp": ("B", "F", "T")},
)
```

生成された HTML をブラウザで開きます。

| 操作 | 動作 |
|---|---|
| ブロックをクリック | 展開 / 折りたたみ |
| ノードをクリック | 右のパネルに形状・統計量・値を表示 |
| ドラッグ / ホイール | 移動 / 拡大縮小 |
| 「SVG保存」 | 現在の表示を SVG として保存 |

解析結果 (IR) だけを扱うこともできます。

```python
graph = scs.trace(model, inputs)
graph.save("out/model.json")
scs.render(scs.Graph.load("out/model.json"), mode="flow", out="out/model_flow.html")
```

## 検証

```
git clone https://github.com/yxlu-0102/MP-SENet external/MP-SENet
uv run python examples/verify_mpsenet.py --checkpoint external/MP-SENet/best_ckpt/g_best_vb
uv run python examples/verify_generic.py
```

`out/` に MP-SENet、ResNet-18、`nn.TransformerEncoder` の HTML が出力されます。
MP-SENet の `pesq` は学習時の指標計算にしか使われず、ビルドに C コンパイラが必要なため、未インストールの場合はスクリプト側で空のモジュールに差し替えます。

## 品質チェック

```
uv run ruff check .
uv run ruff format --check .
uv run pylint src
uv run ty check
uv run pytest
```

## 構成

| パス | 内容 |
|---|---|
| `src/sourcecodescope/tracer.py` | forward hook と `TorchFunctionMode` による動的トレース |
| `src/sourcecodescope/capture.py` | テンソルの統計量と縮小プレビュー |
| `src/sourcecodescope/classify.py` | モジュール / 演算の表示カテゴリ |
| `src/sourcecodescope/postprocess.py` | 孤立ノードの除去、繰り返しブロックの束ね |
| `src/sourcecodescope/ir.py` | IR (JSON として保存・読み込み可能) |
| `src/sourcecodescope/render.py` | HTML の組み立て |
| `src/sourcecodescope/assets/` | ビューア (`viewer.js` / `viewer.css`) と同梱ライブラリ |

## 同梱ライブラリ

- [D3](https://d3js.org/) 7.9.0 (ISC License)
- [elkjs](https://github.com/kieler/elkjs) 0.9.3 (EPL-2.0)

ライセンス文は `src/sourcecodescope/assets/vendor/` にあります。
