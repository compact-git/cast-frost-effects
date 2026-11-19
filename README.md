# Cast Frost 素材メーカー

Cast Frost Web(https://cast.rider.tokyo/)の 🎆 演出・⚡ 発動で使う、**背景が透明な素材**を作る Google Colab ノートブックです。made by compact

- 🖼 静止画:背景が透明な PNG(SDXL + LayerDiffuse)
- 🎬 動画:背景が透明な WebM(VP9・yuva420p、Wan-Alpha + LightX2V)。Cast Frost の「＋ 素材を追加」に入る 30MB 以内に自動で収めます

<a href="https://colab.research.google.com/github/compact-git/cast-frost-effects/blob/main/cast-frost-effects-maker.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab で開く"></a>

## 使い方

**使い方サイト:https://rider.tokyo/apps/cast/effects/**

1. 上の「Colab で開く」を押し、[ファイル] → [ドライブにコピーを保存] で自分のドライブにコピーする(または使い方サイトからノートブックをダウンロードして、Colab にアップロードする)
2. [ランタイム] → [ランタイムのタイプを変更] で GPU を選ぶ(動画は A100 がおすすめ)
3. 「🖼 静止画を作る」か「🎬 動画を作る」のセルの ▶ を押す
4. 使い終わったら「🛑 GPU を止める」のセルを押す

Google Colab Pro の契約が必要です。

## ファイル

| 場所 | 中身 |
|---|---|
| `cast-frost-effects-maker.ipynb` | ノートブック本体(プログラムは圧縮して埋め込んであります) |
| `src/notebook/` | ノートブックのプログラム(`cfx_*.py`)と、組み立て用の `build_notebook.py` |
| `src/` | 使い方サイトの元のファイル。`python3 src/build.py` で `effects/` にサイトとノートブックができます |

## ライセンス

使っているモデル・部品は、すべて商用で使えるライセンスのものです(Apache-2.0 / MIT / CreativeML Open RAIL++-M / Open RAIL-M)。一覧は使い方サイトの「使っているモデルとライセンス」にあります。ほかの作者のプログラムをもとにした部分には、元の作者とライセンスを書いています。
