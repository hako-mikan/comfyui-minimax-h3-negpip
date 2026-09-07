[![](https://img.shields.io/badge/lang-English-blue.svg?style=plastic)](README.md)
[![](https://img.shields.io/badge/%E8%A8%80%E8%AA%9E-%E6%97%A5%E6%9C%AC%E8%AA%9E-green.svg?style=plastic)](README_jp.md)

# ComfyUI MiniMax H3 NegPiP

MiniMax H3 で NegPiP を使えるようにするカスタムノード。
プロンプトに `(word:-1.0)` と書くと、そのトークンの **attention value ベクトルの符号が反転**し、
概念が「足される」のではなく「引かれる」ようになる。

## 使い方

1. `Apply MiniMax H3 NegPiP` ノードを追加する。
2. `MODEL` と `CLIP` を **両方**このノードに通し、その出力を
   `MiniMax H3 Image to Video` / `MiniMax H3 Reference to Video` などの
   conditioning ノードと KSampler に繋ぐ。
3. プロンプトに負の重みを書く:

動作テスト。緑のりんごが出たら成功です。
```text
An apple on the table, daytime.
(red fruit:-5)
```

### 強度の目安

`-1.0` がちょうど符号反転（増幅なしの純粋な引き算）で、そこから **`-5` 程度までが実用範囲**です。
効きが足りなければ段階的に上げてください。ただし同じ語を強めるより、書き方を変えるほうが効くことが多いです。

### 語の選び方 — 単語 1 つより句のほうが効く

```text
(red fruit:-3.0)          ← (red:-3.0) より効く
(black dress, clothes:-3.0)  ← (black:-3.0) より効く
```

負の語は**本文から取り出されて単独でエンコードされる**ので、文脈がありません。
裸の `red` は「何の赤なのか」が定まらない曖昧な状態のままで、引いても狙ったところに当たりません。
**消したい属性と、それが乗っている対象を一緒に書く**と概念が接地して、引き算が効くようになります。
`black dress, clothes` のように言い換えを並べるのも有効です（モデルがどの語で表現しているか分からないため）。

目安は内容語 2〜4 語程度。長い説明文にする必要はありません。
対象の名詞を含めても消えてしまわないのは、連結行が「赤い果物」という結合した概念を表しているからですが、
対象そのものが薄くなり始めたら名詞を外してください。

コンソールの `... N negative, N appended ...` で実際に何トークンが連結・反転されたか確認できます。

## 時間指定（この秒数の間だけ引く）

重みの後ろに `@開始-終了`（秒）を付けると、**その時間帯のフレームからだけ**概念を引きます。

```text
a rainy city street at night, a woman walks past, (neon signs, neon lights:-1.5@2.5-4.0)
```

| 書き方 | 意味 |
| --- | --- |
| `(word:-1.2@2.5-4.0)` | 2.5〜4.0 秒の映像と音の両方から引く |
| `(word:-1.2@v2.5-4.0)` | 映像だけ |
| `(word:-1.2@a2.5-4.0)` | 音だけ |
| `(word:-1.2@2.5-)` | 2.5 秒から最後まで |
| `(word:-1.2@-2.0)` | 冒頭から 2.0 秒まで |

1 つのプロンプトに何個でも書けますし、時間指定なしの重みと混ぜられます。

**時間指定つきの語は、正の重みでも本文から取り出されます。**本文に残したままでは、その語は
クリップ全体の条件付けの一部であり続け、倍率では「大きく/小さく」しかできず「無い」にはできないからです。
取り出して連結した行は、**窓の外では倍率 0（存在しない）**として扱われます。だから次のような書き方ができます:
**時間指定は正直な所そこまで効かないので効いたらいいなぐらいの感覚で使ってください。**
```text
A dress in a desert, (red dress:-3@v-2.5), (green dress:3@v2.5-)
```

### 入力

| 入力 | 既定値 | 説明 |
| --- | --- | --- |
| `value_strength` | 1.0 | 負の重みに掛かる係数。1.0 で `(word:-1.0)` がちょうど符号反転 |
| `apply_positive_weights` | true | `(word:1.3)` のような正の重みも value ベクトルに掛ける |
| `block_start` / `block_end` / `block_stride` | 0 / 999 (= 全 50 block) / 1 | 効かせる DiT ブロックの範囲 |
| `protect_text_rows` | false | 時間指定なしの重みも、映像・音声の行にだけ効かせる（text 行自身は引かない）。強い負の重みで**効果が反転する**（概念が濃くなる）ときに試す。時間指定つきの重みは常にこの動作 |
| `measure_attention_mass` | false | 診断用。連結行が集めている attention 質量を 1 ラン 1 回だけログに出す（最後の注意を参照） |

## 仕組み

MiniMax H3 は cross-attention を持たない single-stream の packed DiT で、各ブロックが

```
[text | cond rows | audio | video]
```

という 1 本の系列に対して joint self-attention をかける。text は先頭の `[0, text_len)` に置かれるので、
**そのブロックの value ベクトルだけ符号を反転すれば NegPiP が成立する**。

### 負の語は本文から取り除かれる

`(black:-2.5)` と書くと、その語は**本文から削除され**、`black` だけを単独でエンコードした
hidden state が conditioning の末尾に足され、その行の V が反転されます。本家 NegPiP と同じ方式です。

- **CLIP 側**: `comfy/text_encoders/minimax.py` の `MiniMaxH3Tokenizer` は
  `disable_weights=True` でトークナイズしているため、素の ComfyUI では `(word:-1.0)` は
  重みとして解釈されず、括弧ごとそのままテキストとして Qwen3-VL に渡っている。
  このノードは重み解釈を有効化し、**conditioning 自体は重み 1.0 のまま**エンコードして
  (Qwen3-VL の hidden state を直接スケールすると LLM 側の正規化と喧嘩するため)、
  トークンごとの重みを conditioning の extra として一緒に運ぶ。
- **DiT 側**: `calc_cond_batch` ラッパで cond の uuid → 重み配列 を対応付け、
  `diffusion_model` ラッパで今処理中の cond の重みを引き当てる。対象ブロックの
  `attn.forward` にフラグを立て、`comfy.ldm.minimax.model.optimized_attention` を
  呼び出し中だけ差し替えて、**該当行の V にだけ倍率を掛ける**（時間指定があればクエリも分割する）。
  `Attention.forward` 本体（RoPE・RMSNorm・量子化パス）には触らず、
  呼び出しが終わったら即座に元に戻すのでモデルは汚れない。


## 注意

- このノードを通すと括弧が **強調構文として解釈される**ようになる。リテラルの括弧を
  出したい場合は `\(` `\)` とエスケープすること (素の H3 では括弧はそのまま文字だった)。
- `embedding:` によるテキスト埋め込みへの負の重みは未対応 (重み 1.0 として扱う)。
- ノードを通した `CLIP` / `MODEL` の出力だけが影響を受ける。元の CLIP オブジェクトは
  変更しないので、同じ CLIP を使う他のノードは素の挙動のまま。
- 時間指定を使うとブロックごとの attention 呼び出しが数回に分割される。int8 系の
  attention バックエンドでは K/V の量子化がその回数だけ走るので、数 % 程度遅くなることがある。
