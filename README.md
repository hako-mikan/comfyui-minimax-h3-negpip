[![](https://img.shields.io/badge/lang-English-blue.svg?style=plastic)](README.md)
[![](https://img.shields.io/badge/言語-日本語-green.svg?style=plastic)](README_jp.md)

# ComfyUI MiniMax H3 NegPiP

A ComfyUI custom node that brings NegPiP to MiniMax H3 (audio-video DiT).
Write `(word:-1.0)` in the prompt and the **sign of that token's attention value vector is
flipped**, so the concept is *subtracted* from the result instead of added.

## Usage

1. Add the `Apply MiniMax H3 NegPiP` node.
2. Route **both** `MODEL` and `CLIP` through it, and feed its outputs to the conditioning
   node (`MiniMax H3 Image to Video` / `MiniMax H3 Reference to Video`) and the sampler.
3. Write negative weights in the prompt:

```text
a woman walking on a beach, (blurry photo:-1.0), (anime illustration:-1.2)
```

### Strength

`-1.0` is an exact sign flip (pure subtraction, no amplification), and **anything up to about
`-5` is usable**. Raise it in steps if the effect is too weak. Rewording usually beats
turning the same word up, though.

### Choosing the words - a phrase beats a single word

```text
(red fruit:-3.0)             ← works better than (red:-3.0)
(black dress, clothes:-3.0)  ← works better than (black:-3.0)
```

A negative group is **lifted out of the prompt and encoded on its own**, so it has no
context. A bare `red` stays an ambiguous "red of what?", and subtracting it does not land
where you wanted. **Name the attribute together with what it sits on** and the concept is
grounded, which is what makes the subtraction bite. Listing synonyms (`black dress, clothes`)
helps too, since you do not know which word the model is using internally.

Two to four content words is the sweet spot; you do not need a sentence. Including the
object's own noun does not delete the object, because the appended row stands for the
combined concept ("a red fruit") - but if the object itself starts thinning out, drop the noun.

The console line `... N negative, N appended ...` tells you how many tokens were actually
appended and flipped.

## Time ranges - subtract only during these seconds

Add `@start-end` (in seconds) after the weight and the concept is subtracted **only from the
frames in that window**.

```text
a rainy city street at night, a woman walks past, (neon signs, neon lights:-1.5@2.5-4.0)
```

| Syntax | Meaning |
| --- | --- |
| `(word:-1.2@2.5-4.0)` | subtract from video *and* audio between 2.5s and 4.0s |
| `(word:-1.2@v2.5-4.0)` | video only |
| `(word:-1.2@a2.5-4.0)` | audio only |
| `(word:-1.2@2.5-)` | from 2.5s to the end |
| `(word:-1.2@-2.0)` | from the start to 2.0s |

Write as many as you like, and mix them with plain untimed weights.

**A time ranged group is lifted out of the prompt even when its weight is positive.** A word
left in the prompt is part of the conditioning for the whole clip, and a multiplier can only
make it louder or quieter, never *absent*. An appended row can be silenced: outside its
window it is multiplied by **zero**. That is what makes this read the way it looks:
**To be honest, I can't strictly adhere to specific time requests, so please use this service with the mindset that it would be nice if the timing happens to work out.**
```text
A dress in a desert, (red dress:-3@v-2.5), (green dress:3@v2.5-)
```

Red is subtracted in the first half and green is added in the second; green does not exist in
the first half, and the red subtraction does not act on the second. Untimed negative weights
still apply to the whole clip.
### Inputs

| Input | Default | Meaning |
| --- | --- | --- |
| `value_strength` | 1.0 | Factor on the negative weights. At 1.0, `(word:-1.0)` is an exact sign flip |
| `apply_positive_weights` | true | Also scale the value vectors for positive weights such as `(word:1.3)` |
| `block_start` / `block_end` / `block_stride` | 0 / 999 (= all 50 blocks) / 1 | Which DiT blocks are affected |
| `protect_text_rows` | false | Make untimed weights act on the video/audio rows only, never on the text rows. Try it if a strong negative weight *inverts* (the concept gets stronger). Time ranged weights always behave this way |
| `measure_attention_mass` | false | Diagnostic: once per sampling run, log how much attention mass the appended rows receive (see the last note below) |

## How it works

MiniMax H3 has no cross-attention. It is a single-stream packed DiT where every block runs one
joint self-attention over

```
[text | cond rows | audio | video]
```

with the text rows first, at `[0, text_len)`. **Flipping the value vectors of those rows inside
the block attention is exactly NegPiP** - nothing else is needed.

### Negative words are removed from the prompt

`(black:-2.5)` **deletes the word from the prompt**, encodes `black` on its own, appends that
hidden state to the end of the conditioning and flips the V of that row. This is the original
NegPiP's own scheme.

- **CLIP side**: `MiniMaxH3Tokenizer` in `comfy/text_encoders/minimax.py` tokenizes with
  `disable_weights=True`, so on stock ComfyUI `(word:-1.0)` is not read as a weight at all -
  the brackets are handed to Qwen3-VL as literal text. This node re-enables weight parsing,
  encodes the conditioning itself **at weight 1.0** (scaling a Qwen3-VL hidden state directly
  fights the LLM's own normalisation) and carries the per-token weights along as conditioning
  extras.
- **DiT side**: a `calc_cond_batch` wrapper maps cond uuid to weight array; a `diffusion_model`
  wrapper looks up the weights of the cond being processed. It flags `attn.forward` on the
  selected blocks and swaps `comfy.ldm.minimax.model.optimized_attention` for the duration of
  the call, **multiplying only the V of the rows in question** (splitting the queries as well
  when a time range is present). `Attention.forward` itself - RoPE, RMSNorm, the quantized
  paths - is untouched, and everything is restored the moment the call returns, so the model is
  never left dirty.

## Notes

- With this node in the graph, brackets are **parsed as emphasis syntax**. Escape them as
  `\(` `\)` if you want literal brackets (stock H3 treats brackets as plain text).
- Negative weights on `embedding:` textual inversions are not supported (treated as 1.0).
- Only the `CLIP` / `MODEL` that come *out* of this node are affected. The original CLIP object
  is not modified, so other nodes sharing that CLIP keep stock behaviour.
- A time range splits each block's attention into a few calls. On int8 attention backends the
  K/V quantization runs once per call, which can cost a few percent of speed.