"""ComfyUI custom node: NegPiP for MiniMax H3.

NegPiP (hako-mikan/sd-webui-negpip) makes a token *subtract* its concept instead of
adding it, by flipping the sign of its attention value vector. MiniMax H3 is a
single-stream packed-token DiT: every block runs one joint self-attention over

    [text | cond rows | audio | video]

with the text rows first, at [0, text_len). So flipping the V slice of a text row
inside the block attention is exactly NegPiP, and no cross-attention is needed.

Two halves:

- CLIP side. comfy.text_encoders.minimax.MiniMaxH3Tokenizer tokenizes with
  disable_weights=True, so (word:-1.0) has no effect at all today. The node
  re-enables prompt weighting, encodes with neutral weights (the conditioning is a
  Qwen3-VL hidden state, not a CLIP embedding: scaling it fights the LLM's own
  normalisation), and ships the per-token weights along with the conditioning.
  Negative groups are lifted out of the prompt, encoded on their own and appended as
  extra text rows (the original NegPiP's own scheme). Flipping the word where it
  stands cannot do better than cancelling the weight of its own presence - it has to
  be written to be negated - and Qwen3-VL is causal, so a word's hidden state is
  "the sentence so far, plus that word" rather than the word. Encoding it alone
  keeps the flipped state clean and the prompt free of the word. The appended rows
  are plain text rows: minimax_token_tags is extended to match, and the packed
  layout picks the new text length up on its own.
- DiT side. A calc_cond_batch wrapper maps cond uuid -> weights, and a
  diffusion_model wrapper looks the current cond up and, for the selected blocks,
  multiplies the value vectors of those text rows by the token's weight.

Time ranges: (word:-1.2@2.5-4.0) subtracts the concept only from the part of the
clip between 2.5s and 4.0s. A value multiplier is a key-side edit that every query
sees, so a time range instead splits the attention *by query*: the rows of the
targeted frames attend over the flipped V, every other row attends over the row's
other multiplier. Queries are partitioned, not duplicated, so this costs one
attention pass either way. Time is exact here - it is the packed sequence's own
geometry, not a hint the text encoder has to interpret.

A time ranged group is always lifted, positive weights included, and outside its
window the appended row is multiplied by *zero*: a row left at its own value would
add the concept everywhere else, which is the opposite of what the range asked for.
That makes (red:-3@v-2.5), (green:3@v2.5-) mean what it looks like it means.

The token refiner is deliberately left alone: it runs over the text tokens only
(and, since extra_conds preprocesses the text embeds, outside this wrapper). Flipping
there subtracts a token from its own neighbours before the joint attention subtracts
it again, which makes the response non-monotonic (same finding as NegPiP's Z-Image
context_refiner note).
"""

from __future__ import annotations

import copy
import logging
import math
import numbers
import re
import threading
from typing import Any, NamedTuple

import torch

from comfy.patcher_extension import WrappersMP

KEY = "minimax_h3_negpip"
COND_WEIGHTS_KEY = "minimax_h3_negpip_weights"
COND_RANGES_KEY = "minimax_h3_negpip_ranges"
COND_APPENDED_KEY = "minimax_h3_negpip_appended"
TOKENS_LIFTED_KEY = "minimax_h3_negpip_lifted"
TEXT_ENCODER_NAME = "qwen3vl_32b"
PAD_TOKEN = 151643
TEXT_TOKEN_TAG = 1  # adaLN modality tag of a text row

MAX_STRENGTH = 8.0
FPS = 24.0             # video pixel frames per second
AUDIO_LATENT_FPS = 40.0  # audio latent frames per second
OPEN_END = 1.0e9

# the CLIP and the DiT half are patched by the same node but can be wired apart in the
# graph, which is silent and looks exactly like "NegPiP does nothing". Count what the
# text encoder emitted so the diffusion side can say so out loud.
_STATS = {"emitted": 0}
_LOG_STATE = {}
_MEASURE_STATE = {}
_ACTIVE = threading.local()


def _log_once(tag, message, *args):
    if _LOG_STATE.get(tag) == (message, args):
        return
    _LOG_STATE[tag] = (message, args)
    logging.info(message, *args)


class TimeRange(NamedTuple):
    stream: str  # "both" | "video" | "audio"
    start: float
    end: float


# --------------------------------------------------------------------------------------
# CLIP side
# --------------------------------------------------------------------------------------

# (word:-1.2), and any group carrying a time range: (word:-1.2@2.5-4.0),
# (word:2@v2.5-) video only, (word:-1@a-2.0) audio only, open ended on either side.
# Both are lifted out of the prompt and appended as their own rows; a plain
# (word:1.3) stays where it is and only scales its value vectors.
NEGPIP_GROUP_RE = re.compile(
    r"(?<!\\)\(([^()]*?):\s*(-?\d+(?:\.\d+)?)\s*"
    r"(?:@\s*([vaVA]?)\s*(\d*(?:\.\d+)?)\s*-\s*(\d*(?:\.\d+)?)\s*)?\)")


def _tidy(text: str) -> str:
    """Close the hole a lifted group leaves behind; the LLM does read the punctuation."""
    for _ in range(3):  # neighbouring groups leave several separators behind
        previous = text
        text = re.sub(r"(?:,[ \t]*){2,}", ", ", text)    # doubled separators
        text = re.sub(r"([.!?;:])[ \t]*,", r"\1", text)  # comma orphaned after a sentence end
        text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text == previous:
            break
    return text.strip(" \t,")


def _split_lifted_groups(text: str):
    """Lift the groups that need their own conditioning rows out of the prompt.

    Negative groups, the way the original NegPiP does it: a negated word has to be
    written to be negated, and a causal LLM encodes it as "the sentence so far, plus
    that word", so flipping it in place can only cancel the weight of its own presence.
    Encoding it on its own keeps the main prompt free of the word and the flipped state
    free of the sentence.

    Time ranged groups for the same reason in time: a word that stays in the prompt is
    part of the conditioning for the whole clip and a multiplier can only make it louder
    or quieter, never absent. An appended row can be silenced outside its window.
    """
    lifted = []

    def replace(match):
        phrase = match.group(1).strip()
        weight = float(match.group(2))
        time_range = None
        if match.group(4) is not None or match.group(5) is not None:
            stream = {"v": "video", "a": "audio", "": "both"}[(match.group(3) or "").lower()]
            start = float(match.group(4)) if match.group(4) else 0.0
            end = float(match.group(5)) if match.group(5) else OPEN_END
            time_range = TimeRange(stream, start, max(start, end))
        if weight >= 0.0 and time_range is None:
            return match.group(0)  # plain emphasis, it belongs to the prompt
        if phrase:
            lifted.append((phrase, weight, time_range))
        return ""

    return _tidy(NEGPIP_GROUP_RE.sub(replace, text)), lifted


def _isolated_tokens(inner_tokenizer, phrase: str):
    """Token ids of one negated phrase, tokenized on its own with no prompt weighting."""
    if inner_tokenizer is None:
        return []
    original = getattr(inner_tokenizer.tokenize_with_weights, "_negpip_original_tokenize",
                       inner_tokenizer.tokenize_with_weights)
    tokens = []
    for batch in original(phrase, return_word_ids=False, disable_weights=True):
        for entry in batch:
            if _is_plain_token(entry[0]):
                tokens.append(int(entry[0]))
    return tokens


def _make_tokenize_with_weights(tokenizer, original_function):
    def tokenize_with_weights(text, *args, **kwargs):
        lifted = []
        if isinstance(text, str):
            text, lifted = _split_lifted_groups(text)
        out = original_function(tokenizer, text, *args, **kwargs)
        if not lifted or not isinstance(out, dict):
            return out

        inner = getattr(tokenizer, TEXT_ENCODER_NAME, None)
        entries = []
        for phrase, weight, time_range in lifted:
            tokens = _isolated_tokens(inner, phrase)
            if tokens:
                entries.append({"tokens": tokens, "weight": weight, "range": time_range})
            else:
                logging.warning("MiniMax H3 NegPiP: %r tokenized to nothing, ignored.", phrase)
        if entries:
            out = dict(out)
            out[TOKENS_LIFTED_KEY] = entries
        return out

    return tokenize_with_weights


def _keep_weights(inner_tokenizer):
    """MiniMaxH3Tokenizer hardcodes disable_weights=True on its text segments."""
    original = getattr(inner_tokenizer, "_negpip_original_tokenize", inner_tokenizer.tokenize_with_weights)

    def tokenize_with_weights(text, *args, **kwargs):
        kwargs["disable_weights"] = False
        return original(text, *args, **kwargs)

    tokenize_with_weights._negpip_original_tokenize = original
    return tokenize_with_weights


def _is_plain_token(x: Any) -> bool:
    return (not torch.is_tensor(x)) and isinstance(x, numbers.Integral)


def _strip_weights(section):
    return [(entry[0], 1.0) + tuple(entry[2:]) for entry in section]


def _expanded_weights(section, embeds_info, seq_len: int):
    """Tokenizer (token, weight) pairs -> one weight per encoded hidden state.

    process_tokens() splices embeddings (a vision block expands to many rows) into the
    token stream, so the pair index is not the hidden state index. embeds_info carries
    the final index and size of every splice, in insertion order, which reproduces the
    layout exactly.
    """
    weights = []
    embed_weights = []
    for entry in section:
        if _is_plain_token(entry[0]):
            weights.append(float(entry[1]))
        else:
            embed_weights.append(float(entry[1]))

    infos = list(embeds_info or [])
    # an embed that failed to splice is absent from embeds_info; only trust the
    # per-embed weights when every one of them made it through
    aligned = len(infos) == len(embed_weights)
    for n, info in enumerate(infos):
        index = int(info.get("index", -1))
        size = int(info.get("size", 0))
        if index < 0 or index > len(weights) or size < 0:
            return None
        weights[index:index] = [embed_weights[n] if aligned else 1.0] * size

    if len(weights) < seq_len:  # process_tokens pads unusable embeds at the end
        weights += [1.0] * (seq_len - len(weights))
    if len(weights) != seq_len:
        return None
    return weights


def _encode_lifted(original_encode, name, clip_model, lifted):
    """Encode every lifted phrase on its own, as extra text rows to append.

    All phrases go in one batch, right padded: the encoder is causal, so trailing pad
    rows cannot reach the content tokens, and the padded rows are sliced back off.
    """
    rows = [entry["tokens"] for entry in lifted]
    width = max(len(row) for row in rows)
    pad = int(getattr(clip_model, "special_tokens", {}).get("pad", PAD_TOKEN))
    batch = [[(int(t), 1.0) for t in row] + [(pad, 1.0)] * (width - len(row)) for row in rows]

    out = original_encode({name: batch})
    encoded = out[0]
    if int(encoded.shape[1]) != width * len(rows):
        logging.warning("MiniMax H3 NegPiP: the appended rows came back as %d tokens instead of %d, "
                        "they were dropped.", int(encoded.shape[1]), width * len(rows))
        return None, [], []

    pieces, weights, ranges = [], [], []
    for k, entry in enumerate(lifted):
        length = len(rows[k])
        pieces.append(encoded[:, k * width:k * width + length, :])
        weights += [float(entry["weight"])] * length
        ranges += [entry["range"]] * length
    return torch.cat(pieces, dim=1), weights, ranges


def _extend_token_tags(extra, count):
    tags = extra.get("minimax_token_tags")
    if tags is None or count <= 0:
        return
    tags = tags.view(-1)
    extra["minimax_token_tags"] = torch.cat(
        [tags, torch.full((count,), TEXT_TOKEN_TAG, dtype=tags.dtype, device=tags.device)])


def _make_encode_token_weights(cond_stage_model):
    original = getattr(cond_stage_model, "_negpip_original_encode", cond_stage_model.encode_token_weights)

    def encode_token_weights(token_weight_pairs):
        name = getattr(cond_stage_model, "clip_name", TEXT_ENCODER_NAME)
        sections = token_weight_pairs.get(name) if isinstance(token_weight_pairs, dict) else None
        if not sections:
            return original(token_weight_pairs)

        # encode with neutral weights: the base encoder would apply them to the hidden
        # states by pair index, which is both misaligned (embeds expand) and the wrong
        # operation for an LLM conditioning. NegPiP applies them on the value vectors.
        stripped = dict(token_weight_pairs)
        stripped[name] = [_strip_weights(section) for section in sections]

        clip_model = getattr(cond_stage_model, getattr(cond_stage_model, "clip", name))
        captured = []
        had_own = "process_tokens" in clip_model.__dict__
        original_process_tokens = clip_model.process_tokens

        def process_tokens(tokens, device):
            out = original_process_tokens(tokens, device)
            captured.append(out[3])
            return out

        clip_model.process_tokens = process_tokens
        try:
            out = original(stripped)
        finally:
            if had_own:
                clip_model.process_tokens = original_process_tokens
            else:
                del clip_model.process_tokens

        if len(sections) != 1:
            logging.warning("MiniMax H3 NegPiP: %d prompt sections, weights ignored.", len(sections))
            return out

        cond = out[0]
        weights = _expanded_weights(sections[0], captured[-1] if captured else [], int(cond.shape[1]))
        if weights is None:
            logging.warning("MiniMax H3 NegPiP: could not map the prompt weights onto the conditioning, ignoring them.")
            weights = [1.0] * int(cond.shape[1])
        ranges = [None] * len(weights)

        lifted = token_weight_pairs.get(TOKENS_LIFTED_KEY) if isinstance(token_weight_pairs, dict) else None
        appended = 0
        extra = dict(out[2]) if len(out) > 2 and isinstance(out[2], dict) else {}
        if lifted:
            rows, row_weights, row_ranges = _encode_lifted(original, name, clip_model, lifted)
            if rows is not None:
                cond = torch.cat([cond, rows.to(device=cond.device, dtype=cond.dtype)], dim=1)
                weights += row_weights
                ranges += row_ranges
                appended = len(row_weights)
                _extend_token_tags(extra, appended)

        if all(w == 1.0 for w in weights):
            return out

        extra[COND_WEIGHTS_KEY] = [float(w) for w in weights]
        extra[COND_APPENDED_KEY] = appended
        timed = [[i, r.stream, r.start, r.end] for i, r in enumerate(ranges) if r is not None]
        if timed:
            extra[COND_RANGES_KEY] = timed
        _STATS["emitted"] += 1
        logging.info("MiniMax H3 NegPiP: %d weighted token(s) of %d, %d negative, %d appended, %d time ranged.",
                     sum(1 for w in weights if w != 1.0), int(cond.shape[1]),
                     sum(1 for w in weights if w < 0.0), appended, len(timed))
        return (cond, out[1], extra)

    encode_token_weights._negpip_original_encode = original
    return encode_token_weights


def patch_clip(clip):
    tokenizer = getattr(clip, "tokenizer", None)
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    if (cond_stage_model is None or tokenizer is None
            or not hasattr(tokenizer, TEXT_ENCODER_NAME) or not hasattr(cond_stage_model, TEXT_ENCODER_NAME)):
        raise RuntimeError("MiniMax H3 NegPiP needs the MiniMax H3 CLIP (CLIPLoader / DualCLIPLoader type minimax_h3).")

    new_clip = clip.clone() if hasattr(clip, "clone") else copy.copy(clip)
    # shallow copies, so the CLIP object the rest of the graph shares keeps stock behaviour
    new_clip.tokenizer = copy.copy(tokenizer)
    inner_tokenizer = copy.copy(getattr(tokenizer, TEXT_ENCODER_NAME))
    inner_tokenizer.tokenize_with_weights = _keep_weights(getattr(tokenizer, TEXT_ENCODER_NAME))
    setattr(new_clip.tokenizer, TEXT_ENCODER_NAME, inner_tokenizer)
    # bind the class function to the copy, so it reaches the patched inner tokenizer
    new_clip.tokenizer.tokenize_with_weights = _make_tokenize_with_weights(
        new_clip.tokenizer, type(tokenizer).tokenize_with_weights)

    new_clip.cond_stage_model = copy.copy(cond_stage_model)
    new_clip.cond_stage_model.encode_token_weights = _make_encode_token_weights(cond_stage_model)
    return new_clip


# --------------------------------------------------------------------------------------
# packed sequence geometry
# --------------------------------------------------------------------------------------

def _minimax_module():
    import comfy.ldm.minimax.model as minimax_model
    return minimax_model


def _is_minimax_h3(dm: Any) -> bool:
    try:
        return isinstance(dm, _minimax_module().MiniMaxH3Model)
    except Exception:
        return False


def _layout_for_call(dm, payload, context, x):
    """The packed layout this forward will build, to read segment offsets off."""
    minimax_model = _minimax_module()
    video, audio = x[0], x[1]
    text_len = int(context.shape[1])
    latent_t = int(video.shape[2])
    lat_h = (int(video.shape[3]) + 1) // 2 * 2
    lat_w = (int(video.shape[4]) + 1) // 2 * 2
    audio_t = int(audio.shape[-1])
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)

    layout = payload.get("layout")
    if layout is None or layout.signature != signature:
        layout = minimax_model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                            keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    return layout, latent_t, audio_t


def _video_frame_starts(latent_t):
    """Pixel frame each latent frame starts at: FRAME_PER_TOKEN repeats 1,4,4,4,4."""
    per_token = _minimax_module().FRAME_PER_TOKEN
    starts = []
    acc = 0
    for k in range(latent_t):
        starts.append(acc)
        acc += per_token[k % len(per_token)]
    return starts, per_token


def _range_rows(time_range: TimeRange, layout, latent_t, audio_t):
    """Seconds -> the packed rows of the generated streams inside that window."""
    segments = {kind: (a, b) for a, b, kind in layout.segments}
    rows = []

    video = segments.get("video")
    if time_range.stream in ("both", "video") and video is not None and latent_t > 0:
        va, vb = video
        frame_rows = (vb - va) // latent_t
        starts, per_token = _video_frame_starts(latent_t)
        first = time_range.start * FPS
        last = time_range.end * FPS
        hit = [k for k in range(latent_t)
               if starts[k] + per_token[k % len(per_token)] > first and starts[k] < last]
        if hit:
            rows.append((va + hit[0] * frame_rows, va + (hit[-1] + 1) * frame_rows))

    audio = segments.get("audio")
    if time_range.stream in ("both", "audio") and audio is not None and audio_t > 0:
        aa, _ = audio
        i0 = max(0, int(math.floor(time_range.start * AUDIO_LATENT_FPS)))
        i1 = min(audio_t, int(math.ceil(time_range.end * AUDIO_LATENT_FPS)))
        if i1 > i0:
            # channel major: [ch0 t0..T-1 | ch1 t0..T-1]
            rows.append((aa + i0, aa + i1))
            rows.append((aa + audio_t + i0, aa + audio_t + i1))
    return rows


def _complement_rows(time_range: TimeRange, layout, latent_t, audio_t, seq_len):
    """Every packed row the range does *not* cover, including text and condition rows.

    An appended row exists only because the prompt asked for it, so outside its window
    it has to contribute nothing at all - leaving it at its own value would add the
    concept everywhere else, which is the opposite of what the range asked for.
    """
    inside = sorted(_range_rows(time_range, layout, latent_t, audio_t))
    rows = []
    cursor = 0
    for a, b in inside:
        if a > cursor:
            rows.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < seq_len:
        rows.append((cursor, seq_len))
    return rows


def _chunks_by_active_set(group_rows, seq_len):
    """Partition [0, seq_len) into runs where the set of active timed groups is constant."""
    bounds = {0, seq_len}
    for rows in group_rows:
        for a, b in rows:
            bounds.add(max(0, min(a, seq_len)))
            bounds.add(max(0, min(b, seq_len)))
    ordered = sorted(bounds)
    chunks = []
    for a, b in zip(ordered, ordered[1:]):
        if b <= a:
            continue
        active = frozenset(i for i, rows in enumerate(group_rows)
                           if any(s <= a and b <= t for s, t in rows))
        if chunks and chunks[-1][2] == active:
            chunks[-1] = (chunks[-1][0], b, active)
        else:
            chunks.append((a, b, active))
    return chunks


# --------------------------------------------------------------------------------------
# DiT side
# --------------------------------------------------------------------------------------

def _multipliers(weights, ranges, appended, cfg):
    """Per text row value multiplier, split into always-on and row restricted groups.

    A group is keyed by ("range", TimeRange) - the rows the multiplier applies to - or
    ("silence", TimeRange), the rows where an appended token has to read as absent.
    """
    value_strength = float(cfg.get("value_strength", 1.0))
    apply_positive = bool(cfg.get("apply_positive_weights", True))
    # protect_text_rows routes the untimed tokens through the same query split as a
    # whole-clip range, so the flipped V reaches the generated streams but not the text
    # rows themselves. Without it a flipped token keeps subtracting from its own residual
    # in all 50 blocks, which is what makes strong weights turn non-monotonic.
    protect = bool(cfg.get("protect_text_rows", False))
    everything = TimeRange("both", 0.0, OPEN_END)
    first_appended = len(weights) - max(0, int(appended or 0))

    glob = ([], [])
    groups = {}
    for i, w in enumerate(weights):
        w = float(w)
        is_appended = i >= first_appended
        if w == 1.0 and not is_appended:
            continue
        if w < 0.0:
            m = w * value_strength
        elif is_appended or apply_positive:
            m = w  # an appended row is there on purpose, apply_positive_weights is about the prompt
        else:
            continue

        time_range = ranges[i] if ranges else None
        if time_range is None and not protect:
            glob[0].append(i)
            glob[1].append(m)
            continue

        window = time_range or everything
        target = groups.setdefault(("range", window), ([], []))
        target[0].append(i)
        target[1].append(m)
        if is_appended:
            silence = groups.setdefault(("silence", window), ([], []))
            silence[0].append(i)
            silence[1].append(0.0)
    return glob, list(groups.items())


def _weights_for_call(cfg, transformer_options, context):
    by_uuid = cfg.get("_by_uuid") or {}
    if not by_uuid:
        if _STATS["emitted"]:
            _log_once("orphan", "MiniMax H3 NegPiP: the prompt carried weights but none reached the model. "
                                "Check that the MODEL output of the NegPiP node is the one feeding the sampler "
                                "(a LoRA/patch node in between is fine, a bypassed branch is not).")
        return None, None, 0
    rows = [by_uuid.get(str(u)) for u in (transformer_options.get("uuids") or [])]
    rows = [r for r in rows if r]
    if not rows:
        return None, None, 0
    weights, ranges, appended = rows[0]
    text_len = int(context.shape[1])
    if len(weights) != text_len:
        logging.warning("MiniMax H3 NegPiP: the conditioning is %d tokens but %d weights were carried, skipping.",
                        text_len, len(weights))
        return None, None, 0
    return weights, ranges, appended


def _index_tensors(state, device, dtype):
    """Position/multiplier tensors for the global set and every timed group."""
    cached = state["cache"].get((device, dtype))
    if cached is None:
        def build(positions, mults):
            return (torch.tensor(positions, device=device, dtype=torch.long),
                    torch.tensor(mults, device=device, dtype=dtype).view(1, 1, -1, 1))

        cached = {
            "global": build(*state["global"]) if state["global"][0] else None,
            "groups": [build(positions, mults) for positions, mults in state["groups"]],
            "timed": (torch.tensor(state["timed_positions"], device=device, dtype=torch.long)
                      if state["timed_positions"] else None),
        }
        state["cache"][(device, dtype)] = cached
    return cached


def _measure_attention_mass(q, k, state):
    """How much attention mass the appended rows actually receive.

    Zeroing a row's V removes its content but leaves its key in the softmax, so every
    other token's contribution is scaled by (1 - that mass). This measures the error
    that silencing leaves behind. One streaming pass over the keys for a sample of
    video queries: cheap compared to the attention it sits next to.
    """
    heads, seq_len, dim = int(q.shape[1]), int(k.shape[2]), int(q.shape[3])
    scale = 1.0 / math.sqrt(dim)
    lo, hi = state.get("video_span", (0, seq_len))
    samples = min(128, max(1, hi - lo))
    picks = torch.linspace(lo, hi - 1, samples, device=q.device).long()
    qs = q[0, :, picks, :].to(torch.float32)  # [heads, samples, dim]

    running_max = None
    running_sum = None
    for start in range(0, seq_len, 2048):
        block = k[0, :, start:start + 2048, :].to(torch.float32)
        logits = torch.einsum("hnd,hkd->hnk", qs, block) * scale
        block_max = logits.amax(dim=-1)
        if running_max is None:
            running_max = block_max
            running_sum = torch.exp(logits - block_max.unsqueeze(-1)).sum(-1)
        else:
            merged = torch.maximum(running_max, block_max)
            running_sum = (running_sum * torch.exp(running_max - merged)
                           + torch.exp(logits - merged.unsqueeze(-1)).sum(-1))
            running_max = merged
    lse = running_max + torch.log(running_sum)  # [heads, samples]

    def mass(rows):
        if not rows:
            return None
        index = torch.tensor(rows, device=k.device, dtype=torch.long)
        logits = torch.einsum("hnd,hkd->hnk", qs, k[0, :, index, :].to(torch.float32)) * scale
        return torch.exp(logits - lse.unsqueeze(-1)).sum(-1)  # [heads, samples]

    text_len = int(state.get("text_len", 0))
    measured = {}
    report = []
    for label, rows in (("appended rows", list(state["timed_positions"])),
                        ("all text rows", list(range(text_len)))):
        values = mass(rows)
        if values is None:
            continue
        # the mean hides the shape: prompt reading is concentrated in a few heads, so
        # report what the worst head does on average, not just the worst single query
        per_head = values.mean(dim=1)
        flat = values.flatten().to(torch.float32)
        measured[label] = {
            "mean": float(values.mean()),
            "p90": float(torch.quantile(flat, 0.90)),
            "p99": float(torch.quantile(flat, 0.99)),
            "max": float(values.amax()),
            "worst_head_mean": float(per_head.amax()),
            "heads_over_1pct": int((per_head > 0.01).sum()),
            "heads": int(per_head.shape[0]),
        }
        m = measured[label]
        report.append("%s (%d rows): mean %.4f%% p90 %.4f%% p99 %.3f%% max %.2f%% | "
                      "worst head mean %.3f%%, %d/%d heads over 1%%"
                      % (label, len(rows), m["mean"] * 100, m["p90"] * 100, m["p99"] * 100,
                         m["max"] * 100, m["worst_head_mean"] * 100, m["heads_over_1pct"], m["heads"]))
    logging.info("MiniMax H3 NegPiP measurement, attention mass from %d sampled video queries over "
                 "%d keys: %s", samples, seq_len, " | ".join(report))
    return measured


def _make_attention_hook(original_attention, container_class):
    def attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options={}, **kwargs):
        state = getattr(_ACTIVE, "state", None)
        if state is None:
            return original_attention(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                                      transformer_options=transformer_options, **kwargs)

        wrapped = isinstance(q, container_class)
        qt = q.take() if wrapped else q
        kt = k.take() if wrapped else k
        vt = v.take() if wrapped else v

        def wrap(t):
            return container_class(t) if wrapped else t

        def run(query, values):
            return original_attention(wrap(query), wrap(kt), wrap(values), heads, mask=mask,
                                      skip_reshape=skip_reshape, transformer_options=transformer_options, **kwargs)

        if state.get("measure") and not _MEASURE_STATE.get("done") and state["timed_positions"]:
            _MEASURE_STATE["done"] = True
            try:
                _measure_attention_mass(qt, kt, state)
            except Exception:
                logging.exception("MiniMax H3 NegPiP: the attention mass measurement failed.")

        index = _index_tensors(state, vt.device, vt.dtype)
        if index["global"] is not None:
            positions, mults = index["global"]
            vt[:, :, positions, :] *= mults

        chunks = state["chunks"]
        usable = (index["timed"] is not None and chunks and mask is None and skip_reshape
                  and qt.ndim == 4 and int(qt.shape[2]) == state["seq_len"])
        if not usable:
            if index["timed"] is not None:
                _log_once("noquery", "MiniMax H3 NegPiP: this attention call cannot be split by query "
                                     "(seq %s), the time ranges are ignored here.", tuple(qt.shape))
            return run(qt, vt)

        # the timed rows are a handful of text tokens: keep them and restore per chunk,
        # so every chunk sees the original V plus only its own groups' multipliers
        timed = index["timed"]
        pristine = vt[:, :, timed, :].clone()
        outs = []
        for a, b, active in chunks:
            vt[:, :, timed, :] = pristine
            for group in active:
                positions, mults = index["groups"][group]
                vt[:, :, positions, :] *= mults
            outs.append(run(qt[:, :, a:b, :], vt))
        vt[:, :, timed, :] = pristine
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    return attention


def _patch_attention_module(attn, state):
    """Flag the blocks NegPiP applies to; the attention hook reads the flag."""
    if getattr(attn, "_negpip_patched", False):
        return None
    original_forward = attn.forward
    had_own = "forward" in attn.__dict__

    def forward(*args, **kwargs):
        previous = getattr(_ACTIVE, "state", None)
        _ACTIVE.state = state
        try:
            return original_forward(*args, **kwargs)
        finally:
            _ACTIVE.state = previous

    attn.forward = forward
    attn._negpip_patched = True
    return (attn, original_forward, had_own)


def _restore_modules(installed):
    for module, original_forward, had_own in installed:
        if had_own:
            module.forward = original_forward
        else:
            try:
                del module.forward
            except AttributeError:
                pass
        try:
            del module._negpip_patched
        except AttributeError:
            pass


def _block_indices(cfg, count):
    start = max(0, int(cfg.get("block_start", 0)))
    end = min(count - 1, int(cfg.get("block_end", count - 1)))
    stride = max(1, int(cfg.get("block_stride", 1)))
    return range(start, end + 1, stride)


def diffusion_model_wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
    cfg = (transformer_options or {}).get(KEY) or {}
    if not cfg.get("enabled", False):
        return executor(x, timestep, context, transformer_options, **kwargs)

    dm = getattr(executor, "class_obj", None)
    if not _is_minimax_h3(dm):
        logging.warning("MiniMax H3 NegPiP: this is not a MiniMax H3 DiT, doing nothing.")
        return executor(x, timestep, context, transformer_options, **kwargs)

    weights, ranges, appended = _weights_for_call(cfg, transformer_options or {}, context)
    if weights is None:
        return executor(x, timestep, context, transformer_options, **kwargs)

    glob, groups = _multipliers(weights, ranges, appended, cfg)
    if not glob[0] and not groups:
        return executor(x, timestep, context, transformer_options, **kwargs)

    layout, latent_t, audio_t = _layout_for_call(dm, kwargs.get("minimax_payload") or {}, context, x)
    seq_len = int(layout.seq_len)
    group_rows = []
    for (kind, window), _ in groups:
        if kind == "range":
            rows = _range_rows(window, layout, latent_t, audio_t)
            if not rows:
                logging.warning("MiniMax H3 NegPiP: %s covers no frame of this clip.", str(window))
        else:
            rows = _complement_rows(window, layout, latent_t, audio_t, seq_len)
        group_rows.append(rows)

    state = {
        "seq_len": seq_len,
        "text_len": int(context.shape[1]),
        "video_span": next(((a, b) for a, b, kind in layout.segments if kind == "video"), (0, seq_len)),
        "measure": bool(cfg.get("measure_attention_mass", False)),
        "global": glob,
        "groups": [mults for _, mults in groups],
        "timed_positions": sorted({i for _, (positions, _) in groups for i in positions}),
        "chunks": _chunks_by_active_set(group_rows, seq_len) if groups else [],
        "cache": {},
    }

    blocks = _block_indices(cfg, len(dm.blocks))
    _log_once("applied", "MiniMax H3 NegPiP: %d always-on and %d row restricted token(s) of %d "
                         "(%d appended), %d/%d blocks, %d attention chunk(s).",
              len(glob[0]), len(state["timed_positions"]), int(context.shape[1]), int(appended or 0),
              len(blocks), len(dm.blocks), len(state["chunks"]))

    minimax_model = _minimax_module()
    original_attention = minimax_model.optimized_attention
    minimax_model.optimized_attention = _make_attention_hook(
        original_attention, minimax_model.AttentionTensorContainer)
    installed = []
    try:
        for i in blocks:
            entry = _patch_attention_module(dm.blocks[i].attn, state)
            if entry is not None:
                installed.append(entry)
        return executor(x, timestep, context, transformer_options, **kwargs)
    finally:
        _restore_modules(installed)
        minimax_model.optimized_attention = original_attention


def _collect_weights_by_uuid(conds) -> dict:
    out = {}
    if not isinstance(conds, list):
        return out
    for group in conds:
        if not isinstance(group, list):
            continue
        for cond in group:
            if not isinstance(cond, dict):
                continue
            weights = cond.get(COND_WEIGHTS_KEY)
            cond_uuid = cond.get("uuid")
            if not weights or cond_uuid is None:
                continue
            try:
                values = [float(w) for w in weights]
            except (TypeError, ValueError):
                continue
            ranges = [None] * len(values)
            for item in cond.get(COND_RANGES_KEY) or []:
                try:
                    position, stream, start, end = int(item[0]), str(item[1]), float(item[2]), float(item[3])
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= position < len(ranges):
                    ranges[position] = TimeRange(stream, start, end)
            try:
                appended = int(cond.get(COND_APPENDED_KEY) or 0)
            except (TypeError, ValueError):
                appended = 0
            out[str(cond_uuid)] = (values, ranges, appended)
    return out


def outer_sample_wrapper(executor, *args, **kwargs):
    # one log line per sampling run: the per-step lines are deduplicated, and without
    # this reset a repeat run with the same settings would print nothing at all
    _LOG_STATE.clear()
    _MEASURE_STATE.clear()
    return executor(*args, **kwargs)


def calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
    by_uuid = _collect_weights_by_uuid(conds)
    if by_uuid:
        # the diffusion wrapper only gets transformer_options, carry the per cond weights there
        model_options = dict(model_options)
        transformer_options = dict(model_options.get("transformer_options", {}))
        cfg = dict(transformer_options.get(KEY, {}))
        cfg["_by_uuid"] = by_uuid
        transformer_options[KEY] = cfg
        model_options["transformer_options"] = transformer_options
    return executor(model, conds, x_in, timestep, model_options)


def patch_model(model, cfg):
    patched = model.clone()

    # prepare_model_patcher() merges the patcher's wrappers into the sampling
    # transformer_options, which is where both wrapper points read them from
    patched.model_options.setdefault("transformer_options", {})[KEY] = cfg
    for wrapper_type, wrapper in ((WrappersMP.OUTER_SAMPLE, outer_sample_wrapper),
                                  (WrappersMP.CALC_COND_BATCH, calc_cond_batch_wrapper),
                                  (WrappersMP.DIFFUSION_MODEL, diffusion_model_wrapper)):
        if hasattr(patched, "remove_wrappers_with_key"):
            patched.remove_wrappers_with_key(wrapper_type, KEY)
        patched.add_wrapper_with_key(wrapper_type, KEY, wrapper)
    return patched


# --------------------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------------------

class ApplyMiniMaxH3NegPiP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "value_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": MAX_STRENGTH, "step": 0.05,
                                             "tooltip": "Multiplier on the negative weights. 1.0 makes (word:-1.0) an exact value flip."}),
                "apply_positive_weights": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Also scale the value vectors of tokens with a positive weight, e.g. (word:1.3). MiniMax H3 ignores prompt weights entirely without this node."}),
            },
            "optional": {
                "block_start": ("INT", {"default": 0, "min": 0, "max": 999}),
                "block_end": ("INT", {"default": 999, "min": 0, "max": 999,
                                      "tooltip": "Last DiT block to affect, clamped to the model's block count."}),
                "block_stride": ("INT", {"default": 1, "min": 1, "max": 16}),
                "protect_text_rows": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Subtract only from the generated video/audio rows, never from the text rows themselves. Try this when a strong negative weight inverts (the concept gets stronger instead of weaker). Time ranged weights always work this way."}),
                "measure_attention_mass": ("BOOLEAN", {"default": False,
                                                       "tooltip": "Diagnostic: log how much attention mass the appended rows receive, once per sampling run. Silencing a row zeroes its value but leaves its key in the softmax, so this is the residual error that leaves behind."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = "model/conditioning/minimax"
    DESCRIPTION = ("NegPiP for MiniMax H3: (word:-1.0) subtracts the concept by flipping the token's "
                   "attention value vector, (word:-1.0@2.5-4.0) only between 2.5s and 4.0s "
                   "(@v = video only, @a = audio only). Connect both MODEL and CLIP through this node, "
                   "before the MiniMax H3 conditioning node.")

    def apply(self, model, clip, value_strength=1.0, apply_positive_weights=True,
              measure_attention_mass=False, protect_text_rows=False,
              block_start=0, block_end=999, block_stride=1):
        if block_end < block_start:
            block_start, block_end = block_end, block_start
        cfg = {
            "enabled": True,
            "value_strength": max(0.0, min(MAX_STRENGTH, float(value_strength))),
            "apply_positive_weights": bool(apply_positive_weights),
            "measure_attention_mass": bool(measure_attention_mass),
            "protect_text_rows": bool(protect_text_rows),
            "block_start": int(block_start),
            "block_end": int(block_end),
            "block_stride": int(block_stride),
        }
        return patch_model(model, cfg), patch_clip(clip)


NODE_CLASS_MAPPINGS = {"ApplyMiniMaxH3NegPiP": ApplyMiniMaxH3NegPiP}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyMiniMaxH3NegPiP": "Apply MiniMax H3 NegPiP"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
