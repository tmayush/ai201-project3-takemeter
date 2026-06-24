# TakeMeter — Planning

## Community

Valorant: r/VALORANT and r/ValorantCompetitive. I picked it because I've played for
about two years, and knowing the community is what lets me label consistently. That's
the part of this project that's easy to get wrong.

## What we're measuring

Discourse health, not whether a take is factually right. Each comment gets sorted by how
it contributes to the conversation, on a constructive-to-toxic axis with neutral in the
middle. The question is how something is said and whether it adds anything, not whether
the opinion is correct.

## Label taxonomy (3 labels)

| Label | Int | Definition |
|---|---|---|
| `constructive` | 0 | Adds value: reasoned argument, helpful advice (lineups, strats, settings), thoughtful analysis, or a respectful answer — even when disagreeing. |
| `neutral` | 1 | On-topic but low-substance and non-hostile: casual chatter, jokes, simple reactions, one-line facts, plain questions. |
| `toxic` | 2 | Hostile or corrosive: insults, flaming, personal attacks, teammate-blaming as abuse, slurs, contemptuous dismissal. |

Mutually exclusive: a comment gets exactly one label. Target distribution >=20% each.

### Edge-case rules (decide once, apply to all 200)

- **Helpful point, rude tone** -> if the insult dominates, `toxic`; if there's a
  genuine helpful core with only a mild edge, `constructive`.
- **Undirected venting** ("I hate this map", "ranked is miserable") -> `neutral`
  (frustration, but no hostility aimed at a person), unless it contains slurs.
- **Sarcasm / disagreement** -> not automatically toxic. Respectful disagreement with
  reasoning is `constructive`; a sarcastic put-down is `toxic`.
- **Pure noise / off-topic / emoji-only** -> `neutral` (no separate "other" bucket).

> Distribution risk: `toxic` will probably be the smallest class, since the subreddits
> get moderated. Plan to oversample heated threads (match results, ranked rage, nerf
> arguments) to get toxic over 20%. (In the end I hit the target by downsampling the
> other two classes instead; see Resolved decisions below.)

## Data collection

- Sources: Reddit comments from the two subreddits. Strip quote blocks (`>`), `@user`
  mentions, and markdown so the model doesn't pick up formatting as a signal. (The
  original plan also listed YouTube comments; I didn't end up needing them.)
- Target: at least 200 examples, roughly balanced across the three labels.
- Context: standalone comments only, just the comment text. Whatever the model sees at
  inference has to match what was there at labeling time, so I'm not splicing in parent
  context. (I considered an Option B where you bake short parent context into the text
  with a `[CONTEXT] ... [REPLY] ...` delimiter, but went with the simpler version.)

## CSV schema

`data/valorant_takes.csv` with two columns:

```
text,label
"<comment text, cleaned>",constructive
"<comment text, cleaned>",neutral
"<comment text, cleaned>",toxic
```

`label` strings must match the `LABEL_MAP` keys in the notebook exactly.

## Stretch features (update this doc before starting each)

- [ ] Inter-annotator reliability (have a friend label 30+, report agreement)
- [ ] Confidence calibration
- [ ] Error pattern analysis
- [ ] Deployed interface

## Hard-to-label cases (need >=3 for README)

1. Self-directed insult word. "my adr is dogshit lmfao hopefully ill improve" has an
   insult word in it but it's aimed at the author's own stats, and they're agreeing with
   someone. I called it `neutral`, not `toxic`. What decides the label is who the
   negativity points at, not the vocabulary.
2. Venting that reads aggressive. "I hate passive value. I hate lurkers. I hate being
   shot in the back…" sounds angry but nobody's getting attacked, so it's `neutral` by
   the venting rule. The bootstrap model called it `toxic`.
3. Sarcasm with positive words. "go work for valorant you're clearly talented" is all
   nice words but it's a sarcastic shot, so `toxic`. Nothing in the text marks it, which
   makes it hard for me and for the model.

## Resolved decisions (were TBD)

- Context (Option A): standalone comments only. The model sees exactly the comment text,
  with no parent context baked in.
- Data source: Reddit's own API (public and OAuth) was IP-blocked, so I pulled the
  comments from the Arctic Shift public archive instead. Same data, different host.
- Labeling: gpt-4o-mini drafted the labels on this rubric, then I reviewed them. I used a
  different model than the Groq `llama-3.3-70b` baseline on purpose, so the baseline
  isn't being graded against its own labels.
- **Balancing:** `toxic` came out the minority (~16%); majority classes downsampled to
  130 each (seed 42) to clear the ≥20%/class target → final 361 examples (36/36/28).
