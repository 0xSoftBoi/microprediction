# Microprediction — Revival Go/No-Go Brief

*Companion to `RETHINK.md`. Question asked: should we "take over and be the main repo," fundamentally
rebuild microprediction, or rewrite it from scratch (possibly in a faster language)? This brief
answers "should we?" before "how?". Date: 2026-06-29. All claims labeled **[verified]** (direct probe
or primary source) or **[inferred]**.*

---

## Verdict

**No — do not revive microprediction as-was, and do not frame this as a "takeover."** The live network
is defunct, adoption was always niche and single-maintainer, and — decisively — **the founder himself
published a candid post-mortem explaining why it failed, then left to build the funded, real-money
version of the same idea (CrunchDAO).** The failure modes were demand-side and incentive-side, none of
which a rewrite (in any language) fixes. Recommendation and salvage paths at the bottom.

---

## 1. The network is dead [verified]

- `api.microprediction.org` → HTTP 404, serves a **PythonAnywhere "Coming Soon" placeholder**. Every
  former live endpoint (`/live/*`, `/budgets`, `/prizes`) 404s. The hosted app is gone, not erroring.
- `www.microprediction.org` (the dashboard) → **301-redirects to `github.com/microprediction`**. The
  standalone site no longer exists. `config.microprediction.org` does not resolve.
- `www.monteprediction.com` (the founder's newer weekend game) **is** live — confirming the lights are
  off specifically on *microprediction*, not on the founder.

## 2. It was never big, and it's a single-maintainer project [verified]

- Main repo `microprediction/microprediction`: **~376★ / 64 forks / 15 contributors**, but **~93% of
  commits are Peter Cotton's**. Recent commits (June 2026) are **README-only touch-ups**; last
  substantive code was **v1.2.0, May 2023**. PyPI: **~650 downloads/month**, frozen since May 2023.
- No public Slack member count, no LinkedIn company page (only Cotton's personal ~25k followers), no
  published count of live crawlers/streams. Peak was qualitatively "hundreds of streams" with tiny
  (~$4k/mo) prize pools. The oft-cited "billion predictions" was the **predecessor system he built for
  Intech Investments**, not the public network. [verified/inferred]
- Independent tutorials are thin; the signature **z-curve/copula** idea got **no independent
  adoption**. The ecosystem's one real external footprint is unrelated: `precise`'s Schur-complement
  portfolio method, upstreamed into **`skfolio`** + an arXiv paper. [verified]

## 3. The founder moved on — to the version that works [verified]

- Peter Cotton is now **Chief Scientific Officer at Crunch Lab / CrunchDAO** (announced Sept 2024),
  after Chief Data Scientist at ExodusPoint (2023). His active builds are **MontePrediction** (public
  distributional-forecasting game) and **midone/skaters** (CrunchDAO tournaments).
- Oct 2023 post *"So Long and Thanks for all the Phish"* announced he was stepping back from
  promoting microprediction. He framed the *idea* as durable ("not killable") but **disengaged from
  operating it**.

## 4. The founder's own post-mortem — why it stalled [verified, primary source]

From his retrospective and corroborating docs/third-party walkthroughs, the causes were **demand- and
incentive-side, not technical**:

1. **Onboarding wall.** To get a write-key you **proof-of-work "mine" a MUID** — the setup script
   itself warns it "takes a long time, sorry" (hours for a hard key). A multi-hour barrier *before*
   any value. Difficulty 12+ just to create a stream.
2. **No staking / no real money.** He explicitly admits "game-theoretic challenges introduced by
   **lack of staking**" — nothing at risk to enforce honest, effortful prediction. Prizes were
   trivial. (This is *exactly* what Numerai's NMR staking solves.)
3. **Demand couldn't form.** A third-party walkthrough (Databutton) flags: you **can't sponsor your
   own streams**, and streams are **just names with no context** — so a modeler can't be told what to
   predict or be paid to care.
4. **Identity-protective audience.** His words: "Quants and data scientists **don't want to know that
   someone else can improve their model**." No organic viral loop; adoption would need top-down
   mandate.
5. **Narrow scope.** Optimized for short-horizon equity-style streams; weak on sparse/reactive tasks.

> The near-total absence of Hacker News / Reddit critique threads is itself a finding: the project
> never generated enough discussion to attract critics. Niche, not contested.

## 5. The living competitors already solved exactly what it lacked [verified]

| Project | Scale (2025–26) | What it fixed that microprediction didn't |
|---|---|---|
| **Numerai / Signals** | **$30M Series C at ~$500M valuation (Nov 2025); ~$550M AUM**; JPMorgan capacity | **Real capital + NMR token staking** — the missing incentive/game-theory layer |
| **CrunchDAO / Crunch Lab** (where Cotton went) | 10k+ ML engineers, USDC payouts, **ADIA Lab** partnership; raised ~$10M | Institutional problem-sponsors + real payouts + curated tournaments |
| **Polymarket / Kalshi** | **~$220B / ~$238B** full-year 2025 volume; ~$21–24B/mo in 2026 | Real-money markets with mainstream (sports/politics) demand |
| **Metaculus** | ~3.97M predictions; quarterly **AI-forecasting benchmark** tournaments | Mission funding + an AI-vs-human benchmark flywheel |
| **Mantic** (2025, $4M pre-seed) | Top-10 in a Metaculus cup | **LLM-orchestrated** forecasting — the current frontier, not human-modeler crowdsourcing |

Reviving microprediction means competing head-on with these — including the one its own founder
joined. The frontier has also shifted toward **LLM-orchestrated** forecasting.

---

## 6. What this means for the original questions

- **"Take over and be the main repo"** — there is no live network to take over; it's MIT-licensed and
  dormant. "Main repo" is earned by running what people connect to, and nobody connects anymore.
  Reviving it is **starting a startup on a dead brand**, not seizing a crown.
- **"Rewrite in a more performant language"** — the bottleneck was never performance; it was
  onboarding, incentives, and demand. A faster client/server makes a dead marketplace faster, not
  alive. (And the *client* should stay Python regardless — it's I/O-bound and its value is the ML
  ecosystem.) See `RETHINK.md` §5–6.

## 7. Recommendation — pick a direction, not a rewrite

Ranked by leverage-to-effort:

1. **Build *on* the winners, don't revive.** If the goal is a real product/return in this space, go
   where capital and users already are — **CrunchDAO / Numerai Signals** — using the modern
   forecasting stack in `RETHINK.md` §3 (quantile output, conformal calibration, foundation-model
   baselines). Highest expected value, lowest infra burden.
2. **Ship a focused modern forecasting library.** The one thing that *travelled* here was a sharp,
   well-packaged algorithm library (`precise` → `skfolio`). A clean, foundation-model-era library for
   **online distributional forecasting + conformal calibration** is a tractable, genuinely useful,
   *yours* artifact. Strong portfolio/OSS play.
3. **Client-v2 + mini-network as an OSS showcase** (`RETHINK.md` Option B) — only if you specifically
   want the "open network" artifact for learning/portfolio, with eyes open that it's a demo, not a
   land-grab.
4. **Full revival of the network** — **not recommended.** It's the most effort for the least leverage:
   dead brand, niche idea, demand-side failure modes unaddressed by engineering, and direct
   competition with $500M–$238B-scale incumbents.

**If you still want to be in this space, the highest-value move is #1 or #2, informed by the
`RETHINK.md` research — not a rewrite of a defunct marketplace.**

---

## Sources (key)
- Live status: direct probes (2026-06-29). Founder retrospective: *So Long and Thanks for all the
  Phish* — https://microprediction.medium.com/so-long-and-thanks-for-all-the-phish-reflections-of-a-content-creator-b7aa1a2764f6
- Third-party walkthrough (onboarding pain): https://medium.com/databutton/what-is-the-deal-with-microprediction-afec82add4fb
- Cotton → Crunch Lab CSO: https://www.rebellionresearch.com/crunch-lab-announces-appointment-of-dr-peter-cotton-as-chief-scientific-officer
- Numerai $30M / $500M: https://chainwire.org/2025/11/20/numerai-raises-30-million-series-c-led-by-top-university-endowments-at-500-million-valuation/
- CrunchDAO / ADIA Lab: https://crunchdao.com/case-studies/adia-lab-crunchdao
- Prediction-market volumes 2025–26: https://www.pewresearch.org/short-reads/2026/05/27/trading-volume-on-prediction-markets-has-soared-in-recent-months/
- `precise` → `skfolio` / Schur paper: https://arxiv.org/pdf/2411.05807
- Repo/PyPI adoption: https://github.com/microprediction/microprediction · https://pypi.org/project/microprediction/
