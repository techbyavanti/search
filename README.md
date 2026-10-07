# Semantic Product Search

A Java service for semantic product search over the [WANDS](https://github.com/wayfair/WANDS)
(Wayfair) furniture catalog. Built as a long-term portfolio project focused on one
thing: **retrieval and ranking quality** — hybrid lexical/semantic search plus
cross-encoder reranking, proven with real offline IR evaluation (nDCG, MRR,
Recall@k) rather than eyeballed results. Sibling portfolio project to
[`recommendation-engine`](https://github.com/avantiwhenever/recommendation-engine)
(a React + GraphQL recommendation engine over the same catalog) — that one
explores a different problem (personalized re-ranking) and a more
service-oriented stack; this one goes deep on one thing instead: proving each
stage of a ranking pipeline actually improves search quality, with the
numbers to back it up.

> **[Live snapshot demo →](https://avantiwhenever.github.io/search/)** — a
> few real, captured queries comparing all six strategies side by side
> (static GitHub Pages page, not a live backend — [run it yourself](HOWTO.md)
> for the real thing).

## Where to go

**Recruiter, skimming this?** The [live demo](https://avantiwhenever.github.io/search/)
is the fastest path — six ranking strategies on a real 43,000-product
catalog, side by side, with real precision/recall numbers instead of a
handful of cherry-picked examples.

**Technical hiring manager?** Start with **[WRITEUP.md](WRITEUP.md)** — the
honest narrative of what was tried, what broke, and what the numbers
actually showed, including two models trained from scratch that *didn't*
beat their baseline. **[RESULTS.md](RESULTS.md)** has the raw evaluation
numbers; **[CI.md](CI.md)** covers the CI/security posture (CVE scanning,
SAST, secret scanning, Dependabot).

**Reviewing this codebase?** Start with **[ARCHITECTURE.md](ARCHITECTURE.md)**
— module layout, data-flow diagram, the six ranking strategies, and the key
design decisions behind them. **[HOWTO.md](HOWTO.md)** gets it running
locally. **[TRAINING.md](TRAINING.md)** covers the two custom-trained
models.

## Docs

| Doc | What's in it |
|---|---|
| [WRITEUP.md](WRITEUP.md) | Long-form narrative: what was tried at each milestone, what broke, and how the final numbers were arrived at |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module layout, data-flow diagram, the six ranking strategies, key design decisions, implementation notes |
| [ROADMAP.md](ROADMAP.md) | Milestone-by-milestone checklist — what's done, in progress, and next |
| [RESULTS.md](RESULTS.md) | Canonical, always-current evaluation numbers (regenerate with `./scripts/run-eval.sh`) |
| [TRAINING.md](TRAINING.md) | Setup, process, and results for this project's two custom-trained models |
| [CI.md](CI.md) | The CI pipeline and security posture (CVE/secret scanning, SAST, Dependabot) |
| [HOWTO.md](HOWTO.md) | Step-by-step local setup, with a verification step after each stage |
