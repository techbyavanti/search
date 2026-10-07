# CI & security posture

One GitHub Actions workflow, `.github/workflows/build.yml`, six jobs on
every push/PR:

- **`test`** — `mvn test`. Fast, needs no Elasticsearch or models (the
  tests that do need them skip gracefully via `Assumptions.assumeTrue`
  when those aren't present).
- **`cve-scan`** — Trivy against every module's resolved `pom.xml`
  dependency tree (SCA), failing on CRITICAL/HIGH/MEDIUM known CVEs.
  Reproduce locally with `./scripts/scan-cves.sh`.
- **`sast`** — Semgrep over the Java (and Python training) source itself
  — not just its dependencies — with the `p/java`, `p/security-audit`,
  and `p/secrets` rulesets.
- **`secret-scan`** — gitleaks against the full commit history, so a
  credential can't slip in and get baked into an image layer later.
- **`docker-lint`** — hadolint against `search-api/Dockerfile` (this is
  what caught that the non-root `USER` directive should be numeric
  `1000:1000`, not a name, for portability to orchestrators that don't
  read `/etc/passwd`).
- **`docker-image-scan`** — builds the real `search-api` runtime image
  and scans it (OS + JRE layer, not just the jar's dependencies) with
  Trivy, plus generates a CycloneDX SBOM as a build artifact via Syft.

`cve-scan`, `sast`, and `docker-image-scan` each run their scanner twice:
once in the format that gates the job (`table` + `exit-code: 1`), once as
SARIF uploaded via `github/codeql-action/upload-sarif` so findings also
land in the repo's **Security → Code scanning** tab instead of only
CI logs — non-blocking, so a SARIF upload hiccup never fails the job.

Repo-level (not workflow files, so not visible in `.github/`, but part of
the same posture): **Dependabot version updates** (`.github/dependabot.yml`
— weekly, covering the Maven reactor, `search-api/Dockerfile`, and the
Actions workflow itself), plus **Dependabot alerts**, **Dependabot
security updates**, **secret scanning**, and **secret scanning push
protection** all enabled in repo settings — free for a public repo, and
the platform-native complement to the gitleaks/Trivy checks above (push
protection in particular blocks a secret *before* it's pushed, which a
CI-time scan structurally can't).

Deliberately **not** in this pipeline: Cosign image signing and registry
immutability controls. Those protect a *registry* this project doesn't
have — there's no ECR/Harbor push step, so wiring up signing would be
config theater, not a real control. Worth adding the day this project
(or reader's fork of it) actually pushes images somewhere.

## Regression eval (not run in CI)

The full-catalog regression eval (ingest all ~43K products with real
embeddings, run all 480 WANDS queries against all four strategies, fail if
any strategy's nDCG@10 drops below the floor in `ci/eval-baseline.json`) is
**not** run in CI — embedding 43K products is genuinely slow, and that cost
doesn't belong on a shared runner on every push. Run it locally instead:

```
./scripts/run-eval.sh --baseline-file ci/eval-baseline.json
```

against a local Elasticsearch instance (see `docker-compose.yml`). Omit
`--baseline-file` to just regenerate `RESULTS.md` without gating.

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — module structure and key decisions
- [HOWTO.md](HOWTO.md) — running it locally
