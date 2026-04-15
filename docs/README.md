# Documentation Index

Design documents and user materials for voice-classifier.

---

## For developers

| Document | Purpose |
|---|---|
| [`requirements.md`](requirements.md) | Requirements specification (functional + non-functional), scope, acceptance criteria |
| [`architecture.md`](architecture.md) | High-level module map and data flow |
| [`detailed-design.md`](detailed-design.md) | Module-level contracts, invariants, extension points |
| [`cli-specification.md`](cli-specification.md) | Authoritative CLI reference (arguments, env vars, exit codes, outputs) |
| [`algorithm.md`](algorithm.md) | Clustering strategy: PCA, sweep design, scoring, quality thresholds |
| [`data-format.md`](data-format.md) | Input CSV specification and preprocessing rules |
| [`test-design.md`](test-design.md) | Test strategy, categories, mocking policy, coverage targets |
| [`security-design.md`](security-design.md) | Threat model, PII handling, credentials, review checklist |
| [`operations.md`](operations.md) | Deployment, routine workflow, runbook for common failures |
| [`development.md`](development.md) | Setup, coding conventions, PR workflow, extension procedures |

## For end users

| Language | Manual |
|---|---|
| English | [`manual/en.md`](manual/en.md) |
| 日本語 | [`manual/ja.md`](manual/ja.md) |
| Español | [`manual/es.md`](manual/es.md) |
| 繁體中文 | [`manual/zh-Hant.md`](manual/zh-Hant.md) |
| 简体中文 | [`manual/zh-Hans.md`](manual/zh-Hans.md) |

All five manuals cover the same scope; they are hand-written per language,
not machine-translated.

---

## Reading order by role

### New developer onboarding

1. [`../README.md`](../README.md) — get the tool running.
2. [`requirements.md`](requirements.md) — understand *what* the tool promises.
3. [`architecture.md`](architecture.md) — learn the modules.
4. [`detailed-design.md`](detailed-design.md) — dig into contracts.
5. [`development.md`](development.md) — set up your environment and workflow.

### Operator / analyst

1. [`../README.md`](../README.md).
2. [`manual/<your language>.md`](manual/).
3. [`operations.md`](operations.md) — when something goes wrong.

### Reviewer / approver

1. [`requirements.md`](requirements.md).
2. [`security-design.md`](security-design.md).
3. [`test-design.md`](test-design.md).

---

## Document maintenance

These design documents are code. Keep them in sync when:

- New CLI flags → update `cli-specification.md`, `manual/*.md`,
  `development.md` §8.
- New source module → update `architecture.md` module map and
  `detailed-design.md` §3.
- New clustering algorithm → update `algorithm.md`, `detailed-design.md`
  §3.3, `development.md` §7.
- New dependency → update `requirements.txt`, `development.md` §11.
- Threat-model-relevant change → walk the `security-design.md` §5 checklist.
