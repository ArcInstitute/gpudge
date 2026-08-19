# Code reviews

Reports from the multi-agent whole-codebase reviews ("ultrareviews") run against
gpudge. They are published because a library that claims to be well-tested
should be willing to show what an adversarial read of it turned up.

**Every report here carries a resolution status**, so a reader can tell a fixed
defect from a live one without cross-referencing the changelog. A findings list
without dispositions reads as a list of known, open bugs — the opposite of what
publishing it is for.

Issue and PR numbers on this page and in the reports — `gpudge_arc#59` and
similar — are provenance: they identify entries in an issue tracker that is not
public and will not resolve from here.

| report | findings | disposition |
|---|---|---|
| [2026-06-13](2026-06-13-gpudge-ultrareview.md) | 23 survived verification, of 28 raw | status table in the report: 9 confirmed defects, 5 fixed in the same PR, 4 filed as issues |
| [2026-06-27](2026-06-27-gpudge-ultrareview.md) | 22 confirmed | all 22 addressed in v0.3.1 (gpudge_arc#59); status table added 2026-08-18, re-verified against the tree at `v0.8.0` |

Not every review is published as a document. The **2026-08** ultrareview — 41
findings filed, 39 put through adversarial verifiers on an H100, 3 refuted — is
recorded in `CHANGELOG.md` instead, under `0.8.0`, where each fix sits beside the
behaviour it changed. That is the more useful place for it: those findings were
acted on immediately rather than triaged over weeks, so a separate report and the
release notes would have been the same document twice.

Two caveats worth stating plainly:

- The 2026-06-13 table's "Issue filed" rows record the disposition **as of that
  date**. Some of those issues have since been closed; the table was not
  rewritten, because a review report is a dated record rather than a live
  tracker.
- Statuses are verified by reading the tree, not by trusting the changelog. Where
  a finding was answered by deciding *not* to change anything, it says so.
