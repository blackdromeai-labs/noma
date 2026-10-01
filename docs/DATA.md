# Data

This page describes how the training and evaluation data were built. The datasets themselves
are not part of this repository.

## Shape of the data

Every item is one decision: a state, one typed question, a target distribution over the
options, an abstain flag, and sometimes a supporting quote. Items are grouped into decision
families (routing, intent, policy, severity, adequacy, relevance, tone and safety, trade-off,
temporal and numeric, adversarial, ambiguous or out of scope, and others) and the training
mix is balanced across families and across yes and no.

About 34,000 decisions were used for training and 1,800 for calibration.

## Where it comes from

Three kinds of material, in roughly this order of volume:

1. **Public datasets with permissive licences**, converted into the decision format. Human
   annotator disagreement, where the source records it, is kept as a soft label.
2. **Code-generated items with exact labels.** Generators write a document and compute the
   answer, so the label is correct by construction: long policies with amendments and
   effective dates, nearly-right worked answers with one slip carried through the working,
   rest rosters across daylight-saving changes, currency-date rules, and register traces with
   look-alike identifiers. Generators sample to quotas (for example, the amendment decides
   the answer in 45% of long-policy items) so the model cannot learn a shortcut.
3. **Model-labelled items**, produced by the cascade below.

## Blind, agreement-gated labelling

Labels from a single model inherit that model's habits. Noma's labels come from a cascade
designed so that no labeller can lean on another:

- Two different frontier models label every item **blind**: opaque item IDs, shuffled option
  order, no access to each other's output or to any answer file.
- Where they agree, the label is accepted and their distributions are averaged into a soft
  target.
- Where they disagree, a third model judges that item only.
- A random 10% of the agreed items is audited by the judge as well. Audit agreement was 98%.

## Contrastive groups

Much of the generated data comes in groups: one realistic document, three questions, and for
each question an edit of at most 8 words that flips its answer. The model sees both versions.
This forces it to read the deciding detail instead of the overall tone of the document.

## Evaluation data and leakage control

- **Sealed set.** 401 decisions across 12 families, human reviewed, held out from every
  training run and every generator. 386 have a definite answer and are scored for accuracy.
  The set is private and is not released, so it stays usable as a test.
- **JevBench public tasks** (easy, original, hard) are used for evaluation only, with one
  disclosed exception described next.
- **Hard-tier split.** The 111 public hard items were split, stratified by family, into 55
  seeds and 56 held-out items. The seeds were used only as abstract structure templates for
  synthetic data: the structure of the problem, never its text. The 56 held-out items were
  never seen by any generator or training run. Scores on the 55 seeds are not reported as
  hard-tier results; see [EVALUATION.md](EVALUATION.md).
- **Decontamination.** Every training item is checked against every evaluation item with
  13-gram overlap and removed on a match. This check caught and removed about 1,600 of our
  own generated items whose template wording echoed a seed; the templates were reworded.

## Not included

The sealed set, raw labeller outputs and the built training files are not in this
repository. The trainer accepts any data in the record format shown in
[TRAINING.md](TRAINING.md).
