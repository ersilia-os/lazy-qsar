"""Which per-descriptor channels the runner has to compute for a given set of outputs.

Pure function over the requested output names, so it runs on a base install. It matters
because featurization is ~93% of inference wall clock and every extra channel is work done
per chunk: asking for one more output must not quietly turn into a second pass.
"""

from lazyqsar.ensemble.channels import required_channels


def test_required_channels_minimal_without_ad():
    """No AD means uniform weights, and nothing then needs the percentile channel.

    `rank` used to require it. It no longer does: a reference rank is read off the pooled
    probability, and the `r` channel now carries only the out-of-fold percentile the
    weighting uses. Asking for `rank` without an applicability domain therefore costs one
    pass, not two.
    """
    assert required_channels(("proba",), has_ad=False) == {"y"}
    assert required_channels(("logit", "lift", "binary"), has_ad=False) == {"y"}
    assert required_channels(("rank",), has_ad=False) == {"y"}
    assert required_channels(("score",), has_ad=False) == {"y", "s"}


def test_required_channels_with_ad_always_needs_rank():
    """The weighting derives its per-sample reliability term from the ranks."""
    assert required_channels(("proba",), has_ad=True) == {"y", "r", "a"}
    assert required_channels(("score",), has_ad=True) == {"y", "r", "s", "a"}
