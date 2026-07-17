#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import date

SINGLEFREQ_INTRO_T_START = "2020-09-01"
SINGLEFREQ_INTRO_T_END = "2021-03-31"

# Match the intro figure: data support is fit with a 120-day cap, then the
# daily convolution PMF is discretized/truncated at tau_max=60.
SINGLEFREQ_EMPIRICAL_MAX_DELAY_DAYS = 120
SINGLEFREQ_PMF_TAU_MAX = 60

_INTRO_START_DATE = date.fromisoformat(SINGLEFREQ_INTRO_T_START)
_INTRO_END_DATE = date.fromisoformat(SINGLEFREQ_INTRO_T_END)
SINGLEFREQ_INTRO_WINDOW_DAYS = (
    _INTRO_END_DATE - _INTRO_START_DATE
).days + 1
