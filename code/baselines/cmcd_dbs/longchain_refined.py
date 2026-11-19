#!/usr/bin/env python3
"""longchain.py chains (IMH / RT / pRT) on an inference-time refined CMCD / DBS path (2026-11-14).
Same as `longchain.py --kind cmcd`, except that after loading the model its coefficients are replaced by those of
refine_cmcd.refined_coeffs(model, MULT, 1, hold=True): T' = MULT * T steps, eta / MULT per step, nets held at the
trained step floor(s) (the refinement that lowered the CMCD work variance). The stepper id carries the factor, so
the chain seeds differ from the unrefined runs. Usage: REFINE_MULT=4 python longchain_refined.py <longchain args>."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # code/
import exp_chains as LCH  # noqa: E402
import refine_cmcd as RC  # noqa: E402

MULT = int(os.environ['REFINE_MULT'])
_init = LCH.CmcdStepper.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    self.m.T0 = self.m.T
    self.m.T = self.m.T0 * MULT
    self.c = RC.refined_coeffs(self.m, MULT, 1, True)
    self.T = self.m.T
    self.id += f'|refine{MULT}h'


LCH.CmcdStepper.__init__ = init
if __name__ == '__main__':
    LCH.main()
