# AST_Clone_Extractability/feasibility.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Dict

from AST_Clone_Extractability.rw_vars import REG_PRE, REG_WITHIN, REG_POST, RWRegions


@dataclass
class FeasibilityResult:
    In: Set[str]
    Out: Set[str]
    CFHazard: bool
    Extractable: bool
    P: int
    R: int


def compute_in_out(rw: RWRegions) -> (Set[str], Set[str]):
    """
    Paper-aligned definitions:

      Def_before(i): variables defined before clone.
        - approximated as V_w_pre plus parameters (defined at entry)
      Use(i): variables read within clone => V_r_within
      Def(i): variables written within clone => V_w_within
      Use_after(i): variables read after clone => V_r_post

      In(i)  = Use(i) ∩ Def_before(i)
      Out(i) = Def(i) ∩ Use_after(i)
    """
def compute_in_out(rw: RWRegions) -> (Set[str], Set[str]):
    # Parameters and variables defined BEFORE the clone
    def_before = rw.vw[REG_PRE] | rw.params_in_method
    
    # Variables that are read within the clone
    use_within = rw.vr[REG_WITHIN]
    
    # Variables that are written within the clone
    def_within = rw.vw[REG_WITHIN]
    
    # Variables that are read AFTER the clone
    use_after = rw.vr[REG_POST]

    # In(i) = variables read that were defined BEFORE the clone
    # This filters out 'status' and 'fileLength' because they weren't in def_before
    In = use_within & def_before
    
    # Out(i) = variables written that are used AFTER the clone
    Out = def_within & use_after
    
    return In, Out

def compute_in_out_types(rw: RWRegions) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Convenience helper (does NOT change your feasibility rule):
      returns (In, Out, InType, OutType)

    Requires rw.var_types (added in rw_vars.py). Unknown types are ignored.
    """
    In, Out = compute_in_out(rw)

    var_types = getattr(rw, "var_types", {}) or {}
    InType = {var_types[v] for v in In if v in var_types}
    OutType = {var_types[v] for v in Out if v in var_types}

    return In, Out, InType, OutType

def decide_extractable(In: Set[str], Out: Set[str], cf_hazard: bool, P: int, R: int) -> bool:
    in_ok = True if P is None else (len(In) <= P)
    out_ok = (len(Out) <= R)

    return in_ok and out_ok and (not cf_hazard)
