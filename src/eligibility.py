"""eligibility.py — Eligibility rules, suppression, and sequencing.

Applies operational next-action logic (REQ-03):
- Gap closed → suppress gap-specific action.
- Appointment scheduled → suppress duplicate outreach.
- Contact cooldown enforcement.
- New barrier → change eligible next actions.

This is operational sequencing, NOT learned full-journey optimization.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schemas import ContactLimit, PolicyConfig


def apply_eligibility_rules(
    candidates_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    config: PolicyConfig,
) -> pd.DataFrame:
    """Apply all eligibility and suppression rules to candidate rows.

    Modifies candidates_df in place: sets 'eligible', 'suppressed',
    'suppression_reason' columns.

    Returns the updated DataFrame.
    """
    df = candidates_df.copy()
    df["eligible"] = True
    df["suppressed"] = False
    df["suppression_reason"] = ""

    # 1. Suppress closed gaps
    if config.suppress_closed_gaps:
        df = _suppress_closed_gaps(df, gaps_df)

    # 2. Suppress duplicate outreach
    if config.suppress_duplicate_outreach:
        df = _suppress_duplicate_outreach(df, gaps_df)

    # 3. Enforce contact cooldown
    for cl in config.contact_limits:
        df = _apply_contact_cooldown(df, cl, config.decision_date)

    # 4. Enforce eligibility flag from gaps
    if config.enforce_eligibility:
        df = _enforce_gap_eligibility(df, gaps_df)

    return df


def _suppress_closed_gaps(
    df: pd.DataFrame, gaps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Suppress interventions for already-closed gaps."""
    closed = set(
        zip(gaps_df.loc[gaps_df["gap_status"] == "closed", "member_id"],
            gaps_df.loc[gaps_df["gap_status"] == "closed", "measure_code"])
    )
    for idx, row in df.iterrows():
        if (row["member_id"], row["measure_code"]) in closed:
            df.at[idx, "eligible"] = False
            df.at[idx, "suppressed"] = True
            df.at[idx, "suppression_reason"] = "gap_already_closed"
    return df


def _suppress_duplicate_outreach(
    df: pd.DataFrame, gaps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Suppress if gap has pending/scheduled status (appointment scheduled)."""
    pending = set(
        zip(gaps_df.loc[gaps_df["gap_status"] == "pending", "member_id"],
            gaps_df.loc[gaps_df["gap_status"] == "pending", "measure_code"])
    )
    for idx, row in df.iterrows():
        if (row["member_id"], row["measure_code"]) in pending:
            df.at[idx, "eligible"] = False
            df.at[idx, "suppressed"] = True
            if df.at[idx, "suppression_reason"]:
                df.at[idx, "suppression_reason"] += "; duplicate_outreach"
            else:
                df.at[idx, "suppression_reason"] = "duplicate_outreach"
    return df


def _apply_contact_cooldown(
    df: pd.DataFrame,
    contact_limit: ContactLimit,
    decision_date: date,
) -> pd.DataFrame:
    """Suppress members who were contacted too recently."""
    if contact_limit.cooldown_days <= 0:
        return df

    cooldown_cutoff = decision_date - timedelta(days=contact_limit.cooldown_days)

    if "last_contact_date" in df.columns:
        for idx, row in df.iterrows():
            lcd = row.get("last_contact_date")
            if lcd is not None and not pd.isna(lcd):
                if isinstance(lcd, str):
                    lcd = date.fromisoformat(lcd)
                if lcd > cooldown_cutoff:
                    df.at[idx, "eligible"] = False
                    df.at[idx, "suppressed"] = True
                    reason = f"contact_cooldown_{contact_limit.cooldown_days}d"
                    if df.at[idx, "suppression_reason"]:
                        df.at[idx, "suppression_reason"] += f"; {reason}"
                    else:
                        df.at[idx, "suppression_reason"] = reason
    return df


def _enforce_gap_eligibility(
    df: pd.DataFrame, gaps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mark ineligible if the gap record itself says not eligible."""
    ineligible = set(
        zip(gaps_df.loc[~gaps_df["eligible"], "member_id"],
            gaps_df.loc[~gaps_df["eligible"], "measure_code"])
    )
    for idx, row in df.iterrows():
        if (row["member_id"], row["measure_code"]) in ineligible:
            df.at[idx, "eligible"] = False
            df.at[idx, "suppressed"] = True
            if df.at[idx, "suppression_reason"]:
                df.at[idx, "suppression_reason"] += "; gap_not_eligible"
            else:
                df.at[idx, "suppression_reason"] = "gap_not_eligible"
    return df


def demonstrate_state_changes(
    candidates_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    config: PolicyConfig,
) -> Dict[str, pd.DataFrame]:
    """Show before/after effect of state changes on eligibility.

    Returns dict with 'before' and 'after' DataFrames plus 'changes' summary.
    """
    before = apply_eligibility_rules(candidates_df, gaps_df, config)

    # Simulate a state change: close the first open gap
    gaps_modified = gaps_df.copy()
    first_open = gaps_modified[gaps_modified["gap_status"] == "open"].index
    if len(first_open) > 0:
        gaps_modified.loc[first_open[0], "gap_status"] = "closed"

    after = apply_eligibility_rules(candidates_df, gaps_modified, config)

    changes = before[["candidate_id", "eligible"]].merge(
        after[["candidate_id", "eligible"]],
        on="candidate_id", suffixes=("_before", "_after"),
    )
    changes = changes[changes["eligible_before"] != changes["eligible_after"]]

    return {"before": before, "after": after, "changes": changes,
            "modified_gap_idx": first_open[0] if len(first_open) > 0 else None}
