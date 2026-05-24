#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   PRODUCTION-LEVEL CREDIT CARD FRAUD DETECTION SYSTEM   ║
║   Hybrid: Rule-Based Override + XGBoost ML Model        ║
║   Run: python3 predict_fraud.py                         ║
╚══════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
import os
import sys

PICKLE_FILE = 'fraud_detection_pipeline.pkl'


# ═══════════════════════════════════════════════════════════
# LOAD PIPELINE
# ═══════════════════════════════════════════════════════════

def load_pipeline():
    """Load the trained model pipeline from pickle file."""
    if not os.path.exists(PICKLE_FILE):
        print("\nERROR: Model file not found.")
        print("Please run the Jupyter notebook first to train the model.\n")
        sys.exit(1)
    pipeline = joblib.load(PICKLE_FILE)
    print(f"  Model loaded  : {pipeline['best_model']}")
    print(f"  Threshold     : {pipeline['best_threshold']}")
    print(f"  Features used : {len(pipeline['selected_features'])}")
    return pipeline


# ═══════════════════════════════════════════════════════════
# USER INPUT
# ═══════════════════════════════════════════════════════════

def ask_number(prompt):
    """Ask user for a number — loops until valid input given."""
    while True:
        val = input(prompt).strip()
        try:
            return float(val)
        except ValueError:
            print("  Please enter a valid number (e.g. 5000 or 1234.56)\n")


def ask_type():
    """Ask user for transaction type — only CASH_OUT and TRANSFER."""
    types = {'1': 'CASH_OUT', '2': 'TRANSFER'}
    print("\nTransaction Type:")
    print("  1. CASH_OUT  (Withdrawing money from your account)")
    print("  2. TRANSFER  (Sending money to another person)")
    while True:
        choice = input("Enter number (1-2): ").strip()
        if choice in types:
            return types[choice]
        print("  Invalid. Please enter 1 or 2.\n")


def get_inputs_for_type(tx_type):
    """Collect transaction details based on transaction type."""
    if tx_type == 'CASH_OUT':
        print("\n  Enter CASH_OUT details:\n")
        amount          = ask_number("  Amount withdrawn                     : ")
        sender_before   = ask_number("  Your balance BEFORE withdrawal       : ")
        sender_after    = ask_number("  Your balance AFTER  withdrawal       : ")
        receiver_before = ask_number("  Agent/ATM balance BEFORE withdrawal  : ")
        receiver_after  = ask_number("  Agent/ATM balance AFTER  withdrawal  : ")

    elif tx_type == 'TRANSFER':
        print("\n  Enter TRANSFER details:\n")
        amount          = ask_number("  Amount transferred                   : ")
        sender_before   = ask_number("  Your balance BEFORE transfer         : ")
        sender_after    = ask_number("  Your balance AFTER  transfer         : ")
        receiver_before = ask_number("  Receiver balance BEFORE transfer     : ")
        receiver_after  = ask_number("  Receiver balance AFTER  transfer     : ")

    return amount, sender_before, sender_after, receiver_before, receiver_after


# ═══════════════════════════════════════════════════════════
# RULE-BASED FRAUD DETECTION
# This runs BEFORE the ML model
# If any rule fires → instantly flagged as fraud
# This catches obvious fraud that ML might miss
# ═══════════════════════════════════════════════════════════

def rule_based_check(amount, sender_before, sender_after,
                     receiver_before, receiver_after):
    """
    Rule-based override system.
    Returns (is_fraud: bool, reason: str)
    These rules are based on real banking fraud patterns.
    """

    # Calculate balance errors
    error_orig = sender_before - amount - sender_after
    error_dest = receiver_after - receiver_before - amount

    # ── Rule 1: Money Duplication ─────────────────────────
    # Sender balance didn't decrease but money was "sent"
    # This means money was duplicated — definite fraud
    if sender_after > sender_before and amount > 0:
        return True, "MONEY DUPLICATION — Sender balance increased after sending money"

    # ── Rule 2: Money Disappearance ───────────────────────
    # Money left sender but never arrived at receiver
    # Both balances changed but amounts don't match
    if abs(error_orig) > 1 and abs(error_dest) > 1:
        # Both sides have errors — money vanished in transit
        if error_orig < -1 and error_dest < -1:
            return True, "MONEY DISAPPEARANCE — Balance mismatch on both sender and receiver"

    # ── Rule 3: Receiver Account Never Updated ────────────
    # Sender balance dropped but receiver balance unchanged
    # Money was taken but never delivered
    sender_dropped = sender_before - sender_after
    receiver_gained = receiver_after - receiver_before
    if sender_dropped > 0 and abs(receiver_gained) < 1 and amount > 1000:
        return True, "RECEIVER NOT UPDATED — Money taken from sender but receiver unchanged"

    # ── Rule 4: Sender Balance Completely Wiped + Large Amount ─
    # Full balance drained in one go for a large amount
    if sender_after == 0 and sender_before > 100000:
        return True, "LARGE ACCOUNT DRAIN — Full high-value account wiped in one transaction"

    # ── Rule 5: Destination always zero ──────────────────
    # Receiver account had 0 before and after — money never arrived
    if receiver_before == 0 and receiver_after == 0 and amount > 10000:
        return True, "DESTINATION ZERO — Large amount sent but receiver balance stayed at zero"

    # ── Rule 6: Extreme balance mismatch ─────────────────
    # The difference between what should have happened and what did is huge
    if abs(error_orig) > amount * 0.5 and amount > 5000:
        return True, "SEVERE BALANCE MISMATCH — Sender balance error exceeds 50% of amount"

    # No rules fired — pass to ML model
    return False, None


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Must EXACTLY match Cell 7 of the notebook
# ═══════════════════════════════════════════════════════════

def build_features(tx_type, amount, sender_before, sender_after,
                   receiver_before, receiver_after, type_mapping):
    """Build all features exactly as done in notebook Cell 7."""

    tx_type_encoded = type_mapping.get(tx_type.upper(), 1)

    # Core balance error signals
    error_bal_orig = sender_before - amount - sender_after
    error_bal_dest = receiver_after - receiver_before - amount

    features = {
        # Raw transaction fields
        'step'                   : 1,
        'type'                   : tx_type_encoded,
        'amount'                 : amount,
        'oldbalanceOrg'          : sender_before,
        'newbalanceOrig'         : sender_after,
        'oldbalanceDest'         : receiver_before,
        'newbalanceDest'         : receiver_after,

        # Core fraud signals
        'errorBalanceOrig'       : error_bal_orig,
        'errorBalanceDest'       : error_bal_dest,

        # Absolute mismatch values
        'sender_balance_error'   : abs(error_bal_orig),
        'receiver_balance_error' : abs(error_bal_dest),

        # Total money flow error — catches duplication and disappearance
        'money_flow_error'       : abs(error_bal_orig) + abs(error_bal_dest),

        # Ratio features
        'amount_to_balance_ratio': amount / (sender_before + 1),

        # Flag features
        'dest_zero_flag'         : int(receiver_before == 0 and receiver_after == 0),
        'sender_wiped'           : int(sender_after == 0),
        'large_transaction'      : int(amount > 200000),

        # Expected balance features
        'orig_balance_expected'  : sender_before - amount,
        'dest_balance_expected'  : receiver_before + amount,
    }

    return features


# ═══════════════════════════════════════════════════════════
# HYBRID PREDICTION (Rule-Based + ML)
# ═══════════════════════════════════════════════════════════

def predict(pipeline, tx_type, amount,
            sender_before, sender_after,
            receiver_before, receiver_after):
    """
    Two-stage hybrid fraud detection:
    Stage 1 — Rule-based check (catches obvious fraud instantly)
    Stage 2 — ML model with probability threshold (catches subtle fraud)
    """

    # ── Stage 1: Rule-Based Check ─────────────────────────
    rule_fraud, rule_reason = rule_based_check(
        amount, sender_before, sender_after,
        receiver_before, receiver_after
    )

    if rule_fraud:
        # Rule fired — return fraud immediately without ML
        return {
            'prediction'   : 1,
            'label'        : '⚠️   FRAUDULENT TRANSACTION DETECTED',
            'fraud_pct'    : 99.0,
            'legit_pct'    : 1.0,
            'method'       : 'Rule-Based Override',
            'rule_reason'  : rule_reason
        }

    # ── Stage 2: ML Model ─────────────────────────────────
    type_mapping      = pipeline['type_mapping']
    selected          = pipeline['selected_features']
    scaler            = pipeline['scaler']
    model             = pipeline['models'][pipeline['best_model']]
    threshold         = pipeline['best_threshold']

    all_features = build_features(
        tx_type, amount, sender_before, sender_after,
        receiver_before, receiver_after, type_mapping
    )

    # Build feature vector in correct column order
    # Only keep features that were selected during training
    available = {k: v for k, v in all_features.items() if k in selected}
    vector        = pd.DataFrame([available])[selected]
    vector_scaled = scaler.transform(vector)

    # Use predict_proba with custom threshold — not just predict()
    # Lower threshold = catches more fraud (fewer false negatives)
    prob       = model.predict_proba(vector_scaled)[0]
    fraud_prob = prob[1]
    prediction = int(fraud_prob >= threshold)

    return {
        'prediction' : prediction,
        'label'      : '⚠️   FRAUDULENT TRANSACTION DETECTED' if prediction == 1
                       else '✅  LEGITIMATE TRANSACTION',
        'fraud_pct'  : round(fraud_prob * 100, 2),
        'legit_pct'  : round(prob[0] * 100, 2),
        'method'     : f'ML Model ({pipeline["best_model"]}, threshold={threshold})',
        'rule_reason': None
    }


# ═══════════════════════════════════════════════════════════
# PRINT RESULT
# ═══════════════════════════════════════════════════════════

def print_result(result):
    """Print prediction result in a clean formatted way."""
    print("\n" + "="*52)
    print("  RESULT")
    print("="*52)
    print(f"  {result['label']}")
    print(f"\n  Fraud chance  : {result['fraud_pct']:.2f}%")
    print(f"  Legit chance  : {result['legit_pct']:.2f}%")
    print(f"  Detected by   : {result['method']}")
    if result['rule_reason']:
        print(f"  Reason        : {result['rule_reason']}")
    print("="*52)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "="*52)
    print("    CREDIT CARD FRAUD DETECTION SYSTEM")
    print("    Hybrid: Rule-Based + ML (XGBoost)")
    print("="*52)

    pipeline = load_pipeline()
    print()

    while True:
        print("\nEnter the transaction details below:")
        print("-"*52)

        tx_type = ask_type()

        amount, sender_before, sender_after, \
        receiver_before, receiver_after = get_inputs_for_type(tx_type)

        result = predict(pipeline, tx_type, amount,
                         sender_before, sender_after,
                         receiver_before, receiver_after)

        print_result(result)

        again = input("\nCheck another transaction? (y/n): ").strip().lower()
        if again != 'y':
            print("\nGoodbye!\n")
            break


if __name__ == '__main__':
    main()
