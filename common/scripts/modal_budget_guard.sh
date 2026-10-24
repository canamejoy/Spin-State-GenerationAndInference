#!/usr/bin/env bash
#
# Hard spend guard for the Modal accounts that run the simulations.
#
# Why this exists: over one day of launches, three accounts went past their
# credit. The cause was not a single expensive run -- it was tracking the
# balance by subtracting predicted costs from the grant instead of reading it.
# The predictions counted GPU-seconds only, ignoring CPU, memory and container
# startup, which add about 8% on a GPU job and dominate a long CPU job. Small
# per-run errors compounded across a dozen launches until the estimate and the
# truth had nothing to do with each other.
#
# So: read the balance, never estimate it, and refuse to launch when an account
# is short. Run `check` immediately before every `modal run`.
#
# Usage:
#   scripts/modal_budget_guard.sh report
#   scripts/modal_budget_guard.sh check <account> <estimated-usd>
#
# Environment:
#   MIN_USD   margin that must remain after the run   (default 1.00)
#   GRANT     credit granted per account, in USD      (default 30.00)
#   ACCOUNTS  directory holding <account>/.modal.toml (default ~/.modal-accounts)
#
# Exit status is 1 on refusal, so it chains safely:
#   scripts/modal_budget_guard.sh check my-account 2.50 || exit 1
#
set -u
MIN_USD=${MIN_USD:-1.00}
GRANT=${GRANT:-30.00}
ACCOUNTS=${ACCOUNTS:-$HOME/.modal-accounts}

balance() {                       # $1 = account -> remaining credit, or ERR
  # Two signals, because neither alone suffices. Modal's `Credits` line reports
  # credits APPLIED, which on a healthy account simply equals the metered cost,
  # so subtracting them always yields zero. `Billed Cost` above zero is the
  # unambiguous sign that the grant is spent and further work lands on the card.
  # Below that the remainder needs the grant, which the summary never states, so
  # GRANT stays an assumption -- but one that can only err in the safe
  # direction, since an exhausted account is caught by Billed Cost whatever the
  # grant was.
  local out metered billed
  out=$(MODAL_CONFIG_PATH="$ACCOUNTS/$1/.modal.toml" \
        timeout 180 modal billing summary 2>&1) || { echo "ERR"; return 1; }
  metered=$(echo "$out" | grep -i "Metered Cost" | grep -oE "[0-9.]+" | head -1)
  billed=$(echo "$out" | grep -i "Billed Cost" | grep -oE "[0-9.]+" | head -1)
  [ -z "$metered" ] && { echo "ERR"; return 1; }
  python3 -c "
billed = float('${billed:-0}')
print('0.00' if billed > 0.005 else f'{max(0.0, $GRANT - $metered):.2f}')"
}

case "${1:-report}" in
  report)
    printf "%-20s %10s %11s\n" account remaining status
    for a in $(ls -1 "$ACCOUNTS" 2>/dev/null); do
      [ -f "$ACCOUNTS/$a/.modal.toml" ] || continue
      b=$(balance "$a")
      if [ "$b" = "ERR" ]; then st="UNREADABLE"
      elif python3 -c "exit(0 if $b >= $MIN_USD else 1)"; then st="usable"
      else st="BLOCKED"; fi
      printf "%-20s %10s %11s\n" "$a" "\$$b" "$st"
    done
    ;;
  check)
    account=${2:?usage: check <account> <estimated-usd>}
    estimate=${3:-0}
    b=$(balance "$account")
    if [ "$b" = "ERR" ]; then
      echo "REFUSED: could not read the balance of $account"; exit 1
    fi
    if ! python3 -c "exit(0 if $b >= $MIN_USD else 1)"; then
      echo "REFUSED: $account has \$$b, below the \$$MIN_USD floor"; exit 1
    fi
    if ! python3 -c "exit(0 if $b >= $estimate + $MIN_USD else 1)"; then
      echo "REFUSED: $account has \$$b and the run costs ~\$$estimate, which" \
           "leaves less than \$$MIN_USD"; exit 1
    fi
    echo "OK: $account has \$$b, the run is estimated at \$$estimate"
    ;;
  *)
    echo "usage: $0 report | check <account> <estimated-usd>"; exit 2
    ;;
esac
