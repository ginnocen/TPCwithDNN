#!/bin/bash -e
cd "$(dirname "$0")"/..

function swallow() {
  local ERR=0
  local TMPF=$(mktemp /tmp/swallow.XXXX)
  local MSG=$1
  shift
  printf "[    ] $MSG" >&2
  "$@" &> $TMPF || ERR=$?
  if [[ $ERR != 0 ]]; then
    printf "\r[\033[31mFAIL\033[m] $MSG (log follows)\n" >&2
    cat $TMPF
    printf "\n" >&2
  else
    printf "\r[ \033[32mOK\033[m ] $MSG\n" >&2
  fi
  rm -f $TMPF
  return $ERR
}


if [[ $TRAVIS_PULL_REQUEST != "false" && $TRAVIS_COMMIT_RANGE ]]; then
  # Only check changed Python files (snappier)
  CHANGED_FILES=($(git diff --name-only $TRAVIS_COMMIT_RANGE | grep -E '\.py$' | grep -vE '^setup\.py$' | grep -vE '*legacy*.py|symmetrypadding3d\.py' || true))
else
  # Check all Python files
  CHANGED_FILES=($(find . -name '*.py' -a -not -name setup.py | grep -vE '*legacy*.py|symmetrypadding3d\.py'))
fi

ERRCHECK=
for PY in "${CHANGED_FILES[@]}"; do
  [[ -e "$PY" ]] || continue
  ERR=
  swallow "$PY: linting" pylint "$PY" || ERR=1
  [[ ! $ERR ]] || ERRCHECK="$ERRCHECK $PY"
done
[[ ! $ERRCHECK ]] || { printf "\n\nErrors found in:$ERRCHECK\n" >&2; exit 1; }
