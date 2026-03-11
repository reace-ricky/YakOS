#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BRANCH="main"

echo "Finding merged PRs with copilot/* head branches..."
gh pr list \
  --state merged \
  --search "head:copilot/" \
  --json headRefName \
  --jq '.[].headRefName' |
while read -r branch; do
  echo "Deleting remote branch origin/$branch"
  git push origin --delete "$branch" || true
done

echo "Finding closed (unmerged) PRs with copilot/* head branches..."
gh pr list \
  --state closed \
  --search "head:copilot/" \
  --json headRefName \
  --jq '.[].headRefName' |
while read -r branch; do
  echo "Deleting remote branch origin/$branch"
  git push origin --delete "$branch" || true
done

echo "Cleaning up local copilot/* branches merged into $DEFAULT_BRANCH..."
git checkout "$DEFAULT_BRANCH"
git pull origin "$DEFAULT_BRANCH"
git branch --merged "$DEFAULT_BRANCH" \
  | grep '^  copilot/' \
  | sed 's/^  //' \
  | xargs -r git branch -d
