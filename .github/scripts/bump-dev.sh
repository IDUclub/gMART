#!/usr/bin/env bash
# Bump the version for the change that was just pushed to dev and push a "bump:" commit.
#
# The merged PR decides the increment (see versioning.py). The commit is pushed with the
# deploy bot's token so that it starts the dev release, which deploys only "bump:" commits.
# A concurrent merge can move dev between our fetch and push: the bump is then rebuilt on the
# new dev and pushed again, so every change gets its own version.
#
# Env: GH_TOKEN (pull-request read/edit), SHA (the pushed commit), GITHUB_REPOSITORY.
set -euo pipefail

script=".github/scripts/versioning.py"

pr="$(gh api "repos/${GITHUB_REPOSITORY}/commits/${SHA}/pulls" \
  --jq '[.[] | select(.merged_at != null and .base.ref == "dev")][0] // empty')"
if [ -n "$pr" ]; then
  number="$(jq -r '.number' <<<"$pr")"
  branch="$(jq -r '.head.ref' <<<"$pr")"
  url="$(jq -r '.html_url' <<<"$pr")"
  labels="$(jq -r '[.labels[].name] | join(",")' <<<"$pr")"
  gh api --paginate "repos/${GITHUB_REPOSITORY}/pulls/${number}/commits" \
    --jq '.[].commit.message | split("\n")[0]' \
    | grep -v '^style: auto-format' >"${RUNNER_TEMP}/items.txt" || true
  title="[${branch}](${url}) (#${number})"
else
  # A direct push to dev: one patch version for it.
  number=""
  branch="dev"
  labels=""
  git log -1 --format='%s' "$SHA" >"${RUNNER_TEMP}/items.txt"
  title="Прямой push в dev (${SHA:0:7})"
fi
increment="$(python3 "$script" kind --branch "$branch" --labels "$labels")"
echo "Branch ${branch}: ${increment} bump"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3 4 5; do
  git fetch --tags --force origin dev
  git reset --hard origin/dev
  old="$(python3 "$script" current)"
  new="$(python3 "$script" next --kind "$increment")"
  python3 "$script" apply --version "$new"
  uv lock
  python3 "$script" changelog --version "$new" --date "$(date -u +%F)" \
    --title "$title" --items-file "${RUNNER_TEMP}/items.txt"
  git add -A
  git commit -m "bump: version ${old} -> ${new} (${branch})"
  if git push origin HEAD:dev; then
    echo "Pushed v${new}"
    break
  fi
  if [ "$attempt" -eq 5 ]; then
    echo "dev kept moving; giving up after ${attempt} attempts" >&2
    exit 1
  fi
  sleep $((attempt * 5))
done

if [ -n "$number" ]; then
  gh pr edit "$number" --repo "$GITHUB_REPOSITORY" --title "v${new} (${branch})"
fi
