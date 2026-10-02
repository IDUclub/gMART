#!/usr/bin/env bash
# Bump the version inside a PR branch right before auto-merge merges it into dev.
#
# Merges the current dev into the branch (version files, uv.lock and CHANGELOG.md conflicts are
# taken from dev and then rewritten), sets the next version after dev for this PR (see
# versioning.py), refreshes uv.lock, writes the PR's CHANGELOG.md section, pushes a "bump:"
# commit to the branch and marks the "version" status of the new head as passed, which lets
# auto-merge merge. Running it again after dev moved replaces the bump instead of stacking one.
#
# The passing "version" status is set with VERSION_STATUS_TOKEN, not GITHUB_TOKEN: GitHub starts
# no workflow for events caused by GITHUB_TOKEN, and that includes the push of a merge that
# auto-merge made because GITHUB_TOKEN passed the last required check - the dev release would
# not run. Pending and failure statuses use GITHUB_TOKEN.
#
# Run it from a copy outside the work tree (it checks out the PR branch, which may predate it).
# Env: GH_TOKEN (contents, statuses, pull requests), VERSION_STATUS_TOKEN (commit statuses),
# GITHUB_REPOSITORY, PR (number).
set -euo pipefail

script="$(dirname "$0")/versioning.py"
context="version"

status() {   # status <sha> <state> <description>
  gh api --silent "repos/${GITHUB_REPOSITORY}/statuses/$1" \
    -f state="$2" -f context="$context" -f description="$3" \
    -f target_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
}

pr="$(gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR}")"
branch="$(jq -r '.head.ref' <<<"$pr")"
url="$(jq -r '.html_url' <<<"$pr")"
labels="$(jq -r '[.labels[].name] | join(",")' <<<"$pr")"
head="$(jq -r '.head.sha' <<<"$pr")"
# A run that starts late (e.g. approved after the merge) must not bump a closed PR: the push
# would recreate its deleted branch.
if [ "$(jq -r '.state' <<<"$pr")" != "open" ]; then
  echo "PR #${PR} is no longer open - nothing to bump"
  exit 0
fi
if [ -z "${VERSION_STATUS_TOKEN:-}" ]; then
  status "$head" failure "Нет секрета VERSION_STATUS_TOKEN: мердж не запустил бы выкатку на dev"
  echo "::error::Repository secret VERSION_STATUS_TOKEN (commit statuses) is not set" >&2
  exit 1
fi
status "$head" pending "Поднимается версия перед мерджем"

increment="$(python3 "$script" kind --branch "$branch" --labels "$labels")"
title="[${branch}](${url}) (#${PR})"
echo "PR #${PR} (${branch}): ${increment} bump"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
# Files every bump rewrites; their conflicts with dev are resolved by the bump itself.
owned="$(python3 - <<'PY'
import tomllib
config = tomllib.load(open("pyproject.toml", "rb"))
files = config.get("tool", {}).get("commitizen", {}).get("version_files", [])
print("\n".join(["pyproject.toml", "uv.lock", "CHANGELOG.md", *(f.split(":")[0] for f in files)]))
PY
)"
owned="$(tr -d '\r' <<<"$owned")"

for attempt in 1 2 3; do
  git fetch --tags --force origin dev "$branch"
  git checkout -q -B "$branch" "origin/${branch}"
  if ! git merge -q --no-edit origin/dev; then
    conflicts="$(git diff --name-only --diff-filter=U)"
    foreign="$(grep -vxF -f <(printf '%s\n' "$owned") <<<"$conflicts" || true)"
    if [ -n "$foreign" ]; then
      git merge --abort
      status "$head" failure "Конфликт с dev: $(echo $foreign | cut -c1-100)"
      echo "::error::PR conflicts with dev outside version files: ${foreign}" >&2
      exit 1
    fi
    # Only version lines, lock entries and changelog sections collide: redo the merge taking
    # dev's side of the colliding hunks only (the PR's other changes in these files stay); the
    # bump below writes this PR's version, lock and section on top.
    git merge --abort
    git merge -q --no-edit -X theirs origin/dev
  fi

  new="$(python3 "$script" next --kind "$increment" --ref origin/dev)"
  base="$(python3 "$script" current --ref origin/dev)"
  python3 "$script" apply --version "$new"
  if [ -f uv.lock ]; then
    uv lock   # pip projects (requirements.txt) have no lock to refresh
  fi
  gh api --paginate "repos/${GITHUB_REPOSITORY}/pulls/${PR}/commits" \
    --jq '.[].commit.message | split("\n")[0]' \
    | grep -vE '^(bump: |style: auto-format|Merge (remote-tracking )?branch )' \
    >"${RUNNER_TEMP}/items.txt" || true
  python3 "$script" changelog --version "$new" --date "$(date -u +%F)" \
    --title "$title" --items-file "${RUNNER_TEMP}/items.txt"
  git add -A
  git diff --cached --quiet || git commit -q -m "bump: version ${base} -> ${new}"
  if [ "$(gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR}" --jq '.state')" != "open" ]; then
    echo "PR #${PR} was closed meanwhile - not pushing"
    exit 0
  fi
  if git push -q origin "HEAD:${branch}"; then
    break
  fi
  if [ "$attempt" -eq 3 ]; then
    status "$head" error "Не удалось запушить bump, перезапустите workflow"
    exit 1
  fi
  sleep $((attempt * 5))   # someone pushed to the branch meanwhile: redo on top of it
done

sha="$(git rev-parse HEAD)"
if ! GH_TOKEN="$VERSION_STATUS_TOKEN" status "$sha" success "v${new}"; then
  status "$sha" failure "VERSION_STATUS_TOKEN не принят (истёк?): обновите секрет"
  echo "::error::VERSION_STATUS_TOKEN could not set the commit status - renew the secret" >&2
  exit 1
fi
gh pr edit "$PR" --repo "$GITHUB_REPOSITORY" --title "v${new} (${branch})" \
  || echo "::warning::could not rename PR #${PR} to v${new} (${branch})"
echo "PR #${PR} ready to merge as v${new}"
