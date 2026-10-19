#!/usr/bin/env bash
# Usage: sbom-finalize.sh <input.cdx.json> <output.cdx.json>
#
# Prepares a CycloneDX SBOM for the archive: sets the publisher (SBOM_ORG_NAME,
# optionally SBOM_ORG_URL) as manufacturer and supplier (BSI TR-03183-2
# "creator of the SBOM"), optionally the product
# name (SBOM_NAME), the release version (SBOM_VERSION) and the SHA-512 of the
# shipped file (SBOM_ARTIFACT), then fails if a required SBOM-level field of
# BSI TR-03183-2 is missing.
# Image SBOMs from Syft lack a lifecycle phase and dependencies of the image
# itself; both are added: they are created in the build pipeline right after
# the image build (BSI requires phase "build"), and the image contains every
# listed component.
# Component-level gaps (e.g. upstream libraries without supplier) are only
# reported, as we cannot fix third-party metadata.
set -euo pipefail

input=$1
output=$2
: "${SBOM_ORG_NAME:?SBOM_ORG_NAME must name the publisher}"

tools="${RUNNER_TEMP:-/tmp}/sbom-tools"
mkdir -p "$tools"
fetch() { # name url sha256
  [ -x "$tools/$1" ] && return
  curl -sSfL -o "$tools/$1" "$2"
  echo "$3  $tools/$1" | sha256sum -c --quiet -
  chmod +x "$tools/$1"
}
fetch jq https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 \
  020468de7539ce70ef1bceaf7cde2e8c4f2ca6c3afb84642aabc5c97d9fc2a0d
fetch sbomqs https://github.com/interlynk-io/sbomqs/releases/download/v1.3.0/sbomqs-linux-amd64 \
  060e12b2afe3d5fcf31ef80cf27728224c95d12bcd417def12a09a594c518dac

hash=""
[ -n "${SBOM_ARTIFACT:-}" ] && hash=$(sha512sum "$SBOM_ARTIFACT" | cut -d' ' -f1)

"$tools/jq" --arg name "${SBOM_NAME:-}" --arg version "${SBOM_VERSION:-}" --arg hash "$hash" \
  --arg org_name "$SBOM_ORG_NAME" --arg org_url "${SBOM_ORG_URL:-}" '
  ({name: $org_name} + (if $org_url != "" then {url: [$org_url]} else {} end)) as $org
  | .metadata.manufacturer = $org
  | .metadata.supplier = $org
  # The purl would point to a package we never publish, and is stale once renamed.
  | if $name != "" then .metadata.component.name = $name | del(.metadata.component.purl) else . end
  | if $version != "" then .metadata.component.version = $version | del(.metadata.component.purl) else . end
  | if $hash != "" then .metadata.component.hashes = [{alg: "SHA-512", content: $hash}] else . end
  | .metadata.lifecycles //= [{phase: "build"}]
  | .metadata.component["bom-ref"] as $root
  | if any(.dependencies[]?; .ref == $root) then .
    else .dependencies += [{ref: $root, dependsOn: [.components[]?["bom-ref"]]}] end
' "$input" > "$output"

"$tools/sbomqs" compliance --bsi-v2 --basic "$output"
missing=$("$tools/sbomqs" compliance --bsi-v2 --json "$output" \
  | "$tools/jq" -r '.sections[] | select(.element_id == "SBOM" and .required and .score < 10) | .section_data_field')
if [ -n "$missing" ]; then
  echo "::error::$output misses required BSI TR-03183-2 fields: $(echo $missing)"
  exit 1
fi
