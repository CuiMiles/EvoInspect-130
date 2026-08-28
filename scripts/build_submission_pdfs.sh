#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
drafts="${repo_root}/submission/drafts"
temporary="$(mktemp -d /tmp/evoinspect-pdf-build-XXXXXX)"
trap 'rm -rf "${temporary}"' EXIT

libreoffice --headless --convert-to pdf --outdir "${temporary}" \
  "${repo_root}/submission/works_intro.html" \
  "${repo_root}/submission/project_document.html" >/dev/null
for name in works_intro project_document; do
  [[ -s "${temporary}/${name}.pdf" ]] || { printf 'missing generated %s.pdf\n' "${name}" >&2; exit 2; }
  # LibreOffice 6 inserts a blank leading page when importing these HTML files. Remove it
  # deterministically, but only after verifying that the first page contains no text.
  pages="$(pdfinfo "${temporary}/${name}.pdf" | awk '/^Pages:/ {print $2}')"
  first_text="$(pdftotext -f 1 -l 1 "${temporary}/${name}.pdf" - | tr -d '[:space:]')"
  if [[ -z "${first_text}" && "${pages}" -gt 1 ]]; then
    pdfseparate "${temporary}/${name}.pdf" "${temporary}/${name}-page-%d.pdf"
    remaining=()
    for page in $(seq 2 "${pages}"); do remaining+=("${temporary}/${name}-page-${page}.pdf"); done
    pdfunite "${remaining[@]}" "${temporary}/${name}-normalized.pdf"
    mv "${temporary}/${name}-normalized.pdf" "${temporary}/${name}.pdf"
  fi
  mv "${temporary}/${name}.pdf" "${drafts}/${name}.pdf"
done
pdfinfo "${drafts}/works_intro.pdf" | grep -q '^Pages:[[:space:]]*1$'
printf 'Generated %s and %s\n' "${drafts}/works_intro.pdf" "${drafts}/project_document.pdf"
