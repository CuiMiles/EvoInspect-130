#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/submission/drafts/auxiliary_material.zip}"
temporary="$(mktemp -d /tmp/evoinspect-auxiliary-XXXXXX)"
trap 'rm -r "${temporary}"' EXIT
package_root="${temporary}/Cuisine_智检演化130_其他"
mkdir -p "${package_root}/models" "${package_root}/src" "${package_root}/evidence_summary"

cp -a "${repo_root}/submission/auxiliary_src/." "${package_root}/"
mkdir -p "${package_root}/src/evoinspect"
for source_file in __init__.py errors.py evaluation.py guarded_adapt.py inference.py provenance.py sequence.py video.py video_evaluation.py; do
  cp "${repo_root}/src/evoinspect/${source_file}" "${package_root}/src/evoinspect/${source_file}"
done
cp "${repo_root}/reports/experiments/remote-gtx2060-20260830-instance49225420/efficientad-m-onnx-fp16/efficientad-m.fp16.onnx" \
  "${package_root}/models/efficientad_m_fp16.onnx"
cp "${repo_root}/evidence/remote_gtx2060_benchmark_20260830.json" \
  "${package_root}/evidence_summary/gtx2060_benchmark.json"
cp "${repo_root}/evidence/video_event_evaluation_20260830.json" \
  "${package_root}/evidence_summary/video_event_metrics.json"
cp "${repo_root}/reports/experiments/guarded-adapt-replay-20260827T194500-cpu/report.json" \
  "${package_root}/evidence_summary/guarded_adapt_metrics.json"

PYTHONPATH="${repo_root}" /home/CuiMinghao/envs/evoinspect-efficientad/bin/python \
  "${repo_root}/scripts/build_auxiliary_demo_assets.py" \
  --run-dir "${repo_root}/reports/experiments/efficientad-m-frozen-20260828T095200Z-shared23/runs/efficientad-m-bottle-s143-20260828T095200Z-shared23" \
  --output-dir "${package_root}/demo"

libreoffice --headless --convert-to pdf --outdir "${package_root}" \
  "${package_root}/README_FIRST.html" >/dev/null
chmod +x "${package_root}/run_demo.sh" "${package_root}/run_demo.py"
(
  cd "${package_root}"
  find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
)

mkdir -p "$(dirname "${output}")"
temporary_zip="${temporary}/auxiliary_material.zip"
(cd "${temporary}" && zip -q -r "${temporary_zip}" "$(basename "${package_root}")" -x '*/__pycache__/*' '*.pyc')
size="$(stat -c '%s' "${temporary_zip}")"
(( size <= 209715200 )) || { printf 'ZIP exceeds 200 MiB: %s\n' "${size}" >&2; exit 3; }
mv "${temporary_zip}" "${output}"
unzip -tq "${output}" >/dev/null
printf '%s %s bytes\n' "${output}" "${size}"
