# Research Artifact Provenance

## Authoritative source

Original private repository: `tesorieror/llm_uml_ui`

Branch: `main`

Pinned source commit:
`38385954a22bb23d10484a3c53edd511b0098277`

The source repository has not been modified as part of this distribution preparation.

## Original staging package

The verified staging copy contained 9,150 copied source files, initially totaling approximately 93.76 MiB.

Source-to-staging SHA-256 checks reported 9,150 identical files and zero mismatches.

`SHA256SUMS.txt` in the staging copy records that historical comparison. The staging package subsequently received separate editorial and distribution metadata.

## Public source package

The selected public package initially contained 3,178 files, approximately 17.9 MiB, and excluded NetCDF binaries, raw LLM responses, and generated code outputs.

All 3,178 files were verified byte-identical by SHA-256 to the selected staging files.

`SHA256SUMS_PUBLIC.txt` is the distribution-specific integrity manifest. It is updated when public documentation changes; it is not the original source-provenance manifest.

The source code and experimental files were not recalculated or modified during this publication preparation.

## Separately prepared data archives

### NetCDF analysis artifacts

`ist_netcdf_artifacts.zip` preserves 18 NetCDF files under their original relative paths, plus `netcdf_manifest.csv`.

Archive SHA-256:
`f3b0ab0e0ed8839e939f3f116829443b5f6d1dcea4a4d19e669abfe3610c85a2`

The 18 original files total approximately 298.24 MiB uncompressed and include 12 unique file hashes. All archive entries were checked after ZIP creation.

### Generation outputs

`ist_generation_outputs.zip` preserves 1,500 original LLM responses, 4,479 extracted code files, and `generation_outputs_manifest.csv`.

Archive SHA-256:
`4d5291ccc4baddf51a63f45446b4d6456867de773178ee991bd170bdee58ae96`

All 5,979 scientific entries were verified after ZIP creation.

## Scope and limitations

- Artifact checks verify content identity and packaging, not semantic or statistical correctness.
- Original calculations were not rerun during packaging.
- A wheel build and basic import test succeeded.
- Independent end-to-end replication has not been established.
- Both archives were subsequently published on Zenodo; their displayed MD5 checksums matched the corresponding local files during preparation.
- The replication GitHub repository is public; the depositor confirmed publication of https://doi.org/10.5281/zenodo.23668583.
