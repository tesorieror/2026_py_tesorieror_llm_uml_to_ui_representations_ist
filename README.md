# UML-to-Web-UI Representation Study: Replication Materials

This repository provides research software, input models, prompts, evaluation outputs, and analysis materials for an empirical study of CRUD-oriented Web UI generation by LLMs.

## Experimental scope

- 10 UML domain models.
- 5 representation conditions: UML/XMI, JSON Schema, XSD, SQL DDL, and Controlled Natural Language.
- 3 locally executed LLMs.
- 10 attempts per model, representation, and LLM combination.
- 1,500 generation attempts; 1,479 evaluable attempts.

## Distribution

The replication materials are divided between this source package and two separately prepared archival ZIP files.

### Source package (GitHub)

- `experiment/input/`: UML models and derived representations.
- `experiment/prompts/`: generation prompts.
- `experiment/metrics/`: evaluation materials.
- `experiment/analysis/`: included analytical files, excluding NetCDF artifacts.
- `src/es_uclm_iseresearch_uml_to_ui/`: Python research code.
- `SHA256SUMS_PUBLIC.txt`: hashes for the 3,177 files in the public package before adding the Zenodo deposit status document.

The public package does not contain the original LLM responses, extracted generated code, or NetCDF artifacts.

### Generation outputs archive (Zenodo, pending deposit)

`ist_generation_outputs.zip` contains:

- `experiment/raw_response/`: 1,500 files.
- `experiment/extracted_code/`: 4,479 files.
- `generation_outputs_manifest.csv`: path, size, and SHA-256.

Archive SHA-256:
`4d5291ccc4baddf51a63f45446b4d6456867de773178ee991bd170bdee58ae96`

### NetCDF analysis archive (Zenodo, pending deposit)

`ist_netcdf_artifacts.zip` contains 18 NetCDF artifacts and `netcdf_manifest.csv`, preserving the original relative file paths.

Archive SHA-256:
`f3b0ab0e0ed8839e939f3f116829443b5f6d1dcea4a4d19e669abfe3610c85a2`

## Restoring archived data

After obtaining both published archives, extract each ZIP into the root of a checkout of this replication repository.

The ZIP entries preserve their original `experiment/...` paths. Extraction restores the corresponding directories alongside the source package. Verify each ZIP against the SHA-256 above and use its internal manifest to check extracted files.

Zenodo DOI reserved: `10.5281/zenodo.23668583`. The deposit remains a draft; the DOI is not yet registered or publicly resolvable. See `ZENODO_DEPOSIT_STATUS.md`.

## Environment and limitations

The recorded Python version is 3.11.9.

`pyproject.toml` configures the replication Python package. `requirements-replication.txt` is the intended public dependency specification.

`requirements.txt` and `_requirements.txt` from the private source are not included in this operational package. See `HISTORICAL_REQUIREMENTS.md`.

A package wheel has been built and its basic import tested, but independent end-to-end installation and scientific reproduction have not been validated.

The file hashes demonstrate artifact identity, not independent reproduction of the reported results.

## Provenance and licensing

See `PROVENANCE.md` for source identity, packaging scope, and verification boundaries.

See `LICENSE_POLICY.md` for the distinct software and research-material licenses.

## Publication status

This repository is currently private. A Zenodo DOI has been reserved, but the dataset remains unpublished. Public release and final manuscript links remain pending.
