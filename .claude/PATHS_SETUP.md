# Setting Up Path Shortcuts

This guide helps you configure common data paths in `.claude/settings.json` so you don't have to type full paths every time.

## Quick Setup

Edit `.claude/settings.json` and update the `paths` section with your actual directories:

```json
"paths": {
  "data": {
    "test": "data/",
    "covid": "/path/to/your/covid/data",
    "samples": "/path/to/your/samples",
    "bruker": "/mnt/bruker_instrument/data"
  },
  "output": {
    "results": "results/",
    "exports": "exports/",
    "reports": "reports/"
  },
  "repos": {
    "r_package": "/Users/you/git/nmr.parser",
    "related": "/Users/you/git/other-nmr-tools"
  },
  "examples_data": {
    "covid_sample": "data/HB-COVID0001/10",
    "reference_experiment": "/path/to/known-good-experiment"
  }
}
```

## Path Types

### Data Paths
Where your NMR data lives:
- **test**: Small test dataset (usually in repo or nearby)
- **covid**: COVID study data
- **samples**: General sample data directory
- **bruker**: Direct connection to Bruker instrument

### Output Paths
Where results go:
- **results**: Analysis results (intermediate)
- **exports**: Final exports (CSV, parquet, DuckDB)
- **reports**: Generated reports and summaries

### Repository Paths
Related codebases:
- **r_package**: Original R nmr.parser package
- **related**: Other NMR analysis tools

### Example Data Paths
Specific experiments you use often:
- **covid_sample**: A specific COVID sample experiment
- **reference_experiment**: Known-good reference data

## How to Use

Once configured, instead of:
```bash
.venv/bin/python examples/parse_nmr_example.py /very/long/path/to/covid/data -v info
```

You can just say:
> "Run the parse_nmr example on the covid data with debug verbosity"

Claude will know to use the path from `settings.json`.

## Examples

**Instead of this:**
> "Read the experiment at /mnt/data/studies/covid-2024/samples/HB-COVID0001/10"

**Just say:**
> "Read the covid_sample experiment"

**Instead of this:**
> "Parse all data in /mnt/bruker_instrument/data/batch_2024_02"

**Just say:**
> "Parse the data in the bruker folder"

## Tips

1. **Use absolute paths** for data outside the repo
2. **Use relative paths** for data inside the repo (like `data/`)
3. **Add new shortcuts** for any paths you use frequently
4. **Update paths** when data moves or new datasets arrive
5. **Share paths** by committing settings.json (or keep local with .gitignore)

## Example Configurations

### Local Development
```json
"data": {
  "test": "data/",
  "covid": "../covid-samples/",
  "samples": "/Users/julien/nmr-data/samples"
}
```

### Server/Cluster
```json
"data": {
  "test": "data/",
  "covid": "/mnt/storage/covid-study/",
  "samples": "/mnt/storage/nmr-samples/",
  "bruker": "/mnt/instruments/bruker/data"
}
```

### Network Drive
```json
"data": {
  "test": "data/",
  "covid": "/Volumes/NMRData/covid/",
  "samples": "/Volumes/NMRData/samples/",
  "bruker": "smb://server/bruker/data"
}
```

## Adding New Shortcuts

To add a new shortcut:

1. Edit `.claude/settings.json`
2. Add to the appropriate section:
```json
"data": {
  "test": "data/",
  "mynewdata": "/path/to/my/new/data"
}
```
3. Save and use it: "Parse mynewdata"

## Updating CLAUDE.md

After setting up paths, you might also want to update `CLAUDE.md` with notes about what each path contains:

```markdown
## Our Data Layout

- **covid data**: HB-COVID study, 144 samples, plasma, 2020-2021
- **samples**: General metabolomics samples, various matrices
- **bruker**: Live instrument data, auto-processed
```

This helps Claude understand your data better!
