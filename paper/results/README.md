# Performance results

Create one directory per benchmark campaign:

```text
results/
└── YYYY-MM-DD-short-name/
    ├── metadata.md
    ├── raw/
    ├── summary.csv
    └── analysis.md
```

Copy [TEMPLATE.md](TEMPLATE.md) to `metadata.md` before starting a campaign. Store unedited command
output and tool-generated CSV files under `raw/`. Treat raw files as immutable after the campaign;
fix analysis errors in derived files and record the correction.

Machine descriptions live under [machines/](machines/). A campaign should link the exact machine
record and add any temporary state such as power mode, background load, cooling, affinity, VM/WSL
limits, and available memory.

Do not commit enormous traces or profiles without deciding whether Git is the appropriate artifact
store. Record a stable external location and checksum when an artifact must remain outside the
repository.
