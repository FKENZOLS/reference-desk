# Moving Reference Desk to another computer

The repository contains application source only. PDFs, indexes, notes, logs,
quarantine, revisions, and corpus backups are excluded by `.gitignore`.

## Publish a private repository

Install GitHub CLI, sign in with `gh auth login`, then run:

```powershell
.\scripts\publish_github.ps1 -Repository reference-desk -Visibility private
```

On the AMD computer, clone the repository and run:

```powershell
git clone https://github.com/YOUR-NAME/reference-desk.git
cd reference-desk
.\scripts\setup.ps1 -Backend rocm
.\start.ps1
```

Use `-Backend auto` to detect NVIDIA, AMD, or CPU. Use `-Backend cuda` to force
the NVIDIA build. The Windows ROCm package requires Python 3.12 and a GPU/driver
supported by AMD's current PyTorch-on-Windows release.

## Transfer without Git

Create a source-only archive:

```powershell
.\.venv\Scripts\python.exe scripts\export_release.py
```

Copy `release/reference-desk-source.zip` to the other computer, extract it, and
run the setup command above. To move the actual corpus and notes as well, use
**Create backup** in the Documents page and restore that separate backup on the
new computer.

## Downloadable GitHub releases

Push a tag such as `v1.0.0`. The included GitHub workflow creates a checked,
source-only ZIP and attaches it to the release:

```powershell
git tag v1.0.0
git push origin v1.0.0
```
