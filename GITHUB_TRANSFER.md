# Install or move Reference Desk

## Install the application from GitHub

After installing Python 3.12, Git, GitHub CLI, and Ollama, open PowerShell:

```powershell
gh auth login --web --git-protocol https
cd "$HOME\Documents"
gh repo clone FKENZOLS/reference-desk
cd reference-desk
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Backend auto
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Use `-Backend rocm` for AMD, `-Backend cuda` for NVIDIA, or `-Backend cpu` for
slow compatibility mode when automatic detection is unsuitable.

## Move PDFs, indexes, and notes

The GitHub repository intentionally excludes personal research data.

1. On the old computer, open **Documents** and select **Create backup**.
2. Copy the resulting backup ZIP to the new computer.
3. Install and start Reference Desk on the new computer.
4. Open **Documents** and restore the backup ZIP.

The backup contains PDFs, indexes, workspace notes, trash, quarantine, and
revision history. Keep it private.

## Update an existing installation

```powershell
cd "$HOME\Documents\reference-desk"
git pull --ff-only
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Backend auto
```

Then start the app normally.
