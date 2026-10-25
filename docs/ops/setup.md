# Machine Setup

How to rebuild the Nebula machine from a fresh Windows 11 install, or set it up on a different PC. Follow the sections in order; each ends with a check. The record of how this machine was actually set up, with dates and findings, is [phase0-notes.md](phase0-notes.md).

Versions below were verified on 2026-10-04. Newer versions are usually fine, except for the model runtimes, which are pinned in [`config/runtime.lock.toml`](../../config/runtime.lock.toml) and change only after a benchmark re-run.

## 0. Hardware and drive layout

| Drive | Role on this machine |
| --- | --- |
| NVMe (Samsung 980 1 TB) | `F:`: Windows, the boot loader, the user profile, and hot Nebula data in `F:\Nebula\` |
| HDD (WD Blue 1 TB) | `D:`: cold data in `D:\NebulaCold\`, next to personal files |
| Old SATA SSD (WD Green 240 GB) | Retired: SATA port disabled in BIOS ([PHASE0_PLAN](../PHASE0_PLAN.md) Section 3.1) |

GPU: RTX 4070 12 GB. CPU: i7-10700K. RAM: 32 GB.

Paths in the repo assume `F:` for hot data and `D:` for cold data. On a machine with other letters, either assign these letters in Disk Management or change the paths in `bench/profiles/*.toml`, the `dir` entries in `config/*.lock.toml`, and the scripts' defaults.

## 1. Windows basics

1. Install the current NVIDIA driver (Game Ready or Studio). Requirement: CUDA 12.4 or later.
   - Check: `nvidia-smi` shows the driver and `CUDA Version: 12.4` or higher. (Verified: 596.49, CUDA 13.2.)
2. Create a Windows recovery USB with the Media Creation Tool (8 GB+), and boot from it once to check it works.
3. Create the data folders:

   ```powershell
   'state','logs','models','worktrees','build','setup','runtime' | % { New-Item -ItemType Directory -Force "F:\Nebula\$_" }
   'models-archive','logs-archive','research','backups-tmp','backups-local','downloads','wsl' | % { New-Item -ItemType Directory -Force "D:\NebulaCold\$_" }
   ```

## 2. Toolchain

Install from an ordinary (non-elevated) PowerShell, then **open a new terminal** so PATH changes apply.

```powershell
winget install -e --id Git.Git
winget install -e --id GitHub.cli
winget install -e --id astral-sh.uv
winget install -e --id OpenJS.NodeJS.LTS
winget install -e --id Kitware.CMake
winget install -e --id Ninja-build.Ninja
winget install -e --id Gitleaks.Gitleaks
winget install -e --id Rclone.Rclone
winget install -e --id Mozilla.sccache
winget install -e --id Tailscale.Tailscale
winget install -e --id Rustlang.Rustup
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Then the Rust components and cargo tools:

```powershell
rustup default stable-x86_64-pc-windows-msvc
rustup component add rustfmt clippy
cargo install --locked cargo-nextest cargo-sweep
```

The CUDA Toolkit is **not** needed: the prebuilt llama.cpp binaries bundle the CUDA runtime. Install it only if llama.cpp ever has to be built from source (PHASE0_PLAN task 3.2).

**Check**: each command prints a version.

| Command | Verified version |
| --- | --- |
| `git --version` | 2.52.0 |
| `gh --version` | 2.89.0 |
| `uv --version` | 0.10.4 |
| `node --version` | 24.14.1 |
| `cmake --version` | 4.3.2 |
| `ninja --version` | 1.13.2 |
| `gitleaks version` | 8.30.1 |
| `rclone version` | 1.75.1 |
| `sccache --version` | 0.18.0 |
| `rustup show active-toolchain` | `stable-x86_64-pc-windows-msvc` |
| `rustc --version` | 1.99.0 |
| `cargo nextest --version` | 0.9.146 |
| `cargo sweep --version` | 0.8.0 |
| `tailscale version` | 1.102.4 |
| `cl` (in a **Developer PowerShell for VS 2022**; it is not on the normal PATH) | MSVC 19.44 x64 |
| `wsl --version` | WSL 2.7.3 |
| `"$env:ProgramFiles\smartmontools\bin\smartctl.exe" --version` (installed in step 4, not on PATH) | 7.5 |

## 3. Environment variables and WSL2

User-scope variables, so builds and caches stay on `F:` and worktrees share one target directory:

```powershell
[Environment]::SetEnvironmentVariable('CARGO_TARGET_DIR', 'F:\Nebula\build\target', 'User')
[Environment]::SetEnvironmentVariable('SCCACHE_DIR', 'F:\Nebula\build\sccache', 'User')
[Environment]::SetEnvironmentVariable('SCCACHE_CACHE_SIZE', '6G', 'User')
[Environment]::SetEnvironmentVariable('RUSTC_WRAPPER', 'sccache', 'User')
```

WSL2: enable it with `wsl --install --no-distribution` (reboot if asked). No distro is needed in Phase 0. Cap its memory so the model server keeps its RAM headroom; the cap is shared by every WSL distro, including Docker Desktop's. Create `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=6GB
```

**Check**: in a new terminal, `$env:CARGO_TARGET_DIR` prints `F:\Nebula\build\target`; `wsl --status` shows default version 2.

## 4. Admin setup (elevated PowerShell)

Clone the repo first (Section 6) to get the scripts, or download them from GitHub.

1. **Page file and SMART monitoring:** `scripts/phase0-ws1-admin.ps1`. It moves the page file to `F:\pagefile.sys` (4 GB initial, **32 GB maximum**), installs smartmontools, takes a first SMART snapshot into `F:\Nebula\state\smart\`, and creates two SYSTEM scheduled tasks: `Nebula\SMART monthly` and `Nebula\SMART on disk error`. Reboot afterwards.
   - The large maximum is for **commit charge**, not paging. On Windows, llama-server commits about as much system memory as the VRAM it uses (~10.7 GB for `standard`). A busy desktop commits ~33 GB. With a 12 GB maximum, the commit limit was ~44 GB, and allocations inside llama-server failed with `bad allocation`. The file starts at 4 GB and only grows when needed. `scripts/set-pagefile.ps1` changes just the page file.
   - Check: `Get-CimInstance Win32_PageFileUsage` lists only `F:\pagefile.sys`; `(Get-CimInstance Win32_OperatingSystem).TotalVirtualMemorySize / 1MB` is about 64 (GB); `F:\Nebula\state\smart\` has one JSON per drive; in an **elevated** shell, `schtasks /query /tn "Nebula\SMART monthly"` finds the task (SYSTEM tasks are hidden from non-elevated queries).
   - The script copies `smart-snapshot.ps1` to `F:\Nebula\setup\`, where the tasks run it from.
2. **SSH over Tailscale:**
   1. Sign in to Tailscale on the desktop and on the laptop.
   2. Put the laptop's **public** key in `F:\Nebula\setup\macbook.pub`. It is kept outside the repo. On the Mac: `cat ~/.ssh/id_ed25519.pub`.
   3. Run `scripts/phase0-ssh-setup.ps1`. It installs OpenSSH Server, writes the key to `administrators_authorized_keys` (an administrator account ignores `~/.ssh/authorized_keys`), turns off password login, and limits port 22 to the Tailscale ranges.
   - Check from the laptop: `ssh <user>@<desktop-tailscale-name>` logs in with the key; `ssh -o PubkeyAuthentication=no <user>@<desktop>` gets `Permission denied (publickey)`; with Tailscale off on the laptop, `ssh` to the desktop's LAN address times out.
   - **Surfshark logs Tailscale out.** Keep the VPN off, or add `tailscaled.exe` and `tailscale-ipn.exe` to Surfshark's Bypasser list.

## 5. GitHub accounts and credentials

Only needed once per machine; the repo settings (Section 9) persist on GitHub.

1. `gh auth login` as the owner account (HTTPS or SSH). Add an SSH key to the owner account: `origin` uses `git@github.com:Xydra01/Nebula.git`, and workflow files can only be pushed this way (the bot token has no `workflow` scope).
2. Store the bot token: `powershell -ExecutionPolicy Bypass -File scripts/store-bot-token.ps1`. It prompts for the `Nebula-dev-bot` classic token (scope `public_repo` only, 90-day expiry), saves it in Credential Manager as `nebula/github_bot_token`, and checks it against the API. Never paste the token anywhere else.
   - Check: `scripts/bot-push-pr.ps1` can push a branch and open a PR.

## 6. Repo

```powershell
cd F:\Users\<you>
git clone git@github.com:Xydra01/Nebula.git "Project Neutron"
cd "Project Neutron"
powershell -ExecutionPolicy Bypass -File scripts/install-hooks.ps1
cd bench; uv sync; cd ..
```

`install-hooks.ps1` sets `core.hooksPath=.githooks`; the hooks run gitleaks on every commit and push.

**Check**: the local checks in [CONTRIBUTING.md](../../CONTRIBUTING.md) pass (`gitleaks git --redact`, then `ruff` and `pytest` in `bench/`).

## 7. Model runtimes

Both are prebuilt llama.cpp releases for Windows x64, CUDA 12.4. Exact releases, install folders and archive SHA-256 hashes are in [`config/runtime.lock.toml`](../../config/runtime.lock.toml).

| Runtime | Used for | Source |
| --- | --- | --- |
| `llama-prism` (PrismML fork, `prism-b10743-adfffbe`) | Bonsai (all profiles) | `PrismML-Eng/llama.cpp` releases |
| `llama-stock` (upstream, `b11342`) | Gemma 4 fallback, embedding model | `ggml-org/llama.cpp` releases |

For each runtime, download the two archives listed under `.archives` (the build and `cudart-llama-bin-win-cuda-12.4-x64.zip`) to `D:\NebulaCold\downloads\`, check their hashes, and unpack both into `install_dir`:

```powershell
Get-FileHash D:\NebulaCold\downloads\<archive>.zip -Algorithm SHA256   # must match runtime.lock.toml
Expand-Archive D:\NebulaCold\downloads\<archive>.zip -DestinationPath <install_dir>
```

Don't use the CUDA 13.x builds of the PrismML fork; they crash on Windows (PrismML llama.cpp #222).

**Check**: `<install_dir>\llama-server.exe --version` prints the pinned build; `--list-devices` shows `CUDA0: NVIDIA GeForce RTX 4070`.

## 8. Models

Every file, its Hugging Face repo, size and SHA-256 is in [`config/models.lock.toml`](../../config/models.lock.toml). Only the entries **without** an `archived` key belong on `F:`:

| Folder in `F:\Nebula\models\` | File | Role |
| --- | --- | --- |
| `bonsai2-27b\` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, `…-mmproj-Q8_0.gguf` | `long`, `lean`, `vision` |
| `bonsai2-27b\` | `Ternary-Bonsai-2-27B-PTQ1_0-kv-bias.gguf` | Generated locally (below) |
| `bonsai2-27b-mtp\` | `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` | `standard` (ADR-006) |
| `fallback-moe\` | `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | Fallback (ADR-005) |
| `embedding\` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | CPU embedding server |

Download each one from `https://huggingface.co/<repo>/resolve/main/<file>` to a `.partial` name, rename it when complete, then check the hash:

```powershell
curl.exe -L --fail -o "<file>.partial" "https://huggingface.co/<repo>/resolve/main/<file>"
Rename-Item "<file>.partial" "<file>"
(Get-FileHash "<file>" -Algorithm SHA256).Hash.ToLower()   # must match models.lock.toml
```

About 33 GB in total. Expect ~8 MB/s from Hugging Face, so allow an hour or more.

**KV bias for PTQ1_0** (used by `long` and `lean` with the q4_0 KV cache). Generate the calibration corpus, then the bias:

```powershell
cd bench
uv run python -c "from nebula_bench.corpus import calibration_text; open(r'F:\Nebula\models\bonsai2-27b\calibration-corpus.txt', 'w', encoding='utf-8').write(calibration_text())"
$M = 'F:\Nebula\models\bonsai2-27b'
F:\Nebula\runtime\llama-prism\b10743-adfffbe\llama-kv-mean-center.exe -m "$M\Ternary-Bonsai-2-27B-PTQ1_0.gguf" -f "$M\calibration-corpus.txt" -o "$M\Ternary-Bonsai-2-27B-PTQ1_0-kv-bias.gguf" -ngl 99 -c 512
```

A regenerated bias may not be byte-identical to the one in the lock file; if the hash differs, record the new one. The server must run with `LLAMA_ATTN_ROT_DISABLE=1` whenever the bias is loaded (the bench launcher sets it).

**Archived models** (on `D:\NebulaCold\models-archive\`, not needed to run): PQ2_0 without MTP, the grafted PTQ1_0+MTP file, and the fallback candidates that lost. The graft can be rebuilt with `bench/nebula_bench/graft_mtp.py`.

## 8a. The `nebula` command

From the repo root, build and install `nebula.exe` and `nebula-daemon.exe` to `F:\Nebula\bin\`. Re-run this after pulling changes, with the daemon stopped (`nebula daemon stop`) so the files aren't locked.

```powershell
.\scripts\install-nebula.ps1
```

Put `F:\Nebula\bin` on the user PATH once, then open a new terminal:

```powershell
[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';F:\Nebula\bin', 'User')
```

Then `nebula daemon start`, `nebula doctor`, `nebula chat`. `nebula --help` lists everything.

### Daemon auto-start at logon

To start the daemon automatically after reboot and logon (Issue #38):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register-daemon-task.ps1
```

This registers a Task Scheduler task (`\Nebula\Nebula daemon (logon)`) under your user account with a 30-second delay so it does not compete with desktop startup. Use `-Unregister` to remove it.

## 9. GitHub repo settings (only if the repo is recreated)

These live on GitHub, not on the machine. Current state, for reference:

- Public, MIT licence. Secret scanning and push protection on.
- Ruleset `protect-main` on the default branch:
  - PR required, with 1 approval from a code owner (`CODEOWNERS`: `* @Xydra01`) after the last push
  - stale approvals dismissed
  - no force-push, no deletion
  - repository admins can bypass via PR only
  - required status check **CI ok** from GitHub Actions
- `Nebula-dev-bot` is a collaborator with write access.

## 10. Final checks

From the repo's `bench\` folder:

```powershell
uv run nebula-smoke --profile pq2mtp        # standard profile: chat, tool calls, json_schema, offload, auth
uv run python -m nebula_bench.embed         # embedding server: CPU only, retrieval sanity, contention
```

Close other GPU users first. Ollama, LM Studio and a `llama-cpp-turboquant` build are installed on this machine, and anything holding VRAM makes llama-server fail to load. The desktop alone uses ~1.4 GB of VRAM.

Expected: smoke checks all pass. `standard` uses ~10.2 GB of VRAM at 32K context and generates ~60–84 tokens/s; the embedding check ranks 8/10 queries first and uses no VRAM.
