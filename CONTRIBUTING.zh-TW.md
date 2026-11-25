# 參與 OrbitWin 開發

[English](CONTRIBUTING.md)（以英文版為準）

本檔說明一個變更怎麼進入 OrbitWin：pull request 會怎麼處理、怎麼安裝與測試，以及
變更要守的規則。OrbitWin 做什麼見 [README.zh-TW.md](README.zh-TW.md)。

## Pull request 怎麼到達程式碼

本 repository 由一個私有的主線匯出，pull request 不在 GitHub 上合併。pull request
開在這裡，目標是 `main`。CI 在 pull request 上通過之後，變更才會被套用。接著維護者把
它套用到私有主線，匯出需要時會先調整它。匯出發布的是一份檔案樹，不是主線的
commits，所以下一個匯出的 commit 帶著你的變更，並以一行 `Co-authored-by:` 列出你。
維護者關閉 pull request 時附上那個 commit 的連結，並關閉它連結的 issue：pull request
沒有合併就關閉時，GitHub 不會關閉它以 `Fixes #` 指名的 issue。

## 安裝

需要 Python 3.11 到 3.14（CI 四個版本都測）、[uv](https://docs.astral.sh/uv/) 與
`make`。照 [README.zh-TW.md](README.zh-TW.md) 的「安裝」安裝。

CI 裝的也是這些，另外加上 [.github/workflows/ci.yml](.github/workflows/ci.yml) 的
Install 步驟列出的套件。CI 不安裝 OrbitWin 本身，而是從 `src` import（該檔的
`PYTHONPATH: src`）。可編輯安裝 `uv pip install --no-deps -e .` 給你 `orbitwin`
指令，CI 沒有這個指令。執行 OrbitWin 的測試用 `sys.executable -m orbitwin` 或呼叫
`orbitwin.cli.main`，不以名稱呼叫 `orbitwin`。在子行程裡執行它的測試，把
`PYTHONPATH` 設成 `src` 的絕對路徑：CI 的 `PYTHONPATH` 是相對的 `src`，子行程在
別的工作目錄就找不到它。

快速測試會執行它寫進 `TMPDIR` 的程式，所以 `TMPDIR` 不能在以 `noexec` 掛載的
檔案系統上。

### 改釘住的版本

1. 在 `requirements.txt` 或 `requirements-dev.txt` 中釘住它的那一個裡改。
2. 改 `vendor/wheels/constraints.txt` 裡同一個套件的那一行。
3. 照 [vendor/wheels/README.md](vendor/wheels/README.md) 重新抓離線 bundle。抓取
   執行 `$PYTHON -m pip download`，所以要指定一個有 pip 的直譯器：
   `PYTHON=/path/to/python3 bash scripts/fetch_offline_wheels.sh`。`uv venv` 建的
   venv 沒有 pip。重新抓取因為與 `constraints.txt` 的某一行衝突而失敗時（新版本需要
   某個相依套件的另一個版本），把那一行也改掉，再重新抓取。把 `constraints.txt` 與
   抓取寫出的 `SHA256SUMS.txt` 一起 commit。

快速測試把釘住的版本、bundle 的 manifest 與文件寫的 wheel 數量互相核對：見
`tests/unit/test_wheel_constraints.py`。

## 執行檢查

`make` 的各個 target 與 CI 跑什麼，見 [README.zh-TW.md](README.zh-TW.md) 的「開發」。
CI 不跑 `make test-slow`、`make fuzz` 與 `make test-gmat`：跑你的變更碰到的那幾個，
並在 pull request 裡寫明跑了哪些。

`make test` 在單一 process 裡跑。要用多個 CPU 跑快速測試，照
[.github/workflows/ci.yml](.github/workflows/ci.yml) 的 Fast suite 步驟跑：用該檔
Install 步驟加裝的套件，以及它的 `env` 設定的變數。

直接跑 `pytest` 不等於 `make test`。`pyproject.toml` 沒有排除任何 marker，所以直接跑
`pytest` 也會收進 `slow` 與 `gmat` 測試，而 `gmat` 測試在沒有 GMAT 時失敗。

### 對 GMAT 的測試

```bash
export GMAT_CONSOLE=/opt/GMAT-R2026a/bin/GmatConsole
export GMAT_STARTUP=/opt/GMAT-R2026a/bin/gmat_startup_file.txt
make test-gmat
```

標了 `gmat` 的測試會執行 stock GmatConsole R2026a，下載連結在 README.zh-TW.md 的
「安裝」。[docs/GMAT_SETUP_LINUX.md](docs/GMAT_SETUP_LINUX.md) 寫了這些路徑假設的
目錄配置、一個 preflight，以及測試需要的 `TMPDIR`。新的 GMAT 測試要標上 `gmat`，
並使用 `tests/conftest.py` 的 `gmat` fixture，檢查這兩個變數的是同一檔的
`gmat_install`。

其他測試不需要 GMAT，它們跑的是測試替身 `scripts/fake_gmat_console.py`。

## 變更的規則

- 每個行為變更都附 regression test：沒有這個變更時失敗、有了才通過的測試。也要加
  對抗性測試：變更附近、仍須拒絕的輸入。不論什麼理由都不可刪除對抗性測試。不可為了
  讓測試通過而降低安全邊際。
- 遇到 parser 不認得的 GMAT 語法就 fail closed。parser 刻意只接受一個窄的子集
  （[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 的 "Parser boundary"）。其他語法
  它一律丟出 `ParseError`，印成 `orbitwin: ERROR [parse]:`
  （[docs/ERROR_CODES.md](docs/ERROR_CODES.md)），什麼都不改寫。放寬子集的變更，要為
  新的寫法加測試，也要為鄰近、仍須拒絕的寫法加測試。
- 不可吞掉例外後回報成功。
- 不可把分析（二體）估計標成 GMAT verified。
- 每次 GMAT 執行都有自己的執行目錄，兩次執行不共用 ReportFile 路徑。
- 診斷用的 subscriber（為了檢查一次執行而加的 ReportFile 與 `Report` 指令）不可
  進入要提交的 script。
- 單位寫進名稱：`inc_deg`、`dt_s`、`policy.dv_limit_km_s`。
- `src/orbitwin` 的每個函式都要有型別註記：`make typecheck` 對這個套件跑 mypy，
  設定是 `disallow_untyped_defs = true` 與 `python_version = "3.11"`
  （`pyproject.toml`），所以 Python 3.11 沒有的呼叫 mypy 也會抓出來。
- seed 固定、排序穩定，同一次執行跑兩次得到同一個答案。
- ruff 保持零 finding，不加新的 `noqa`。
  [tests/unit/test_policy_docs_match_code.py](tests/unit/test_policy_docs_match_code.py)
  的 ruff 測試把樹限制在刻意保留的抑制（`KEPT_NOQA`）與審過的 `[tool.ruff]` 表
  （`RUFF_CONFIG`）。新的抑制或對這個表的變更寫進那個檔案，附上理由，放在同一個
  pull request。
- 每個 `.script` 檔都只用 ASCII：GMAT R2026a 拒絕含有大於 127 位元組的 script
  （`tests/unit/test_source_script_encoding.py`）。
- `src/orbitwin/` 的每個模組留在自己那一邊：兩種賽制共用的核心只 import 核心，
  排位賽的模組與對決（`src/orbitwin/duel/`）互不 import。新模組要加進
  `tests/cross_profile/test_import_boundaries.py` 裡它所屬的集合。
- 絕不 commit 競賽主辦單位的材料（[NOTICE](NOTICE)）。需要它們的測試在公開的
  repository 裡會略過（`tests/conftest.py` 的 `organizer_file`）。新的測試用你自己
  的 script 或 `examples/` 裡的 script 寫。

## 新的競賽

一個競賽的規則書是規則包裡的一個規則集：[docs/RULE_PACKS.md](docs/RULE_PACKS.md)
說明怎麼寫、什麼需要寫程式，以及哪些指令接受哪些規則集。在隨附的規則包裡加規則集，
會改變 `orbitwin rules --json` 印出的清單，而 `tests/unit/test_rule_packs.py` 釘住
這份清單，所以也要更新它。比賽日的指令 `orbitwin init` 與 `orbitwin run` 只接受
`src/orbitwin/workspace.py` 的 `DAY_RULES`。

## 文件

文件以英文寫。旁邊有 `.zh-TW.md` 檔的文件有繁體中文版，章節相同，以英文版為準
（[docs/README.zh-TW.md](docs/README.zh-TW.md)）。改了有中文版的文件，就在同一個
pull request 裡逐節改中文版。

不會寫繁體中文時，在 pull request 裡說明。中文版裡有些值會拿來與程式碼核對：哪一個
需要你的新值時，快速測試會失敗並指出檔案，請把那裡的那一行也改掉。

## Commits

標題行寫成英文的祈使句，說這個變更做什麼。

每個 commit 都以 `git commit -s` 簽署。它加上的 `Signed-off-by:` 行，表示你對這個
commit 確認 [Developer Certificate of Origin 1.1](https://developercertificate.org/)。
除非你另外聲明，你有意提交以供納入的貢獻適用 Apache License 2.0
（[LICENSE](LICENSE) 第 5 條）。

## Pull requests

- 連結這個 pull request 解決的 issue（`Fixes #123`）。
- 說明改了什麼、為什麼，以及跑了哪些檢查與結果。pull request 範本會問這兩件事。
