# OrbitWin

[![CI](https://github.com/PanNavi23/orbitwin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PanNavi23/orbitwin/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[English](README.md)（以英文版為準）

OrbitWin 為 GMAT script 規劃並驗證多次脈衝的太空船攔截。給它主辦的雙船攔截題目
script，它在競賽的 ΔV 上限內搜尋機動排程，把最好的幾個交給 stock GmatConsole 重播，
再寫出要上傳的 script 與報告。它為國家太空中心（TASA）2026 年軌道設計競賽而寫；每屆
競賽的規則是一個規則包（[docs/RULE_PACKS.md](docs/RULE_PACKS.md)）。

搜尋用二體模型，只負責提出候選。哪些候選可行由 GmatConsole 重播決定，分數由主辦的
驗證器決定。OrbitWin 不宣稱任何答案是最佳解。

## 安裝

需要 Python 3.11 以上與 GMAT R2026a
（[下載](https://sourceforge.net/projects/gmat/files/GMAT/)）。

```bash
git clone https://github.com/PanNavi23/orbitwin && cd orbitwin
uv venv && . .venv/bin/activate
uv pip install -r requirements-dev.txt && uv pip install --no-deps -e .

make test          # 約 5000 個測試，單一 process；耗時見「開發」
orbitwin --version
```

沒有 `uv` 就用 `python3 -m venv .venv` 與
`pip install -r requirements-dev.txt && pip install --no-deps -e .`。請從
`requirements-dev.txt` 安裝，不要從 `pyproject.toml`：套件的 metadata 只寫版本範圍，
測試與比賽日的工具要的是釘死的版本。`requirements-dev.txt` 把
`vendor/wheels/constraints.txt` 當作 constraints 讀入，它連那些套件依賴的套件也釘住，
也就是離線 bundle 與 CI 安裝的那一組。numpy、SciPy 或 PyYAML 不是 `requirements.txt`
釘的版本時，`orbitwin` 與比賽日的 scripts 拒絕執行；設 `ORBITWIN_ALLOW_UNPINNED=1`
可照樣執行。從 wheel 安裝的套件沒有 `requirements.txt`，不檢查
（[docs/WORKSPACE.md](docs/WORKSPACE.md)）。

比賽機器可能沒有網路。事前在有網路的機器上抓好離線 bundle，隨 repo 帶去：

```bash
bash scripts/fetch_offline_wheels.sh   # 39 個 wheel，涵蓋 Python 3.11 到 3.14
bash scripts/bootstrap_offline.sh      # 在比賽機器上：只裝 SHA256SUMS.txt 列出的 wheel，--require-hashes
. .venv/bin/activate                   # 每開一個終端機都要先跑
```

## 比賽日

完整流程見 [docs/RUNBOOK.md](docs/RUNBOOK.md)。

最短的路是工作區（[docs/WORKSPACE.md](docs/WORKSPACE.md)）：一個資料夾放隊號、GMAT
路徑、當天公告的值與官方 script，`orbitwin run` 把它們交給 `orbitwin day`，跑完第 1 到
7 步。

```bash
orbitwin init contest --team-number <隊號> --expect-commit <前一天選定的 commit> \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"
cp OFFICIAL.script contest/input/
$EDITOR contest/announce.toml          # 先填 T_max，係數公告後再填
orbitwin run contest                   # 每個讀回停一次；兩人核對後加 --confirm-readback <步驟>
orbitwin status contest                # 缺什麼、跑到哪、上傳哪兩個檔
```

下面是同樣的步驟手動執行，也就是 `orbitwin day` 實際跑的指令，以排位賽為例（規則
2026-10-06，距離項 2026-11-04 修訂）。初賽的設定封存在 `config/history/preliminary/`。

```bash
. .venv/bin/activate   # 下面的 python 與 orbitwin 都要用 .venv 裡的
export GMAT_CONSOLE=/opt/GMAT/bin/GmatConsole
export GMAT_STARTUP=/opt/GMAT/bin/gmat_startup_file.txt

# 1. 保存原始輸入，設為唯讀
cp OFFICIAL.script input/original/official.script && chmod a-w input/original/official.script

# 2. 公告的 T_max 寫進當天的 config（不要手改 YAML），兩人核對印出的值。
#    若警告 gmat.workers 多於核心數，以 --gmat-workers <不超過它給的數> 另寫新檔，
#    第 3、4 步都用它取代 run/day.yaml。
python scripts/ranking_day_config.py --output run/day.yaml --horizon-s <T_max> --team-number <隊號>

# 3. 檢視，再做第一次搜尋（run/search-01）
orbitwin inspect input/original/official.script --config run/day.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" --output run/scenario.json
bash scripts/competition_day.sh input/original/official.script run run/day.yaml

# 4. 係數公告後：寫新的 config（2026-11-04 修訂版定 ΔV 為 m/s、C_r 為 km），
#    再搜一次（run/latest），然後重排
python scripts/ranking_day_config.py --output run/day-scored.yaml --horizon-s <T_max> \
  --kt-per-s <kt> --ct-s <Ct> --kv <kv> --cv <Cv> --delta-v-unit m/s --cr-km <C_r> \
  --day-config run/day.yaml
bash scripts/competition_day.sh input/original/official.script run run/day-scored.yaml
orbitwin rank input/original/official.script --config run/day-scored.yaml \
  --candidates run/latest/candidates.json --top 10 --output run/rank-scored.json \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"

# 5. 認證選定的候選
orbitwin certify input/original/official.script --config run/day-scored.yaml \
  --candidates run/latest/candidates.json --candidate C-xxxxxxxxxxxxxxxx \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP" --runs 3 --output run/final
```

有 console 時，每個指令先跑 state probe，初始狀態以 GMAT 讀到的為準。入軌狀態寫成
EarthFixed 的模板，沒有 console 讀不了。

係數公告後要再搜一次：排位賽的搜尋依係數挑起點。`orbitwin rank` 幾秒內就能重排任何
已存的 archive，某一項整個平掉時會警告，通常是係數的單位填錯了。`ranking_day_config.py`
也印出兩個上限，任何解的最高分與任何 miss 的最高分，以及距離項在 10、100、1000、
10000 km 的值和降到 49 分、25 分的位置（約 `5 + 0.2·C_r` 與 `5 + 1.048·C_r`）。給了
`--zero-burn-dmin-km`（第一次 solve 結束時印出的零機動最近距離），它也印出那個距離的
分數，讓兩人核對 C_r。miss 以最近距離計分，公式為 `100/(1 + e^{((d − 5)/C_r)²})`，
可能勝過昂貴的命中。

## 上傳前的檢查

2026-09-05 那天，repo 裡每一道檢查都通過了，上傳的成績還是少了 22 分。下面三個工具
是從那天來的。

```bash
# 地形：哪些到達時刻可達、要編列幾次脈衝（ΔV 那一欄是上界，不是下限）
python scripts/safe_delta_v_floor.py official.script --config run/day-scored.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"

# 獨立稽核：從主辦的初始狀態密集重播，不信求解器自己的回報
# （重播是二體；力模型更豐富時它不判攔截與分數，AUDIT 那一行會寫明）
python scripts/audit_burn_table.py official.script burns.json --config run/day-scored.yaml \
  --gmat-console "$GMAT_CONSOLE" --startup-file "$GMAT_STARTUP"

# 上傳前：讀那份要上傳的 .txt，確認最後一列在 5 km 內。報告檔名照模板的佔位符，
# 隊號 54 時 ODC_2026_TeamXX.txt 變成 ODC_2026_Team54.txt；certify 會印出檔名。
python scripts/check_report_rows.py run/final/submission/ODC_2026_Team54.txt \
  --config run/day-scored.yaml --certificate run/final/submission/certification.json
```

最後一項就是那 22 分的教訓：主辦的驗證器從報告的最後一列取完成時間，沒有去找第一次
進入球內的那一列。script 跑滿 `4TA`，就被判成從未攔截
（[docs/LESSONS_2026-09-05.md](docs/LESSONS_2026-09-05.md)）。

`check_report_rows.py` 要求報告是證書列出的 bytes，script 是證書裡提交的那一份，給了
`--delimiter` 等格式旗標也照查。證書欄位型別不對，或數值是 NaN、無限大，回傳 2。
`--script` 讓它照該 ReportFile 的 Delimiter、WriteHeaders、FixedWidth 與 Add 清單切欄。
這支 script 在本 checkout 內執行（它讀 `src/`），需要 Python 3.11 以上與專案的 venv；
numpy、SciPy 或 PyYAML 不是釘的版本時它拒絕執行，回傳 2。

排位賽有兩點不同（runbook 第 7 步）：

- `audit_burn_table.py` 用二體重播。在規則第 7 節的擾動模型下，它仍判單次上限、總量、
  間隔、時間窗，以及到 T_max 為止的地表與高度牆。它不判攔截與分數：在 R1 上，它把一個
  GMAT 認證的命中算成 19.97 km 的 miss。
- 計分的 miss 要對照 T_max。帶 `--config` 時（如上），T_max 取自設定檔的
  `policy.horizon_s`，不要再給 `--horizon-s`：與設定檔不同的值回傳 2。不帶 `--config` 時，
  以 `--horizon-s` 給當天設定檔的 `policy.horizon_s`（不是 certification.json 裡求解器時鐘的
  `horizon_s`）；沒給時它印 `VERDICT: unchecked` 並回傳 1。最後一列在 T_max 時，
  `check_report_rows.py` 回傳 0，並以最近的一列計價。

## 輸出檔名由證據決定

一次執行寫出的 script 叫什麼名字，不由你指定：

| 候選 stage | 檔名 | 需要的證據 |
|---|---|---|
| `analytical` / `renderable` | `analytical_candidate.script` | 二體模型與事件精修 |
| `public_gmat_verified` | `public_gmat_candidate.script` | 任一 stock GMAT 成功重播 |
| `organizer_gmat_verified` | `organizer_gmat_candidate.script` | `gmat.organizer_compatible` 且指紋相符 |
| `certified` | `team_solution.script` | 上一列，加上 exact script 重播、report parity 與重複的 exact runs |

`solve` 從不寫出 `team_solution.script`。`summary.json` 記錄這次執行做了什麼，沒有結果
時也記錄原因。

## 專案結構

```text
src/orbitwin/    套件本身：parser、Lambert、事件偵測、renderer、GMAT runner、CLI
config/          quick、standard、deep、OOD 與排位賽設定
scripts/         上傳前檢查、fuzz、release 驗證、GMAT test double
docs/            數學、policy、runbook、實測記錄、審查（索引：docs/README.md）
tests/           單元、整合與 slow OOD 測試
examples/        我們自己的兩次機動模板、排位賽演練 R1 到 R9、不同軌道面的場景
notebooks/       Jupyter 前端，只畫圖；不進 release 壓縮檔
input/original/  比賽當天放官方 script 的地方（第 1 步後唯讀；gitignore）
vendor/          離線 wheel bundle 與 GMAT 安裝說明（binary 不進 git）
output/          執行產出（gitignore）
```

## 開發

```bash
make test          # 約 5000 個快速的單元與整合測試，單一 process
make test-isolated # 同一批，每個檔案一個 process（慢 7%，2026-08-21）-> output/make/
make test-slow     # 慢的搜尋：OOD、多脈衝、攔截後可達性
make fuzz          # 1000 個固定種子的 Lambert OOD 案例
make lint          # ruff；有任何 finding，CI 就失敗
make typecheck     # mypy；有任何 error，CI 就失敗
make coverage      # 96% 的 statement（2026-11-04，含 worker 與 subprocess）-> output/make/
make validate      # release 驗證 -> output/validation/（不進 git）
make package       # manifest、ZIP，並在解壓後的副本再跑一次快速測試 -> dist/
```

CI 在每次 push 到 main 與每個 pull request 時，於 Python 3.11、3.12、3.13、3.14 跑
`test`，並跑 `checks`（先 `make lint` 再 `make typecheck`，lint 失敗時 typecheck 照跑）。兩個 job 都
跑在 GitHub 的 `ubuntu-latest`。有自己 runner 的私有 repo 以 repo 變數指定，值是 runner
標籤的 JSON 陣列：`ORBITWIN_RUNS_ON_TEST` 給 `test`，`ORBITWIN_RUNS_ON_CHECKS` 給
`checks`，例如 `gh variable set ORBITWIN_RUNS_ON_TEST --body '["linux","x64","my-runner"]'`。
`ORBITWIN_RUNS_ON_TEST` 只能列這個 repo 自己的 runner；要用 GitHub 的 runner，就把它刪掉。
公開的 repo 不看這兩個變數。在 [`.github/workflows/ci.yml`](.github/workflows/ci.yml) 裡，
`jobs:` 上方的註解列出要讓 fork 開的 PR 跑不到你的 runner，organization 與 repo 該怎麼
設定，以及重跑時讀到什麼；`test` job 上的註解寫出 test 的每個 leg 在各種 runner 上的時間
上限。

`make validate` 把證據寫在執行它的那台機器的 `output/validation/`。證據不進 git，也不
進 release 壓縮檔；其中 `environment.json` 記下它跑的 commit（`commit`），以及當時
tracked 檔案有沒有改動（`tracked_changes`）。`make package` 拒絕 tracked 檔案有改動的
樹，也拒絕 HEAD 既不是（先 fetch 過的）origin/main、也不是 `v<版本>` tag 的 commit。所以
順序是：`git fetch`、checkout origin/main、`make validate`、讀過 `output/validation/`、
`make package`。驗證要跑 30 到 80 分鐘；這段時間 main 若有變動，`make package` 會拒絕，
這時在新的 origin/main 上重跑驗證，讓 `commit` 等於要打包的 commit。

`make validate` 先確認 `PYTHON` 指定的直譯器 import 得到專案依賴、版本合乎 pin。
`PYTEST_ARGS` 只傳給其中的 coverage 那一步，而且只接受 `-n`、`--numprocesses`、
`--maxprocesses`、`--dist`、`--maxschedchunk` 與 `--durations`（例如裝了 pytest-xdist 時
`PYTEST_ARGS="-n 16"`）：選擇測試的參數會讓證據變少，一律拒絕；`PYTEST_ADDOPTS` 或
`PYTEST_PLUGINS` 有值時也拒絕。同一個 checkout 一次只能跑一個（`output/validate/lock`）。
中途失敗或被中斷時，`output/validation/` 還原成執行前的樣子，該次的輸出移到
`output/validate/failed-<時間>-<pid>`。只有 SIGKILL 會留下 `output/validate/previous`，
裡面是執行前的證據；下一次會拒絕執行，直到有人比對、還原需要的部分並刪掉它。
2026-11-04 在 i5-14500 上，整個 `make validate PYTEST_ARGS="-n 16"` 花了 1794 秒（coverage
399 秒）。離線 bundle 沒有 pytest-xdist，所以同一天從它安裝、斷網
（`bwrap --unshare-net`）跑完花了 4711 秒。

`test-isolated` 與 `coverage` 的 log、XML、coverage 資料與 pytest 快取寫在 gitignore 的
`output/make/`（`REPORT_DIR` 可改）。`make clean` 執行 `scripts/clean.sh`，它不需要
Python，清的東西與 `make package`（`build_release.clean()`）相同：快取、coverage 資料、
`src/*.egg-info` 與 bytecode；`dist/`、`build/`、`output/`、`.venv` 與其他 checkout 都不碰。
有一個測試確認兩份實作的結果相同。

在 CI 裡，`make test` 以 pytest-xdist 平行：在綁 16 顆 CPU 的自有 runner 上，Python
3.11、3.13、3.14 各花 164、158、257 秒（2026-11-02 的 main）；2026-11-12 在同一台上，每個
leg 從開始到結束約 6 分鐘。GitHub 給公開 repo 的 runner 是 4 個 vCPU，會慢上好幾倍。單一
process 慢得多：2026-11-03 在 i5-14500（14 核 20 執行緒）、CPython 3.13.14 上，當時 main 的
2476 個測試花了 1024 秒，一分鐘 load 平均 5.5。`make package` 在解壓副本的測試輸出停住
10 分鐘、或總共超過 4 小時時停下（單一測試跑滿 300 秒時 faulthandler 會把它的 stack 寫進
log，所以卡住的測試最多 900 秒後就被停下）。它先送 SIGINT，再結束同一個 process group
裡的所有 process，並把已印出的輸出留在封存檔旁邊。Ctrl-C、SIGTERM、SIGHUP 也一樣。

## 已知邊界

- Parser 只接受 GMAT 語法的一個窄子集，也就是競賽模板用的那些
  （[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 的「Parser boundary」）：顯式 `Create`、
  每個物件欄位一個 assignment、恰好兩艘太空船、顯式的 `ImpulsiveBurn.Element1/2/3`、以經過
  時間為終止條件的單行 `Propagate`（同步，或分開但每艘船的終止條件相同）加上模板自己的 epoch
  對齊段、slotted 模板交替的 `Propagate`/`Maneuver`（open 模板沒有 `Maneuver`，機動由我們
  加上），以及只作用於繪圖 subscriber 的 `Toggle`、`PenUp`、`PenDown` 與只作用於 XYPlot 的
  `MarkPoint`、`ClearPlot`。遇到其他語法就停下，不會猜。
- 搜尋用二體模型。模板的力模型若有 J2、drag、SRP 或第三體，只有 GMAT 重播能告訴你
  一個排程是否可行。
- 恰好共線或反跖的 Lambert 幾何沒有唯一的轉移平面。Cartesian shooting fallback 處理這種
  情形，最後仍由 GMAT 重播決定。
- stock GmatConsole 拒絕建立 `OpenFramesInterface` 的 script。這時 `certify` 與
  `baseline` 改飛只拿掉圖的那些行、並逐行對照提交 script 證明過的副本
  （`gmat.console_copy`，排位賽設定檔開啟），上傳的 script 保留圖。`solve` 飛的診斷副本
  也不含圖。`gmat.console_copy` 關著、或證明不了副本時，`certify` 停在 `exact-run-01`，
  報告只能從 GMAT GUI 來（[docs/RUNBOOK.md](docs/RUNBOOK.md) 第 6b 步）。
- 沒有任何東西能證明答案是全域最佳。多個 Lambert 分支、Sobol 起點、多脈衝 fallback、
  Pareto archive 與 OOD 測試只降低漏掉解的機會。
- 主辦允許改哪些欄位、失敗任務怎麼計分、分數如何取位、驗證器的容許誤差，都還沒全部
  確認（[docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md](docs/SCORING_STRATEGY_AND_OPEN_QUESTIONS.md)）。

## 授權

Apache-2.0（[LICENSE](LICENSE)）。主辦提供的題目、規則與範本不在此授權範圍內（[NOTICE](NOTICE)）。

套件原名 `tasa-orbitwin`，改名為 `orbitwin` 時版本號沒變，同樣是 0.3.0
（[RELEASE_NOTES.md](RELEASE_NOTES.md)），兩者只能從名稱分辨。環境裡裝過舊名稱的話，先
`uv pip uninstall tasa-orbitwin`，再照上面安裝。兩個名稱裝在同一個環境時，解除安裝舊
名稱會刪掉 `orbitwin` 指令（不是 `-e` 的安裝連 `orbitwin/*.py` 也刪掉），`uv pip list`
卻仍列出 `orbitwin`；遇到這種情形，以 `uv pip install --reinstall --no-deps -e .` 修復。
