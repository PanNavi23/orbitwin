# docs/

[English](README.md)（以英文版為準）

依「你現在要做什麼」排列。旁邊有 `.zh-TW.md` 的文件有繁體中文版，以英文版為準。

## 比賽當天

| 文件 | 回答什麼 |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | 從安裝到上傳的逐步流程，含三道提交前閘門與停損規則 |
| [`WORKSPACE.md`](WORKSPACE.md) | 工作區：`orbitwin init`、`run`、`status`，公告值寫在 `announce.toml`，錯誤代碼 |
| [`ERROR_CODES.md`](ERROR_CODES.md) | `orbitwin: ERROR [<code>]:` 的每個代碼與結束狀態 |
| [`ANNOUNCEMENT.md`](ANNOUNCEMENT.md) | 公告值由兩人各自照原樣輸入一份：格式、兩份比對、與 script 的比對 |
| [`DESIGN_VARIABLES.md`](DESIGN_VARIABLES.md) | 規則允許隊伍自己設定的值（SC_Human 的 epoch）：`orbitwin set-epoch` 與它留下的紀錄 |
| [`LESSONS_2026-09-05.md`](LESSONS_2026-09-05.md) | 2026-09-05 初賽**量到**的八件事：驗證器讀最後一列、單位、係數如何反轉搜尋策略、可達上界 |
| [`SCORING_STRATEGY_AND_OPEN_QUESTIONS.md`](SCORING_STRATEGY_AND_OPEN_QUESTIONS.md) | 分數公式實際在決定什麼，以及仍待主辦書面確認的問題 |

## 金牌賽／銅牌賽

| 文件 | 回答什麼 |
|---|---|
| [`DUEL_ENGINE.md`](DUEL_ENGINE.md) | 六回合追逃賽的規則逐條對照、裁判與賽局求解器設計、未知項與待問主辦的問題 |
| [`DUEL_ENGINE_OPEN_FINDINGS.md`](DUEL_ENGINE_OPEN_FINDINGS.md) | 追逃賽引擎設計審查還沒結案的 finding，各自標明涉及的章節與實作階段 |

## 規則與假設

| 文件 | 回答什麼 |
|---|---|
| [`COMPETITION_POLICY.md`](COMPETITION_POLICY.md) | 程式碼實際編碼了哪些規則，以及由哪個模組執行 |
| [`ASSUMPTIONS_AND_GAPS.md`](ASSUMPTIONS_AND_GAPS.md) | 只用於分析種子的假設，與尚未取得許可的空白 |
| [`SOURCES.md`](SOURCES.md) | 主辦的初賽規則、0510 說明會簡報與 workshop 範本各是什麼、從哪裡來，以及 GMAT 的下載連結 |
| [`ASYNC_START_AND_INSERTION.md`](ASYNC_START_AND_INSERTION.md) | 兩艘船 epoch 不同與 EarthFixed 入軌：對齊段、時間原點、d_min 視窗 |
| [`RULE_PACKS.md`](RULE_PACKS.md) | 每一輪的規則是規則包裡的一個 TOML 檔，以 `orbitwin rules` 列出 |

## 數學與數值

| 文件 | 回答什麼 |
|---|---|
| [`DIFFERENT_PLANES_MATH.md`](DIFFERENT_PLANES_MATH.md) | 不同平面攔截的三維推導；為什麼不能退化成共面 Hohmann |
| [`NUMERICAL_RESOLUTION.md`](NUMERICAL_RESOLUTION.md) | 候選排序的數值解析度——多小的差異才是真的差異 |
| [`OOD_STRATEGY.md`](OOD_STRATEGY.md) | OOD 涵蓋到哪裡，以及它明確**不**承諾什麼 |

## 工程

| 文件 | 回答什麼 |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 資料流與信任邊界；parser 為何刻意只支援窄子集 |
| [`GMAT_IN_THE_LOOP.md`](GMAT_IN_THE_LOOP.md) | `solve` 怎麼用 GMAT 自己的飛行修正最好的幾個計畫 |
| [`GMAT_SETUP_LINUX.md`](GMAT_SETUP_LINUX.md) | 在 Linux 上裝公開 R2026a，以及它與主辦 build 的差別 |
| [`STACK_ANSWER_MAP.md`](STACK_ANSWER_MAP.md) | 哪幾層改動會改變答案，哪幾層不會（2026-08-26 實測） |
| [`PERFORMANCE.md`](PERFORMANCE.md) | 時間實際花在哪，以及哪些加速手段實測**不划算**（numba、GPU） |
| [`ADVERSARIAL_REVIEW.md`](ADVERSARIAL_REVIEW.md) | 0.3.0 的對抗性審查與當時未完成的 gate |
| [`audit/2026-08-19/`](audit/2026-08-19/) | 已修復的四個 P0 與其證據快照（歷史，非現行程式碼） |
