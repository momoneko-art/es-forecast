# 開発メモ (Development Log)

このファイルは、Es予測ダッシュボードの設計判断・修正履歴・未対応事項を記録する開発者向けメモです。
（次にこのプロジェクトを触るAIセッション、および将来の自分自身が経緯を追えるように残しています）

## アーキテクチャ概要

- `scripts/fetch_pskreporter.py` — PSKReporterから直近15分のFT8受信報告(10m/6m)を取得
- `scripts/fetch_nict.py` — NICT電離圏概況ページから各地点の実測foEs(E層臨界周波数)をスクレイピング。最優先のground-truth証拠として扱う
- `scripts/fetch_noaa.py` — NOAA SWPCから (1) 現在のKp指数の実測値、(2) 今後24時間以内のKp"予報"のピーク値、の両方を取得
- `scripts/fetch_tropo.py` — Open-MeteoのGFSデータから対流圏ダクト(トロポダクト)指数を計算。地表付近の気圧面(1000/975/950/925/900/850/700hPa)で気温減率を見て、負の勾配(逆転層)を検出
- `scripts/climatology.py` — 季節・時刻からの統計的Es活動ベースライン + `nict_floor_from_foes()` (実測floor)
- `scripts/heatmap.py` — 全国グリッドへの空間補間(気候ベースライン+PSKReporter+NICT floor)
- `scripts/fetch_jetstream.py` — 【実験・未反映】250hPaジェット気流風速の記録専用(下記「ジェット気流の記録」参照)
- `scripts/build_index.py` — 上記を統合しGitHub Actions上で15分ごとに実行、`data.json`を生成
- `es-forecast-agent/agent.py` — build_index.pyと同等のロジックを持つ、ユーザーのPCで常時稼働するスタンドアロン版(GitHub Contents API経由でdata.json等を直接push)。**gitでは管理していない**ファイル
- `index.html` — `data.json`を読み込んで表示する静的ダッシュボード(GitHub Pages)

### es_indexの算出式(要点)
```
combined_boost = 1 - (1-evidence_boost) * (1-nict_boost)   # PSKReporterとNICTのnoisy-OR
es_index = clima * (1 + 0.6 * combined_boost)
es_index = max(es_index, nict_floor_from_foes(foes_mhz))   # NICT実測は気候ベースラインでキャップされない
es_index = clamp(0, 100)
```
`nict_floor_from_foes()`はMUF(oblique)≈3×foEsの経験則に基づき、foEsをそのまま0-100スケールの下限値に変換する(climatology.py参照)。

### Kp予報の活用
`kp_for_modifier = max(実測Kp, 今後24時間以内のKp予報ピーク)` を`climatology_index()`に渡すことで、稚内(aurora_sensitive)などオーロラ性Esが起きやすい局は、実際にKpが上がる前から予報ベースで先行して指数が上がる。`aurora_forecast_active`フラグでUIにバッジ表示。

### NICT単発欠測の引き継ぎ(2026-09-02追加)
NICTのスクレイプが1サイクルだけ`unknown`になる(電離層観測の瞬間的な穴、または一時的なページ取得失敗)ケースに対応。直前の「実測OK」だった値を`NICT_CARRY_FORWARD_MAX_SECONDS`(45分)以内なら引き継ぎ、`stale_minutes`をUIに表示する。**元の測定時刻を保持したまま**経過時間をカウントするため、欠測が連続しても正しく期限切れになる。

### データ取得状況パネル(2026-09-02追加)
`build_index.py`/`agent.py`の`summarize_status()`が、各上流データソース(PSKReporter・NICT・NOAA Kp実測・NOAA Kp予報・GFSダクト)ごとに`ok`/`warn`/`error`の平易な状態を`data.json`の`system_status`に出力し、`index.html`のヘッドライン直下にピル表示する。**ユーザーが対処するための機能ではなく**(外部サービスの障害はユーザー側でどうにもできないため)、数値が控えめに見える理由を一目で分かるようにする、また不具合報告時のスクリーンショットから即座に原因箇所を特定できるようにするための表示。

### ジェット気流の記録(実験・2026-09-02追加、未反映)
`scripts/fetch_jetstream.py`(agent.pyでは`fetch_jetstream()`関数として同等ロジックをミラー)が、Open-MeteoのGFSから250hPa(ジェット気流高度)の風速を4地点平均で取得し、`history.csv`の`jet250_kmh`列と`data.json`の`jet250`に記録する。**es_indexやヒートマップの計算には一切使っていない**、将来のバックテスト用のデータ収集のみの機能。2026-09-02時点でhistory.csvはまだ1日分程度しか無く、相関を検証できる段階ではないため、まずはデータを溜める段階。

## 修正履歴 (このセッションでの主な変更)

| 日付 | 内容 |
|---|---|
| 2026-09-01 | **es_index計算式の欠陥修正**: 気候ベースラインが低い時間帯だと、NICTの強い実測値があってもes_indexが上がらなかった問題。`nict_floor_from_foes()`を追加し、実測値が気候ベースラインを下限突破できるように修正 |
| 2026-09-01 | **トロポダクト指数の過小評価バグ修正**: 気圧面のサンプリングが粗すぎ(1000/925/850/700hPaで650-1500mの間隔)、実際の薄い(200-500m)ダクト層が平均化されて消えていた。Open-Meteoが提供する近地表の気圧面を全て使うよう拡張(1000/975/950/925/900/850/700hPa)。`INDEX_VERSION`定数で旧式のキャッシュを強制的に再取得させるゲートも追加 |
| 2026-09-01 | 自局Es交信チャンスパネルの表示順を、自局登録の直後・地図の前に変更 |
| 2026-09-01 | 各局のトレンド矢印を直近4サイクル分の履歴として表示(↓→↑↑のような連続矢印)。2回連続上昇は赤色でハイライト |
| 2026-09-02 | **NOAA Kp"予報"(3日先)の組み込み**: 実測Kpだけでなく、今後24時間以内の予報ピークも見てオーロラ性Es局の指数を先行して上げるように |
| 2026-09-02 | **GitHub Actions push失敗の修正**: PC常駐エージェントとGitHub Actionsが同じmainブランチへ同時にpushしてrace conditionが起きていた問題。fetch+rebase+リトライ(Actions側)、sha再取得+リトライ(agent.py側)を追加。**PATに`Workflows: Read and write`権限が必要**で、これがないとワークフローファイル自体の更新がGitHubにブロックされる(2026-09-02にユーザーが権限追加、解決済み) |
| 2026-09-02 | **NICT単発欠測の引き継ぎ機能**: 上記「NICT単発欠測の引き継ぎ」参照。稚内で実際に1サイクルだけ欠測が起き、直前の11-12MHzの実測値が反映されなかった実例をきっかけに実装 |
| 2026-09-02 | **データ取得状況パネル追加**: 各上流データソースのok/warn/error状態を平易な言葉でダッシュボードに表示 |
| 2026-09-02 | **ジェット気流データの記録開始(実験・未反映)**: HRO/RMOB流星電波観測データの自動取得可否も調査したが、画像グラフのみ・個人運営サイトの寄せ集めで自動取得が非現実的と判断し保留。ジェット気流の方はOpen-Meteoで250hPa風速がそのまま取得できることを確認し、まずは`jet250_kmh`としてhistory.csvへの記録を開始(バックテストは数週間〜のデータ蓄積後) |

## 未対応・保留中の項目

- **ATOM Cam2連携**: 27.005MHz違法無線のATOM Cam2音声通知をフィードバックループに組み込む案(①手動ボタン／②公式SoraCam Webhook／③atomcam_tools自作改造)を3案提示済み。ユーザー未選択
- **短期トレンド外挿(精度向上ロードマップ①)**: 既存のtrend_historyと気候曲線の形状を使った短期予測。未実装
- **ジェット気流相関の検証(精度向上ロードマップ③)**: `jet250_kmh`の記録は開始済み(2026-09-02〜)。history.csvに数週間〜1シーズン分のデータが溜まったら、実際にEs発生と相関があるかバックテストして、根拠が確認できたらes_indexへの反映を検討する
- **HRO/RMOB流星電波観測データ**: 調査の結果、自動取得は非現実的と判断し保留(詳細は上記アーキテクチャ概要参照)。個人運営サイトの状況が変わった場合は再検討の余地あり
- 通知機能、MSTID画像解析、foF2データ追加(いずれも古いメモにあった未着手項目)

## デプロイ手順

`es-forecast-deploy`スキル参照。要点:
- クラウドのサンドボックスから直接GitHubへpushできないため、必ずユーザーのPC(OneDriveフォルダ`es-forecast`)経由でgit操作する
- GitHub Actionsが15分ごとに自動push(`[skip ci]`)しているため、push前に必ず`fetch`+`rebase`する
- PAT(Personal Access Token)はどのファイルにも保存せず、`$HOME/mnt/es-forecast/Personal Access Token.txt`から都度読み込んで使う
- `agent.py`はgit管理外。`SendUserFile`→`device_commit_files`でOneDriveの`es-forecast-agent/`フォルダへ直接転送し、ユーザーに再起動してもらう
