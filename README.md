# Es予測ダッシュボード

日本の代表的なEs(スポラディックE層)観測4地点(稚内・国分寺・山川・大宜味)を対象に、実際のFT8伝播データ(PSKReporter)とNOAA宇宙天気(Kp指数)、季節・時刻の統計モデルを組み合わせてEs活動指数を算出するダッシュボードです。

[ameblo.jp/jl7khn の EDFS (Es Dynamics Forecast System) 記事](https://ameblo.jp/jl7khn/entry-12976221040.html) を参考にした個人プロジェクトです。

## 仕組み

- `scripts/fetch_pskreporter.py` — PSKReporterから直近15分の10m/6m帯FT8受信報告を取得し、受信者グリッドロケータから4地点付近の受信数を集計
- `scripts/fetch_noaa.py` — NOAA SWPCから最新のKp指数を取得
- `scripts/fetch_nict.py` — NICTの電離圏概況モバイルページ(15分更新)から各地点の実測foEs値をスクレイピング(実測のE層臨界周波数、最優先の証拠として扱う)
- `scripts/climatology.py` — 季節・時刻から統計的なEs活動ベースラインを算出
- `scripts/build_index.py` — 上記を統合し、`history.csv` に蓄積した実測データが十分たまるとRidge回帰モデルを自動学習し、`data.json` を生成
- `index.html` — `data.json` を読み込んで表示する静的ダッシュボード(GitHub Pagesで公開)
- `.github/workflows/update.yml` — 15分ごとにGitHub Actions上で上記パイプラインを実行し、結果をリポジトリにコミット

## 公開設定(初回のみ手動)

Settings → Pages → Build and deployment → Source を **Deploy from a branch** にし、Branch を **main / (root)** に設定してください。
