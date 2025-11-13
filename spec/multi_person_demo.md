全体の要件
tools\demo\demo.pyをマルチパーソン推論可能にすること。コードはtools/demo/demo_multi.pyに作成すること。コンフィグはhmr4d\configs\demo.yamlを再利用すること。
現在は    data = load_data_dict(cfg)が１人分かつ、pred = model.predict(data, static_cam=cfg.static_cam)が１人を前提としている。
human検知 -> tracking -> pose -> vit featuresを
人数分trackingまでしてから、以下pose -> vit feature -> model.predict(data, static_cam=cfg.static_cam)を人数分回す。

demo.pyのGet bbx tracking resultセクションで
torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)このように保存しているが、これをtorch.save({"id": id, "bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)とする。
そしてidごとにループする。

---

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
現在はTrackerを用いて以下のようにbbx_xyxy、bbx_xysを検出している。
tracker = Tracker()
bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge

マルチパーソン推論するために、Tracker()にget_multi_trackメソッドを追加する。
これはuse top1 trackではなく、任意の人物を指定できるようにする。
人物の指定についての方法としてmatplotlibのUIをメソッド内で出してユーザーの選択を求めるようにする。
ユーザー選択画面のしようとしては、track_historyで取得したbboxとidをフレームにオーバーレイして、どのidを選択するかゆだねる。
id選択はユーザーに数字を入力させる。
UIの実装はhmr4d/utils/ui/で行う。以下のように使用する。
parse track_history & use top1 track
id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
<-ここに挿入してユーザーが指定したidをreturn
track_id = id_sorted[0] # これは消去
以下は同じで、get_multi_trackメソッドはbbx_xyxy_multi_trackを返す。

---

実装結果
- `hmr4d/utils/ui/tracker_selection.py`でmatplotlibベースのプレビューUIとCLIフォールバックを追加し、複数人物のトラックIDを選択可能にした。
- Trackerに`get_multi_track`を実装し、UIで選ばれたIDごとに補間済みの`bbx_xyxy`テンソルを返すようにした。また`torch.save({"id": id, ...})`形式で保存するユーティリティを追加。
- `tools/demo/demo_multi.py`を新規作成し、選択した各人物の前処理・推論・描画を個別ディレクトリで実行、最後に2画面動画をマージするパイプラインを整備した。
