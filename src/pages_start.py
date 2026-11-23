from common import steps, tip, warn, need, items, tab, DL, OPEN, BADGE, VERSION, NB_FILE, GITHUB

PAGES = []

PAGES.append(dict(id='index', group='top', nav='トップ(何が作れるか)', title='Cast Frost 素材メーカーの使い方',
lead='Cast Frost Web の 🎆 演出・⚡ 発動 で使う、背景が透明な画像と動画を、Google Colab で作るノートブックです。',
kw='トップ 目次 できること はじめに 何が作れる',
body=f'''
<h2>何が作れるか</h2>
<div class="cards">
  <a class="card" href="g-image.html" style="border-top:5px solid #2f86d8"><span class="ci">🖼</span><b>背景が透明な静止画</b><small>PNG。スタンプ・アイコン・立ち絵・テロップの枠・光のエフェクトなど</small></a>
  <a class="card" href="g-video.html" style="border-top:5px solid #7c4dff"><span class="ci">🎬</span><b>背景が透明な動画</b><small>WebM(VP9)。炎・きらきら・煙・光などのループするエフェクト</small></a>
  <a class="card" href="g-i2v.html" style="border-top:5px solid #f0700f"><span class="ci">🎞</span><b>画像を動かす</b><small>1枚の画像を動かして、背景を抜いた透明な動画に</small></a>
  <a class="card" href="g-tools.html" style="border-top:5px solid #0f9d74"><span class="ci">🛠</span><b>持っている素材を加工</b><small>背景の切り抜き・黒を透明に・動画の仕上げ・描き直し</small></a>
  <a class="card" href="g-castfrost.html"><span class="ci">🎆</span><b>Cast Frost にそのまま入れる</b><small>🎆 演出・⚡ 発動の「＋ 素材を追加」に入る形(1つ30MBまで)で書き出します</small></a>
  <a class="card" href="g-prompt.html"><span class="ci">✍</span><b>日本語で頼める</b><small>プロンプトは日本語でOK。ノートブックの中で英語に直してから作ります</small></a>
</div>
<ul>
  <li><b>どこで動くか</b>:Google Colab(ブラウザの中で動く、Google のコンピューター)。自分の PC の性能は関係ありません。</li>
  <li><b>使うもの</b>:Google アカウント・<b>Colab Pro</b> の契約・Google ドライブ(作った素材の保存先)。</li>
  <li><b>操作</b>:セルの [[play]](再生ボタン)を押すと、セルの中に入力欄が出ます。プロンプトを書いて [[ui:✨ 作る]] を押すだけです。</li>
  <li><b>使っているモデル</b>:すべて商用で使えるライセンスのものです(配信の収益化にも使えます)。くわしくは [[p:licenses|使っているモデルとライセンス]]。</li>
</ul>

<h2>ノートブックを開く</h2>
<div class="dlBox">{OPEN}</div>
<p>ボタンを押すと、Google Colab でノートブックが開きます。そのままセルの [[play]] を押せば使えます(手順は [[p:start|準備]])。ボタンから開けば、<b>いつも最新の版</b>が開きます。作った素材は、自分の Google ドライブに保存されます。</p>
<p><small>ファイルとして手元に置きたい時は、<a href="dl/{NB_FILE}" download>ノートブック(v{VERSION})をダウンロード</a> することもできます。</small></p>

<h2>はじめての人へ</h2>
<p>[[p:start|準備]] → [[p:g-image|静止画を作る]] または [[p:g-video|動画を作る]] → [[p:g-castfrost|Cast Frost に入れる]] の順に読むと、最後まで進められます。使い終わったら [[p:g-stop|GPU を止める]] のを忘れずに。</p>
<div class="btnRow"><a class="bigBtn" href="start.html">🚀 準備を始める</a><a class="bigBtn ghost" href="g-prompt.html">✍ プロンプトの例を見る</a></div>

<h2>このサイトの見方</h2>
<ul class="linkList">
  <li><b>使い方</b>:[[p:g-image|静止画を作る]] / [[p:g-video|動画を作る]] / [[p:g-i2v|画像から動画を作る]] / [[p:g-tools|素材を加工する]] / [[p:g-result|作り直す・ダウンロード・保存先]] / [[p:g-castfrost|Cast Frost に入れる]] / [[p:g-prompt|プロンプトの書き方の例]] / [[p:g-stop|使い終わったら(GPU を止める)]]</li>
  <li><b>画面の項目</b>:[[p:r-image|🖼 静止画のセル]] / [[p:r-video|🎬 動画のセル]] / [[p:r-i2v|🎞 画像から動画のセル]] / [[p:r-tools|🛠 加工のセル]] / [[p:r-common|共通の部分(結果・記録)]] — 画面に出るすべての項目の説明</li>
  <li><b>資料</b>:[[p:size|30MB と長さ・画質]] / [[p:trouble|困った時]] / [[p:glossary|用語集]] / [[p:licenses|モデルとライセンス]] / [[p:changelog|更新履歴]]</li>
</ul>
'''))

PAGES.append(dict(id='start', group='top', nav='準備(はじめに)', title='準備(はじめに)',
lead='はじめて使う時に、上から順に進める手順です。2回目からも「Colab で開く」ボタンから開いて、「3. セルを押す」から進めます。',
kw='はじめ 最初 準備 Colab Pro ノートブック 開く Colab で開く GPU 選ぶ ドライブ つなぐ 許可 アップロード ダウンロード GitHub 最新',
related=['g-image', 'g-video', 'g-stop', 'trouble'],
body=f'''
<h2>0. 必要なもの</h2>
{items([
 ('Google アカウント', 'Colab と Google ドライブで使います(ログインしておきます)。'),
 ('Colab Pro の契約', 'A100・L4 などの GPU を使うのに必要です。使った分だけ「コンピューティング ユニット」が減ります(目安は [[p:g-stop|使い終わったら]])。'),
 ('Google ドライブの空き', 'ふつうに使うと <b>約12GB</b>(静止画のモデルをしまう分)。作った素材は 1つ 1〜30MB です。空きが少ない時は、しまわずに使うこともできます([[p:r-image|静止画のセル]] の「モデルをドライブにしまう」)。'),
 ('パソコンとブラウザ', 'PC 版の Google Chrome がおすすめです。動画の見本(透明な WebM)は Chrome で見られます。'),
])}

<h2>1. ノートブックを開く</h2>
<h3>A. 「Colab で開く」ボタンで開く(おすすめ)</h3>
<div class="dlBox">{OPEN}</div>
{steps(
 'このボタン(ほかのページの上にある [[ui:▶ Colab で開く]] でも同じ)を押します。Google Colab が開き、ノートブックが出ます。Google アカウントにログインしていない時は、右上の [[ui:ログイン]] からログインします。',
 'そのまま「2. GPU を選ぶ」に進みます。',
 'セルを押した時に「このノートブックは Google が作成したものではありません」と出たら、[[ui:このまま実行]] を押します。',
)}
<ul>
  <li><b>いつも最新の版</b>が開きます。新しい版が出ても、入れ直す必要はありません。</li>
  <li>作った素材は、自分の <b>マイドライブ / CastFrost素材</b> に保存されます(ノートブックを保存しなくても大丈夫です)。</li>
  <li>ノートブックそのものを手元に残したい時は、メニューの [[ui:ファイル]] → [[ui:ドライブにコピーを保存]] を押すと、<b>マイドライブ / Colab Notebooks</b> にコピーが入ります(その時は、そのコピーの版のまま使うことになります)。</li>
</ul>
{tip('ブラウザのブックマークに、ボタンの開く先(Colab のページ)を入れておくと、次から 1 回で開けます。')}

<h3>B. ファイルをダウンロードして開く</h3>
<p>Colab で開くボタンが使えない時や、ファイルとして持っておきたい時の方法です。</p>
{steps(
 f'<a href="dl/{NB_FILE}" download>ノートブック(v{VERSION})をダウンロード</a> します。',
 '<a href="https://colab.research.google.com/" target="_blank" rel="noopener">Google Colab</a> を開き、メニューの [[ui:ファイル]] → [[ui:ノートブックをアップロード]] で、ダウンロードしたファイルを選びます。',
 'アップロードしたノートブックは、自動で <b>マイドライブ / Colab Notebooks</b> に入ります。次からは、ドライブのそのファイルをダブルクリック(または [[ui:アプリで開く]] → [[ui:Google Colaboratory]])で開けます。',
)}
{tip('B の方法で入れたノートブックは、新しい版が出ても自動では変わりません。新しい版を使う時は、もう一度ダウンロードして入れ直してください(作った素材とモデルは、別のフォルダに残っています)。')}

<h2>2. GPU を選ぶ</h2>
{steps(
 'ノートブックを開いた状態で、メニューの [[ui:ランタイム]] → [[ui:ランタイムのタイプを変更]] を押します。',
 '[[ui:ハードウェア アクセラレータ]] で GPU を選びます(下の表)。初めは <b>A100</b> が選ばれています。',
 '動画を作る時は、[[ui:ハイメモリ]] もオンにしておきます(初めからオンです)。',
 '[[ui:保存]] を押します。',
)}
<table class="tbl">
<tr><th>GPU</th><th>静止画</th><th>動画</th><th>ユニット(1時間あたりの目安)</th></tr>
<tr><td><b>A100</b></td><td>◎ 1枚 約10秒</td><td>◎ 5秒の動画 1本 約2〜3分(おすすめ)</td><td>約5.4</td></tr>
<tr><td><b>L4</b></td><td>○ 1枚 約25秒</td><td>△ 1本 約6〜12分(試験的)</td><td>約1.7</td></tr>
<tr><td><b>T4</b></td><td>○ 1枚 約60秒</td><td>✕ 作れません</td><td>約1.2</td></tr>
</table>
<ul>
  <li>静止画だけを作る日は、L4 や T4 にするとユニットの減りが少なくなります。</li>
  <li>🎞 画像から動画を作る は、動画と同じく <b>A100</b> がおすすめです(L4 は試験的、T4 では作れません)。🛠 素材を加工する は、T4 でも使えます。</li>
  <li>選んだ GPU が混んでいて使えない時は、Colab が別の GPU をつなぐことがあります。セルを押すと、つながった GPU が画面の上に出ます。</li>
  <li>動画は GPU のメモリをたくさん使うので、A100 以外では「試験的」です。くわしくは [[p:size|30MB と長さ・画質]] の「GPU ごとにできること」。</li>
</ul>

<h2>3. セルを押す(ドライブにつなぐ)</h2>
{steps(
 '使いたいセルの左の [[play]](<b>再生ボタン</b>)を押します。セルは色つきの見出しで 2 つに分かれています:<b>✍ 文章(命令)から作る</b>([[cell:🖼 静止画を作る]]・[[cell:🎬 動画を作る]])と、<b>🧩 素材(画像/動画)から作る</b>([[cell:🎞 画像から動画を作る]]・[[cell:🛠 素材を加工する]])。',
 '「このノートブックは Google が作成したものではありません」と出たら、[[ui:このまま実行]] を押します。',
 '「Google ドライブへのアクセスを許可」の画面が出たら、自分のアカウントを選んで [[ui:許可]] します。作った素材は <b>マイドライブ / CastFrost素材</b> に保存されます。',
 '準備(部品のインストール → モデルのダウンロード → GPU への読み込み)が始まります。1回目は、静止画・加工で約3〜5分、動画・画像から動画で約10〜15分かかります。',
 'セルの下に、色つきの帯(セルごとに色が違います)と入力欄が出たら、準備ができています。',
)}
{warn('ドライブへのアクセスを許可しなくても使えますが、その時は作ったものをドライブに保存せず、ダウンロードだけになります。モデルもドライブにしまわないので、毎回ダウンロードになります。')}

<h2>4. 使い終わったら</h2>
<p>いちばん下の [[cell:🛑 GPU を止める]] を押すか、メニューの [[ui:ランタイム]] → [[ui:ランタイムを接続解除して削除]] を押します。つないだままだと、何もしていなくてもユニットが減り続けます。くわしくは [[p:g-stop|使い終わったら(GPU を止める)]]。</p>
'''))
