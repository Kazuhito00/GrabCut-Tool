# GrabCut-Tool
OpenCVのgrabCutを利用した2値セグメンテーション向けのアノテーションツールです。<br>
<img src="https://user-images.githubusercontent.com/37477845/114702034-86dc4000-9d5e-11eb-9f8a-203c4bea191d.gif" width="75%">

# Requirements
* OpenCV 3.4.2 or Later
* Pillow 6.1.0 or Later

# Usage
 
サンプルの実行方法は以下です。 <br>
起動後は以下の流れで操作します。<br>
1. nキー(前の画像へ)、pキー(次の画像へ)で画像を選びEnterで決定
2. 前景に指定する領域をマウス右ドラッグで指定しEnterで決定
3. マウス操作で後景・前景を指定 ※処理毎にアノテーション画像を保存します<br>マウス左ドラッグ：後景指定<br>マウス右ドラッグ：前景指定
 
```bash
python grabcut_tool.py
```
* --input<br>
インプット画像のパス<br>
デフォルト：image
* --output<br>
アノテーション画像の保存パス<br>
デフォルト：output
* --width<br>
表示画像・保存画像の横幅<br>
デフォルト：512
* --height<br>
表示画像・保存画像の縦幅<br>
デフォルト：512
* --thickness<br>
前景・後景指定時の線の太さ<br>
デフォルト：4
* --mask_alpha<br>
画像表示時の後景黒塗りつぶしの透明度<br>
デフォルト：0.7
* --iteration<br>
grabCutの反復回数<br>
デフォルト：5

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
GrabCut-Tool is under [Apache-2.0 License](LICENSE).
