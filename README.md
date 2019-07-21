## One class Mask-RCNN

The repo is based on the implementation of Mask-RCNN by (https://github.com/matterport/Mask_RCNN).<br>
Great thanks to the developers.

This repo is utility to train one-class mask and run inference by Mask-RCNN.

<br>

### 単一クラス用Mask-RCNN
このrepoはInstance segmentationを単一クラスに対して適用するためのユーティリティです。

元画像-二値化マスク画像をそれぞれ`dataset`の`imag`および`mask`に配置するだけで準備は完了です。<br>
※二値化マスクではインスタンスが分離されている必要があります。これはopencvのブロブ検出によってインスタンスの判断を行うためです。

1. `python -m src prepare -d ./dataset`
2. `python -m src train -d ./dataset`
3. `python -m src validation -d ./dataset`

結果はすべて`dataset/logs`に出力されます。

<br>

### 環境作成
- 要求パッケージのインストール<br>
    MASK-RCNNフォルダ内で`pip install -r requirements.txtx`

- imgaugパッケージのinstall
※この手順はimgaugのインストに失敗した場合に必要
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely

    インストール方法　：　https://imgaug.readthedocs.io/en/latest/source/installation.html

- Shapely のインストール
※この手順はshapelyのインストに失敗した場合に必要
UnicodeDecodeError => condaに切り替えでおｋ

- pycocotoolsのインストール要求
https://github.com/waleedka/coco

    Note: Edit PythonAPI/Makefile and replace "python" with "python3".


    PythonAPIディレクトリの中で
    `python setup.py build_ext install`

- Mask-RCNNのインストール

    `python setup.py develop` or `python setup.py install`

