### Mask-RCNN

The project is based on the implementation of Mask-RCNN by (https://github.com/matterport/Mask_RCNN).
Great thanks to the contributors.



### 環境作成

- imgaugパッケージのinstall
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely

    インストール方法　：　https://imgaug.readthedocs.io/en/latest/source/installation.html

- Shapely のインストール
UnicodeDecodeError => condaに切り替えでおｋ

- pycocotoolsのインストール要求
https://github.com/waleedka/coco

    Note: Edit PythonAPI/Makefile and replace "python" with "python3".


    PythonAPIディレクトリの中で
    `python setup.py build_ext install`

- Mask-RCNNのインストール

    `python setup.py develop` or `python setup.py install`


### 画像からのデータセット作成

(1) RGBimage - 白黒mask からのデータセット作成

imageとmaskは同じサイズであり、白黒マスクはインスタンスが分離されているような画像を用意する
