"""
Numpyによる2次元畳み込み (convolve2d) 参考実装
=================================================

引用元:
    aa_debdeb (2019)「【画像処理】Numpyで空間フィルタリング(畳み込み演算)」Qiita
    https://qiita.com/aa_debdeb/items/e74eceb13ad8427b16c6

概要:
    scipy.signal.convolve2d を使わずに、NumPy の stride_tricks と einsum だけで
    2次元畳み込みを実装した例。グレースケール・カラー両対応。

実装のポイント:
    - np.lib.stride_tricks.as_strided でカーネルサイズに合わせたスライディング
      ウィンドウビューを作成することで、明示的なループを避けて高速化する。
    - np.einsum('kl,ijkl->ij', kernel, strided_image) により、各ウィンドウに
      対してカーネルとの要素積の総和を一括計算する。
    - np.pad でエッジ補填 (boundary='edge') を行い、出力サイズを入力と同じに保つ。

使用例:
    - 平均化フィルタ (ボケ処理)
    - Sobelフィルタ (エッジ検出)
    天文画像への応用として、スカイバックグラウンドの空間平滑化や
    コズミックレイ後の残差チェックなどに利用できる。
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  内部関数: パディング
# ------------------------------------------------------------------ #

def _pad_singlechannel_image(image, kernel_shape, boundary):
    """グレースケール画像 (2-D) をカーネル半幅分だけパディングする。"""
    return np.pad(
        image,
        ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),)),
        boundary,
    )


def _pad_multichannel_image(image, kernel_shape, boundary):
    """カラー画像 (3-D, shape: H×W×C) をカーネル半幅分だけパディングする。
    チャンネル次元はパディングしない。"""
    return np.pad(
        image,
        ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),), (0,)),
        boundary,
    )


# ------------------------------------------------------------------ #
#  内部関数: 畳み込み本体
# ------------------------------------------------------------------ #

def _convolve2d(image, kernel):
    """
    グレースケール画像に対するパディング済み畳み込み。

    Parameters
    ----------
    image : ndarray, shape (H, W)
        パディング済み入力画像。
    kernel : ndarray, shape (kH, kW)
        畳み込みカーネル。

    Returns
    -------
    ndarray, shape (H - kH + 1, W - kW + 1)

    Notes
    -----
    np.lib.stride_tricks.as_strided で (out_H, out_W, kH, kW) の
    ビュー配列を作り、np.einsum で一括内積計算を行う。
    メモリコピーを伴わないため大きな画像でも比較的省メモリ。
    """
    shape = (
        image.shape[0] - kernel.shape[0] + 1,
        image.shape[1] - kernel.shape[1] + 1,
    ) + kernel.shape
    strides = image.strides * 2
    strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)
    return np.einsum('kl,ijkl->ij', kernel, strided_image)


def _convolve2d_multichannel(image, kernel):
    """カラー画像 (H×W×C) に対してチャンネルごとに _convolve2d を適用する。"""
    out = np.empty((
        image.shape[0] - kernel.shape[0] + 1,
        image.shape[1] - kernel.shape[1] + 1,
        image.shape[2],
    ))
    for i in range(image.shape[2]):
        out[:, :, i] = _convolve2d(image[:, :, i], kernel)
    return out


# ------------------------------------------------------------------ #
#  公開関数
# ------------------------------------------------------------------ #

def convolve2d(image, kernel, boundary='edge'):
    """
    NumPy だけで実装した 2-D 畳み込み関数。

    Parameters
    ----------
    image : ndarray
        入力画像。shape は (H, W) または (H, W, C)。
        画素値は float 推奨 (整数の場合は事前に正規化すること)。
    kernel : ndarray, shape (kH, kW)
        畳み込みカーネル。
    boundary : str or None, default 'edge'
        np.pad に渡すパディング方式。
        'edge'  : 端の画素値で埋める (デフォルト)
        'reflect': 鏡面反射
        None    : パディングなし (出力サイズが小さくなる)

    Returns
    -------
    ndarray
        入力と同じ空間サイズ (boundary が None でない場合) の畳み込み結果。
    """
    if image.ndim == 2:
        pad_image = (
            _pad_singlechannel_image(image, kernel.shape, boundary)
            if boundary is not None
            else image
        )
        return _convolve2d(pad_image, kernel)
    elif image.ndim == 3:
        pad_image = (
            _pad_multichannel_image(image, kernel.shape, boundary)
            if boundary is not None
            else image
        )
        return _convolve2d_multichannel(pad_image, kernel)
    else:
        raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")


# ------------------------------------------------------------------ #
#  使用例
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    # --- 画像読み込み & 正規化 ---
    image_name = 'sample.png'   # 任意の画像ファイルに変更すること
    original_image = plt.imread(image_name)
    if np.issubdtype(original_image.dtype, np.integer):
        original_image = original_image / np.iinfo(original_image.dtype).max

    # --- 平均化フィルタ (3×3) ---
    def create_averaging_kernel(size):
        return np.full(size, 1.0 / (size[0] * size[1]))

    averaging3x3_image = convolve2d(original_image, create_averaging_kernel((3, 3)))
    averaging5x5_image = convolve2d(original_image, create_averaging_kernel((5, 5)))

    # --- Sobel フィルタ (水平方向エッジ検出) ---
    gray_image = (
        0.2116 * original_image[:, :, 0]
        + 0.7152 * original_image[:, :, 1]
        + 0.0722 * original_image[:, :, 2]
    )
    sobel_h_kernel = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1],
    ])
    sobel_h_image = convolve2d(gray_image, sobel_h_kernel)
    value_range = max(abs(sobel_h_image.min()), abs(sobel_h_image.max()))

    # --- 表示 ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_image);        axes[0].set_title('Original')
    axes[1].imshow(averaging3x3_image);    axes[1].set_title('Avg 3x3')
    axes[2].imshow(averaging5x5_image);    axes[2].set_title('Avg 5x5')
    axes[3].imshow(sobel_h_image, cmap='bwr', vmin=-value_range, vmax=value_range)
    axes[3].set_title('Sobel H')
    plt.colorbar(axes[3].images[0], ax=axes[3])
    plt.tight_layout()
    plt.show()
