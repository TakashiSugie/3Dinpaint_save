import numpy as np

# 配列の情報を見るために用意した関数です。これはcsv読み込みとは関係ありません。


def array_info(x):
    print("配列のshape", x.shape)
    print("配列の要素のデータ型", x.dtype)
    if len(x) >= 10:
        print("配列の中身（上から10列）n", x[:10], "n")
    else:
        print("配列の中身n", x, "n")


# ここから下がメイン
file1_data = np.loadtxt("1_2.csv",       # 読み込みたいファイルのパス
                        delimiter=",",    # ファイルの区切り文字
                        skiprows=1,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                        usecols=(1, 2)  # 読み込みたい列番号
                        )
np.save("./FP/file1_data", file1_data)


file2_data = np.loadtxt("1_2.csv",       # 読み込みたいファイルのパス
                        delimiter=",",    # ファイルの区切り文字
                        skiprows=1,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                        usecols=(3, 4)  # 読み込みたい列番号
                        )
np.save("file2_data", file2_data)
