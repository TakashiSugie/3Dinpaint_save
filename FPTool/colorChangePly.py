# 点群から座標を読み込み、その座標の点の色を変更する
# 新しいファイルとしてply.bakとして作成

from splitPoint import readPoint
import shutil


def colorChange(PointListFromTxt):
    plyPath = "./mesh/IMG_4464.ply"
    f = open(plyPath, "r")
    back_name = plyPath + ".bak"
    shutil.copy(plyPath, back_name)

    with open(plyPath+".bak", 'w') as writeF:
        while True:
            # line = f.readline().split('\n')[0]
            line = f.readline()
            if line.startswith('element vertex'):
                num_vertex = int(line.split(' ')[-1])
            # print(line)
            writeF.write(line)
            if line.startswith('end_header'):
                break

        contents = f.readlines()
        vertex_infos = contents[:num_vertex]
        face_infos = contents[num_vertex:]

        for v_info in vertex_infos:
            str_info = [float(v) for v in v_info.split('\n')[0].split(' ')]
            vx, vy, vz, r, g, b, hi = str_info
            featurePointFromPly = [vx, vy, vz]
            if featurePointFromPly in PointListFromTxt:
                print("colorchange!")
                print("%d %d %d %d" % (r, g, b, hi))
                v_info = v_info.replace("%d %d %d %d" % (
                    r, g, b, hi), "255 255 0 %d" % (hi))
            writeF.write(v_info)
        for face_info in face_infos:
            writeF.write(face_info)


PointListFromTxt, _ = readPoint("./txt/featurePoint.txt")
colorChange(PointListFromTxt)
