# 点群の特徴点を示したファイルから特徴点を読む
import numpy as np


def readFP3D(txtPath):
    f = open(txtPath, "r")
    line = f.readline()
    #indexList = []
    FPList = []
    FPDict = {}
    while line:
        dotLines = line.split("・")
        for dotLine in dotLines:
            AllLine = dotLine.strip().split(" ")
            if len(AllLine) == 4:
                FPList.append(AllLine)
        line = f.readline()
    f.close
    for FP in FPList:
        FPDict[str(FP[0])] = [float(FP[1]), float(FP[2]), float(FP[3])]
    #FPDict = sorted(FPDict.items())
    # print(FPDict)
    return FPDict


def read3Dnp(npPath):
    FPnp = np.load(npPath)
    print(FPnp.shape)
    FPDict = {}

    for idx in range(FPnp.shape[0]):
        #FPDict[str(FP[0])] = [float(FP[1]), float(FP[2]), float(FP[3])]
        # [FPDict[FPnp[idx][1]], FPDict[FPnp[idx][1]],FPnp[idx][1], FPDict[FPnp[idx][1]],
        FPDict[FPnp[idx][0]] = [float(FPnp[idx][1]), float(
            FPnp[idx][2]), float(FPnp[idx][3])]

    return FPDict


def readFromMesh(mesh_fi):
    ply_fi = open(mesh_fi, 'r')
    Height = None
    Width = None
    hFov = None
    vFov = None
    while True:
        line = ply_fi.readline().split('\n')[0]
        if line.startswith('element vertex'):
            num_vertex = int(line.split(' ')[-1])
        elif line.startswith('element face'):
            num_face = int(line.split(' ')[-1])
        elif line.startswith('comment'):
            if line.split(' ')[1] == 'H':
                Height = int(line.split(' ')[-1].split('\n')[0])
            if line.split(' ')[1] == 'W':
                Width = int(line.split(' ')[-1].split('\n')[0])
            if line.split(' ')[1] == 'hFov':
                hFov = float(line.split(' ')[-1].split('\n')[0])
            if line.split(' ')[1] == 'vFov':
                vFov = float(line.split(' ')[-1].split('\n')[0])

        elif line.startswith('end_header'):
            break
    contents = ply_fi.readlines()
    vertex_infos = contents[:num_vertex]
    face_infos = contents[num_vertex:]
    verts = []
    colors = []
    faces = []
    for v_info in vertex_infos:
        str_info = [float(v) for v in v_info.split('\n')[0].split(' ')]
        if len(str_info) == 6:
            vx, vy, vz, r, g, b = str_info
        else:
            vx, vy, vz, r, g, b, hi = str_info
        verts.append([np.abs(vx), np.abs(vy), np.abs(vz)])
        colors.append([r, g, b, hi])
    verts = np.array(verts)
    print((np.sum(verts, axis=0))/float(verts.shape[0]))
    try:
        colors = np.array(colors)
        colors[..., :3] = colors[..., :3]/255.
    except:
        import pdb
        pdb.set_trace()

    for f_info in face_infos:
        _, v1, v2, v3 = [int(f) for f in f_info.split('\n')[0].split(' ')]
        faces.append([v1, v2, v3])
    faces = np.array(faces)

    # return verts, colors, faces, Height, Width, hFov, vFov


# FPDict = read3Dnp("../FP_3D/IMG_4482.npy")
# print(FPDict)
# FPDict = readFP3D("../FP_3D/FP_IMG_4482.txt")
# print(FPDict)
# readFromMesh("../mesh/IMG_4473.ply")
