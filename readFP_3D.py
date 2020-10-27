# 点群の特徴点を示したファイルから特徴点を読む

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
            if line.split(' ')[1] == 'featurePoint・':
                #featureP = list(line.split(' ')[-1].split('\n')[0])
                featureP = line
            print(line)
        elif line.startswith('end_header'):
            break


#FPDict = readFP3D("./FP_3D/FP_IMG_4465.txt")
