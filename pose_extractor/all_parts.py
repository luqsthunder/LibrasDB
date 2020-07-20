HAND_PARTS = { "Wrist":                 0, "ThumbMetacarpal":         1,
               "ThumbProximal":         2, "ThumbMiddle":             3,
               "ThumbDistal":           4, "IndexFingerMetacarpal":   5,
               "IndexFingerProximal":   6, "IndexFingerMiddle":       7,
               "IndexFingerDistal":     8, "MiddleFingerMetacarpal":  9,
               "MiddleFingerProximal": 10, "MiddleFingerMiddle":     11,
               "MiddleFingerDistal":   12, "RingFingerMetacarpal":   13,
               "RingFingerProximal":   14, "RingFingerMiddle":       15,
               "RingFingerDistal":     16, "LittleFingerMetacarpal": 17,
               "LittleFingerProximal": 18, "LittleFingerMiddle":     19,
               "LittleFingerDistal":   20
             }
HAND_PARTS_NAMES = [k for k, v in HAND_PARTS.items()]

INV_HAND_PARTS = {v: k for k, v in HAND_PARTS.items()}

HAND_PAIRS = [ ["Wrist",                  "ThumbMetacarpal"],
               ["ThumbMetacarpal",        "ThumbProximal"],
               ["ThumbProximal",          "ThumbMiddle"],
               ["ThumbMiddle",            "ThumbDistal"],
               ["Wrist",                  "IndexFingerMetacarpal"],
               ["IndexFingerMetacarpal",  "IndexFingerProximal"],
               ["IndexFingerProximal",    "IndexFingerMiddle"],
               ["IndexFingerMiddle",      "IndexFingerDistal"],
               ["Wrist",                  "MiddleFingerMetacarpal"],
               ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
               ["MiddleFingerProximal",   "MiddleFingerMiddle"],
               ["MiddleFingerMiddle",     "MiddleFingerDistal"],
               ["Wrist",                  "RingFingerMetacarpal"],
               ["RingFingerMetacarpal",   "RingFingerProximal"],
               ["RingFingerProximal",     "RingFingerMiddle"],
               ["RingFingerMiddle",       "RingFingerDistal"],
               ["Wrist",                  "LittleFingerMetacarpal"],
               ["LittleFingerMetacarpal", "LittleFingerProximal"],
               ["LittleFingerProximal",   "LittleFingerMiddle"],
               ["LittleFingerMiddle",     "LittleFingerDistal"] ]

BODY_PARTS = { "Nose":    0, "Neck":       1, "RShoulder":    2, "RElbow":  3,
               "RWrist":  4, "LShoulder":  5, "LElbow":       6, "LWrist":  7,
               "RHip":    8, "RKnee":      9, "RAnkle":      10, "LHip":   11,
               "LKnee":  12, "LAnkle":    13, "REye":        14, "LEye":   15,
               "REar":   16, "LEar":      17, "Background":  18 }
BODY_PARTS_NAMES = [k for k, v in BODY_PARTS.items()]

INV_BODY_PARTS = {v: k for k, v in BODY_PARTS.items()}

"""
Lista contendo as juntas do corpo de acordo com o modelo treinado MPI pelo 
openpose.
"""
BODY_PAIRS = [ ["Neck",      "RShoulder"],
               ["Neck",      "LShoulder"],
               ["RShoulder", "RElbow"],
               ["RElbow",    "RWrist"],
               ["LShoulder", "LElbow"],
               ["LElbow",    "LWrist"],
               ["Neck",      "RHip"],
               ["RHip",      "RKnee"],
               ["RKnee",     "RAnkle"],
               ["Neck",      "LHip"],
               ["LHip",      "LKnee"],
               ["LKnee",     "LAnkle"],
               ["Neck",      "Nose"],
               ["Nose",      "REye"],
               ["REye",      "REar"],
               ["Nose",      "LEye"],
               ["LEye",      "LEar"]
             ]

BODY_ANGLE_PAIRS = [
    [0, 1, 2], [0, 1, 5], [0, 1, 11], [0, 1, 8], [1, 5, 6], [1, 2, 3], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]

BODY_ANGLE_PAIRS = [[INV_BODY_PARTS[x[0]], INV_BODY_PARTS[x[1]], INV_BODY_PARTS[x[2]]] for x in BODY_ANGLE_PAIRS]

HAND_ANGLE_PAIRS = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8], [0, 9, 10], [9, 10, 11],
                    [10, 11, 12], [0, 13, 14], [13, 14, 15], [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20]]

HAND_ANGLE_PAIRS = [[INV_HAND_PARTS[x[0]], INV_HAND_PARTS[x[1]], INV_HAND_PARTS[x[2]]] for x in HAND_ANGLE_PAIRS]