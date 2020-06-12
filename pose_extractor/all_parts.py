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