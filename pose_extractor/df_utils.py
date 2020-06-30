import pandas as pd
from pose_extractor.openpose_extractor import DatumLike
from pose_extractor.all_parts import *


def update_xy_pose_df(datum: DatumLike, df: pd.DataFrame, frame: int,
                      person_id: int, person_pos_id: int, body_parts: list,
                      hand_parts: list):
    """
    Adiciona uma linha no dataframe (df) contendo a pessoa, frame e as juntas
    do corpo e da mão caso utilizadas.

    Parameters
    ----------
    datum: DatumLike
        Resultado do openpose contendo as juntas.

    df: pd.DataFrame
        DataFrame das juntas

    frame: int
        Frame do video correspondente a junta.
    person_id: int
        id da pessoa a ser coloca no dataframe. Deve ser igual ao ID da pessoa
        na legenda.

    person_pos_id: int
        posição da pessoa dentro do datum. Cada vez que o openpose processa uma
        imagem, ele não rastreia os esqueletos. Logo, o id da mesma pessoa
        entre frames processados pelo openpose pode variar.

    body_parts: list[str]
        Lista contendo o nome das juntas utilizadas para o corpo. Essas juntas
        devem ser nomeadas igual ao pose_extractor.all_parts.BODY_PARTS

    hand_parts: list[str]
        Lista contendo o nome das juntas utilizadas para as mãos. Essas juntas
        devem ser nomeadas igual ao pose_extractor.all_parts.HAND_PARTS

    Returns
    -------
    df: pd.DataFrame
        dataframe atual com a linha da pose adicionada a ele.
    """

    pose_dic = {
        'person': [person_id],
        'frame': [frame]
    }

    for part in body_parts:
        try:
            joint = {part: [datum.poseKeypoints[person_pos_id,
                                                BODY_PARTS[part]][: 2]]}
            pose_dic.update(joint)
        except IndexError:
            pose_dic.update({part: None})

    if hand_parts is not None:
        for part in hand_parts:
            try:
                joint_val = datum.handKeypoints[0][person_pos_id,
                                                   HAND_PARTS[part]][: 2]
                joint = {(part + 'r'): [joint_val]}
                pose_dic.update(joint)
            except IndexError:
                pose_dic.update({(part + 'r'): None})

            try:
                joint_val = datum.handKeypoints[1][person_pos_id,
                                                   HAND_PARTS[part]][: 2]
                joint = {(part + 'l'): [joint_val]}
                pose_dic.update(joint)
            except IndexError:
                pose_dic.update({(part + 'l'): None})

    # if head_joints is not None:
    pose_df = pd.DataFrame(data=pose_dic)
    return df.append(pose_df, ignore_index=True)
