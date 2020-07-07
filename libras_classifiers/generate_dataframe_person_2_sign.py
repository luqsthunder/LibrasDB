from libras_classifiers.librasdb_loaders import DBLoader2NPY
import numpy as np


class DataframePerson2Sign:
    """
    A classe encontra quem esta sinalizando em lingua de sinal em um video com
    poses previamente extraidas.
    Isso é feito analisando quem possui mais movimento.
    """

    def __init__(self, db, method='both', video_window=None):
        """

        Parameters
        ----------

        db: DBLoader2NPY

        method: str

        video_window: list

            A lista deve conter 2 elementos indicando a distância em frames
            anterior a posição do frame atual a ser processado e a distância
            posterior do frame atual.
            Caso nulo, sera utilizado o video inteiro como janela.

        """

        if method not in ['grad', 'dist', 'both']:
            raise ValueError('method not valid {}, '
                             'correct are {}'.format(method,
                                                     ['grad', 'dist', 'both']))

        self.method = 'grad dist' if method == 'both' else method
        self.db = db
        self.video_window = video_window if video_window is not None \
            else [-1, -1]
        self.__method_map = {'grad': self.__make_derivate_around_a_frame,
                             'dist': self.__make_distance_between_frames}

    @staticmethod
    def __make_distance_between_frames(sample, beg, mid, end):
        all_frames_in_window = sample[sample['frame'] <= end]
        all_frames_in_window = \
            all_frames_in_window[all_frames_in_window['frame'] >= beg]

        persons_id_in_video = all_frames_in_window.person.unique()
        person_all_joints_dist = [0] * len(persons_id_in_video)
        for p_id in persons_id_in_video:
            person_joints = \
                all_frames_in_window[all_frames_in_window['person'] == p_id]
            for it in range(beg + 1, end):
                frame1 = person_joints[person_joints['frame'] == it]
                frame0 = person_joints[person_joints['frame'] == (it - 1)]

                frame1 = frame1.values[0, 2:]
                frame0 = frame0.values[0, 2:]
                dist = frame1 - frame0
                dist = np.array(list(filter(lambda x: x==x, dist)))
                dist = dist ** 2
                dist = np.sum(np.array([np.sum(x) for x in dist]))
                print(f'sum after pow 2: \n {dist}\n')
                dist = np.sqrt(dist)
                print(f'sqrt: \n {dist}\n')
                dist = np.sum(dist)
                print(f'sum final: \n {dist}\n')
                person_all_joints_dist[p_id] += dist

        print('all_persons_joint', person_all_joints_dist)
        return np.argmax(np.array(person_all_joints_dist))

    def __make_derivate_around_a_frame(self, sample, beg, mid, end):
        return None

    def process_single_sample_region(self, sample, beg, mid, end):
        pass

    def process_single_sample(self, sample):
        """
        Processa um unico exemplo de um video de poses.

        Parameters
        ----------
        sample: pd.DataFrame

        Returns
        -------

        """
        res = self.method.split(' ')

        beg_frame_video = int(sample.frame.iloc[0])
        end_frame_video = int(sample.frame.iloc[-1])
        print(f'beg_frame : {beg_frame_video}, end_frame: {end_frame_video}\n'\
               '{sample}')

        window_beg = self.video_window[0] if self.video_window[0] != -1 \
            else (end_frame_video - beg_frame_video) // 2

        window_end = self.video_window[1] if self.video_window[1] != -1 \
            else (end_frame_video - beg_frame_video) // 2

        window_size = window_beg + window_end
        window_size = int(window_size)

        who_talking = {}
        for it in range(beg_frame_video, end_frame_video - window_size + 1):
            mid_curr_window = it + window_beg \
                if it + window_beg <= end_frame_video else end_frame_video

            beg_curr_window = it - window_beg \
                if it - window_beg >= beg_frame_video else beg_frame_video

            end_curr_window = mid_curr_window + window_end \
                if mid_curr_window + window_end <= end_frame_video \
                else end_frame_video

            print('process single sample', beg_curr_window, mid_curr_window,
                  end_curr_window)
            for method_name in res:
                talker_id = self.__method_map[method_name](sample,
                                                           beg_curr_window,
                                                           mid_curr_window,
                                                           end_curr_window)
                who_talking.update({method_name: {'id' :talker_id,
                                                  'frames': [beg_curr_window,
                                                             mid_curr_window,
                                                             end_curr_window]}})

        return who_talking

    def process_all(self):
        pass
