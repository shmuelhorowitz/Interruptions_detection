from retinaface.pre_trained_models import get_model
import face_recognition
from deepface import DeepFace  # deep face for emotion detection (can also detect age, gender and race)
import cv2
import dlib
import numpy as np
import pandas as pd
import os
import pickle
from utils.utils import is_url
import youtube_dl
from imageai.Detection import ObjectDetection  # yolo
from FER_model.model import FacialExpressionModel
from utils.face_func import get_iou_match, iou, inflate_locations, is_only_one_match, convert_yolo_detections, bbox_to_centers
from collections import defaultdict
from scipy.spatial.distance import cdist


class VideoAnalyser(object):
    def __init__(self, video_path, models_lib, config):
        self.faces_df = pd.DataFrame(
            columns=['frame_timestamp_msec', 'face_id', 'box', 'embedding_deepf', 'emotion_deepf', "embedding_fer",
                     "emotion_fer"])
        dict_keys = ['face_index', "frame_time", "box", 'face_locations', 'face_encodings', "embedding_fer_vec",
                     "pred_fer_vec", "embedding_deepf_vec", "pred_deepf"]
        self.faces_dict = dict(zip(dict_keys, [[] for _ in range(len(dict_keys))]))
        self.last_frame_face_indexes = []
        self.real_last_frame_face_indexes = []
        self.init_faces_dict(dict_keys)
        self.models_lib = models_lib
        self.config = config
        if is_url(video_path):
            video_url = video_path
            ydl_opts = {}
            ydl = youtube_dl.YoutubeDL(ydl_opts)
            info_dict = ydl.extract_info(video_url, download=False)
            need_to_process = True
            formats = info_dict.get('formats', None)
            for f in formats:
                # I want the better resolution, so I set resolution as 360p
                if f.get('format_note', None) == '360p' and need_to_process:
                    # get the video url
                    url = f.get('url', None)

                    # open url with opencv
                    cap = cv2.VideoCapture(url)
                    self.video = cap
                    print('cap opened: ', str(cap.isOpened()))
                    break
        else:
            #  in case video path is mp4 movie:
            self.video = cv2.VideoCapture(video_path)

        self.fer_model = FacialExpressionModel(os.path.join(self.models_lib, "fer_model.json"),
                                               os.path.join(self.models_lib, "fer_model_weights.h5"))
        if self.config.detector.yolo.enable:
            model_path = os.path.join(self.models_lib, "yolo.h5")
            detector = ObjectDetection()
            detector.setModelTypeAsYOLOv3()
            detector.setModelPath(model_path)
            detector.loadModel()
            self.yolo_detector = detector
        if self.config.detector.resnet.enable:
            model_retina_fd = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
            model_retina_fd.eval()
            self.detector_retina = model_retina_fd
        if self.config.detector.dlib.enable:
            cnn_face_detector = dlib.cnn_face_detection_model_v1(os.path.join(
                self.models_lib, "dlib_mmod_human_face_detector.dat"))
            self.dlib_cnn_detector = cnn_face_detector
            faces_cnn = cnn_face_detector(frame, 1)

    def __del__(self):
        self.video.release()

    def init_faces_dict(self, dict_keys=None):
        if dict_keys is None:
            dict_keys = self.faces_dict.keys()
        self.faces_dict = dict(zip(dict_keys, [[] for _ in range(len(dict_keys))]))
        self.last_frame_face_indexes = []
        self.real_last_frame_face_indexes = []

    def get_movie_features(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, frame_count

    def save_face_dict(self):
        if len(self.faces_dict["frame_time"]) > 0 and  len(set(self.faces_dict["frame_time"])) > self.config.minimal_frames_to_save:
            with open(os.path.join(self.config.full_out_name, f"res_analysis_dict_{int(self.faces_dict['frame_time'][0] / 1000)}"
                                                                 f"_to_{self.faces_dict['frame_time'][-1] / 1000}ses.pkl"), "wb") as handle:
                pickle.dump(self.faces_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # returns camera frames along with bounding boxes and predictions. take B
    def remove_small_detections(self,face_locations, min_size=25):
        for (top, right, bottom, left) in face_locations:
            if bottom - top < min_size or right - left < min_size:
                face_locations.remove((top, right, bottom, left))
        return face_locations

    def get_frame_B(self, frame_timestamp_msec=None, save_frame=False):
        if frame_timestamp_msec is not None:
            self.video.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp_msec)
        is_frame, frame = self.video.read()
        if not is_frame:
            return None, None, {}

        annotation = self.detector_retina.predict_jsons(frame)
        if len(annotation) < 2:
            return None, None
        retina_locations = [(i['bbox'][1], i['bbox'][2], i['bbox'][3], i['bbox'][0]) for i in annotation]
        retina_locations = self.remove_small_detections(retina_locations)
        if self.config.detector.yolo.enable:
            print("detect with yolo")
            _, yolo_detections = self.yolo_detector.detectObjectsFromImage(input_image=frame, input_type="array",
                                                                           output_type="array",
                                                                           minimum_percentage_probability=30)
            yolo_locations = convert_yolo_detections(yolo_detections)
            face_locations = self.combine_detections(retina_locations, yolo_locations)
        else:
            face_locations = retina_locations
        temp_crop_file = "/tmp/temp_crop.png"
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        current_embedding_fer, current_pred_fer, current_pred = [], [], []
        for face_ind, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
            annotation_xy = [left, top, right, bottom]
            x = max(annotation_xy[0], 0)
            y = max(annotation_xy[1], 0)
            w = annotation_xy[2] - annotation_xy[0]
            h = annotation_xy[3] - annotation_xy[1]
            crop = frame[y:y + h, x:x + w, :]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray_crop, (48, 48))
            embedding_fer, pred_fer = self.fer_model.predict_emotion_vector(roi[np.newaxis, :, :, np.newaxis])

            cv2.imwrite(temp_crop_file, crop)
            obj = DeepFace.analyze(img_path=temp_crop_file, actions=('emotion',),
                                   enforce_detection=False)
            demo = obj['emotion']
            pred = max(demo, key=demo.get)

            current_embedding_fer.append(embedding_fer)
            current_pred_fer.append(pred_fer)
            current_pred.append(pred)

        first_frame_flag = len(self.faces_dict['frame_time']) == 0
        is_first_scene, face_indexes = self.associate_faces(face_locations, face_encodings, frame.shape)  # associate betwan faces in the new frame and the previous frame. if its the first frame return the random index for each face
        if is_first_scene:
            print(f"first scene: {frame_timestamp_msec/1000} s")
            self.save_face_dict()  # if it's the first frame do nothing, else keep previous face dict and than continue to the scene
            self.init_faces_dict()  # clean the dictionary
        self.faces_dict['frame_time'] = self.faces_dict['frame_time'] + [frame_timestamp_msec] * len(face_locations)
        self.faces_dict['face_index'] = self.faces_dict['face_index']  + face_indexes

        self.faces_dict['face_encodings'] = self.faces_dict['face_encodings'] + face_encodings#list(map(lambda x: face_encodings[x], face_indexes))
        self.faces_dict['face_locations'] = self.faces_dict['face_locations'] + face_locations#list(map(lambda x: face_locations[x], face_indexes))
        self.faces_dict["embedding_fer_vec"] = self.faces_dict['embedding_fer_vec'] + current_embedding_fer#list(map(lambda x: current_embedding_fer[x], face_indexes))
        self.faces_dict["pred_fer_vec"] = self.faces_dict['pred_fer_vec'] + current_pred_fer#list(map(lambda x: current_pred_fer[x], face_indexes))
        self.faces_dict["pred_deepf"] = self.faces_dict['pred_deepf'] + current_pred #list(map(lambda x: current_pred[x], face_indexes))un_
        self.real_last_frame_face_indexes = face_indexes
        face_indexes = self.add_undetected_faces(face_indexes, frame_timestamp_msec)
        self.last_frame_face_indexes = face_indexes
        if save_frame:
            self.save_frame(frame, is_first_scene)
        return is_frame, frame

    def add_undetected_faces(self, face_indexes, frame_timestamp_msec):
        """
        add spurious of undetected faces
        :return:
        """
        undetected = set(self.last_frame_face_indexes) - set(face_indexes)
        prev_number = len(self.last_frame_face_indexes)
        for old_face in undetected:
            arg_face_arg = np.argwhere(np.array(self.last_frame_face_indexes) == old_face)[0,0]
            self.faces_dict['frame_time'] = self.faces_dict['frame_time'] + [frame_timestamp_msec]
            self.faces_dict['face_index'] = self.faces_dict['face_index'] + [old_face]
            self.faces_dict['face_encodings'] = self.faces_dict['face_encodings'] + [[]]
            self.faces_dict['face_locations'] = self.faces_dict['face_locations'] + [self.faces_dict["face_locations"][-len(face_indexes)-prev_number:][arg_face_arg]]
            self.faces_dict["embedding_fer_vec"] = self.faces_dict['embedding_fer_vec'] + [[]]
            self.faces_dict["pred_fer_vec"] = self.faces_dict['pred_fer_vec'] + [[]]
            self.faces_dict["pred_deepf"] = self.faces_dict['pred_deepf'] + [[]]
            face_indexes.append(old_face)
        return face_indexes


    def save_frame(self, frame, is_first_scene):
        color = (40, 255, 12)
        if is_first_scene:
            color = (244, 0, 12)
        locations = self.faces_dict['face_locations'][-len(self.last_frame_face_indexes):]
        encodings = self.faces_dict['face_encodings'][-len(self.last_frame_face_indexes):]
        for ((top, right, bottom, left), encoding,  face_index) in zip(locations, encodings, self.last_frame_face_indexes):
            if len(encoding) == 0:
                continue
            label = f" id: {face_index}"
            font_size = 0.4
            y_space = 3
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
            if top > h + y_space * 2:
                cv2.rectangle(frame, (left, top - h - y_space * 2), (left + w, top), color, cv2.FILLED)
                cv2.putText(frame, label, (left, top - y_space), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
            else:
                cv2.rectangle(frame, (left, 0), (left + w, h + y_space * 2), color, cv2.FILLED)
                cv2.putText(frame, label, (left, h + y_space), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(self.config.frames_out_dir, f"fr_frame time {int(self.faces_dict['frame_time'][-1]/1000):03} s.jpg"), frame)

    @staticmethod
    def combine_detections(retina_locations, yolo_locations):
        iou_mat = iou(np.array(retina_locations), np.array(yolo_locations))
        additional_yolo = set(range( len(yolo_locations))) - set(iou_mat.argmax(axis=1))
        return retina_locations + [yolo_locations[i] for i in additional_yolo]

    def associate_faces(self, face_locations, face_encodings, frame_shape):
        """
        associate faces betwean current frame and last frame
        :param frame_shape height * width (y first)
        :return: order vector of the faces in the new frame
        """
        prev_number = len(self.last_frame_face_indexes)
        real_prev_number = len(self.real_last_frame_face_indexes)

        current_number = len(face_locations)
        if len(self.faces_dict['face_index']) == 0 \
                or current_number < real_prev_number / 4 \
                or real_prev_number < current_number / 4\
                or current_number < 4 \
                or real_prev_number < 4:
            is_first_scene = True
            face_indexes = [i for i in range(len(face_locations))]
        else:
            inflated_locations = inflate_locations(np.array(face_locations), frame_shape, inflate_ratio = 1.1)
            iou_mat = iou(np.array(self.faces_dict["face_locations"][-prev_number:]), np.array(inflated_locations))
            iou_candidates = iou_mat > self.config.tracking.threshold_iou

            one_match_to_each_current, in_place_participants_rows, order_rows = is_only_one_match(iou_candidates, axis=0)
            one_match_to_each_prev, in_place_participants_col, order_columns = is_only_one_match(iou_candidates, axis=1)
            if in_place_participants_rows < prev_number/4 or in_place_participants_col < current_number/4:
                is_first_scene = True
                face_indexes = [i for i in range(len(face_locations))]
            else:
                is_first_scene = False
                face_indexes = []
                for face_it in range(current_number):
                    face_candidates =  iou_candidates[:, face_it]
                    if np.sum(iou_mat[:, face_it] > 0) == 0 :
                        face_index = len(np.unique(self.faces_dict['face_index'] + face_indexes))
                    elif np.sum(iou_mat[:, face_it] > 0) == 1:
                        face_index = self.last_frame_face_indexes[np.argmax(iou_mat[:, face_it])]
                    else:
                        optional_faces = list(np.squeeze(np.argwhere(iou_mat[:, face_it] > 0)))
                        encodings_to_compare = []
                        for i in optional_faces:
                            if len(self.faces_dict['face_encodings'][-prev_number:][i])>0:
                                encodings_to_compare.append(self.faces_dict['face_encodings'][-prev_number:][i])
                            else:
                                encodings_to_compare.append(np.zeros(128))
                        matches = face_recognition.compare_faces(encodings_to_compare, face_encodings[face_it])
                        face_index = self.last_frame_face_indexes[optional_faces[np.argmax(matches)]]
                        # print(f"number of suitable faces: {(iou_mat[:, face_it] )}")
                    face_indexes.append(face_index)


        return is_first_scene, face_indexes

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self, frame_timestamp_msec=None, save_frame=False):
        if frame_timestamp_msec is not None:
            self.video.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp_msec)
        print(f"frame time: {frame_timestamp_msec}")
        is_frame, frame = self.video.read()
        if not is_frame:
            return False, np.nan, np.nan
        temp_crop_file = "/tmp/temp_crop.png"
        box, embedding_deepf_vec, pred_deepf_vec = [], [], []
        embedding_fer_vec, pred_fer_vec, annotation = [], [], []
        # gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # kernel_size = 5
        # blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        # edges = cv2.Canny(blur_gray, 30, 200)

        use_retina_flag = True
        if use_retina_flag:
            annotation = self.detector_retina.predict_jsons(frame)
            retina_locations = [(i['bbox'][1], i['bbox'][2], i['bbox'][3], i['bbox'][0]) for i in annotation]
            face_locations = retina_locations
        else:
            face_locations = face_recognition.face_locations(frame)

        if len(face_locations) <= 2:
            return False, 0, 0

        face_encodings = face_recognition.face_encodings(frame, face_locations)
        dict_keys = self.faces_dict.keys()
        current_frame_dict = defaultdict(list)
        first_frame_flag = len(self.faces_dict['frame_time']) == 0
        if not first_frame_flag:
            last_frame_faces_ind = np.where(
                np.array(self.faces_dict['frame_time']) == self.faces_dict['frame_time'][-1])
            faces_in_last_frame = len(last_frame_faces_ind)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if first_frame_flag:
                ious = []
                new_face = True
                matches = []
            else:

                ious = np.array(get_iou_match(np.array(self.faces_dict['face_locations'])[last_frame_faces_ind],
                                              (top, right, bottom, left)))
                new_face = False
                matches = face_recognition.compare_faces(
                    np.array(self.faces_dict['face_encodings'])[last_frame_faces_ind], face_encoding)
                face_distances = np.array(
                    face_recognition.face_distance(np.array(self.faces_dict['face_encodings'])[last_frame_faces_ind],
                                                   face_encoding))

            iou = -1

            if not first_frame_flag:
                if np.any(ious > 0.9) or (np.any(ious > 0.4) and np.sort(ious)[-2] < 0.1):
                    weighted_dis = ious
                    iou = max(ious * 10)
                    face_index = self.faces_dict['face_index'][last_frame_faces_ind[0][np.argmax(ious)]]
                elif np.any(ious > 0.3):
                    last_face_to_check = len(last_frame_faces_ind)
                    face_matching = list(np.array(self.faces_dict['face_index'][-last_face_to_check:])[np.where(
                        matches[-last_face_to_check:])])  # take into considaration only matches from the last X frmae

                    if ious[np.argmin(face_distances)] > 0.25:
                        iou = ious[np.argmin(face_distances)]
                        face_index = self.faces_dict['face_index'][last_frame_faces_ind[0][np.argmin(face_distances)]]
                    else:
                        new_face = True
                elif np.min(face_distances) < 0.1:
                    iou = ious[np.argmin(face_distances)]
                    face_index = self.faces_dict['face_index'][last_frame_faces_ind[0][np.argmin(face_distances)]]

                else:
                    new_face = True

            if new_face:
                face_index = len(np.unique(self.faces_dict['face_index']))

            if face_index in current_frame_dict["face_index"]:
                # means there are two similar faces in the frame - we take the closest face
                second_face_order = current_frame_dict["face_index"].index(face_index)
                second_face_iou = 1
                # if iou > current_frame_dict["face_index"]

            # continue with top,right,bottom left:
            annotation_xy = [left, top, right, bottom]
            x = max(annotation_xy[0], 0)
            y = max(annotation_xy[1], 0)
            w = annotation_xy[2] - annotation_xy[0]
            h = annotation_xy[3] - annotation_xy[1]
            self.faces_dict["box"].append(np.array([x, y, x + w, y + h]))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop = frame[y:y + h, x:x + w, :]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray_crop, (48, 48))
            embedding_fer, pred_fer = self.fer_model.predict_emotion_vector(roi[np.newaxis, :, :, np.newaxis])

            cv2.imwrite(temp_crop_file, crop)
            obj = DeepFace.analyze(img_path=temp_crop_file, actions=('emotion',),
                                   enforce_detection=False)
            demo = obj['emotion']
            pred = max(demo, key=demo.get)
            # current_frame_dict['face_index'].append(face_index)
            # current_frame_dict['best_match'].append(best_match)
            # current_frame_dict['ious'].append(ious)
            # current_frame_dict['distances'].append(face_distances)

            self.faces_dict['frame_time'].append(frame_timestamp_msec)
            self.faces_dict['face_encodings'].append(face_encoding)
            self.faces_dict['face_locations'].append((top, right, bottom, left))
            self.faces_dict['face_index'].append(face_index)
            self.faces_dict["embedding_fer_vec"].append(embedding_fer)
            self.faces_dict["pred_fer_vec"].append(pred_fer)
            self.faces_dict["embedding_deepf_vec"].append(list(demo.values()))
            self.faces_dict["pred_deepf_vec"].append(pred)

            #     TODO: add chekc that each face appears not more than one time. If yes: take the face with the hithest
            #           iou and the second face will be assosoated to its second option

            # Draw a box aif save_frame:round the face -out of the if/else
            if save_frame:
                color = (40, 255, 12)
                label = f" id: {face_index}-{pred_fer}"
                font_size = 0.4
                y_space = 3
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
                if top > h + y_space * 2:
                    cv2.rectangle(frame, (left, top - h - y_space * 2), (left + w, top), color, cv2.FILLED)
                    cv2.putText(frame, label, (left, top - y_space), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
                else:
                    cv2.rectangle(frame, (left, 0), (left + w, h + y_space * 2), color, cv2.FILLED)
                    cv2.putText(frame, label, (left, h + y_space), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 255), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, label, (x, y), font, 0.3, (0, 0, 255), 1)
                # cv2.putText(frame, f"{face_index}_{iou:.2f}", (left + 6, bottom - 6), font, 0.2, (0, 0, 0), 1)

        # for k in self.faces_dict.keys():
        #     self.faces_dict[k].extend(current_frame_dict[k])
        # df = pd.DataFrame({'box': box, 'embedding_deepf': embedding_deepf_vec, 'emotion_deepf': pred_deepf_vec,
        #                    "embedding_fer": embedding_fer_vec, "emotion_fer": pred_fer_vec})
        df = pd.DataFrame()
        return True, frame, df


    def get_frame_participants_data(self, frame_timestamp_msec=None):
        """
        This function help to create the dataset distribution for example age and gender
        """
        if frame_timestamp_msec is not None:
            self.video.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp_msec)
        print(f"frame time: {frame_timestamp_msec}")
        is_frame, frame = self.video.read()
        if not is_frame:
            return [], []
        temp_crop_file = "/tmp/temp_crop.png"
        annotation = self.detector_retina.predict_jsons(frame)
        try:
            face_locations = [(i['bbox'][1], i['bbox'][2], i['bbox'][3], i['bbox'][0]) for i in annotation]
            age, gender = [], []
            for (top, right, bottom, left) in face_locations:
                annotation_xy = [left, top, right, bottom]
                x = max(annotation_xy[0], 0)
                y = max(annotation_xy[1], 0)
                w = annotation_xy[2] - annotation_xy[0]
                h = annotation_xy[3] - annotation_xy[1]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                crop = frame[y:y + h, x:x + w, :]
                cv2.imwrite(temp_crop_file, crop)
                obj = DeepFace.analyze(img_path=temp_crop_file, actions=('emotion','age', 'gender'), enforce_detection=False)
                age.append(obj['age'])
                gender.append(obj['gender'])
            return age, gender
        except:
            return [], []