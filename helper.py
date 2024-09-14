import glob
import math
import os
import pickle
import shutil

import numpy as np
import torch
import tqdm

import ffmpeg
import cv2

import mediapipe as mp

from basicsr.utils import imwrite
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
from gfpgan import GFPGANer

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


def detect_face_mesh(frame):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts_3d = np.zeros([478, 3])
        if not results.multi_face_landmarks:
            print("****** WARNING! No face detected! ******")
        else:
            image_height, image_width = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                for index_, i in enumerate(face_landmarks.landmark):
                    x_px = min(math.floor(i.x * image_width), image_width - 1)
                    y_px = min(math.floor(i.y * image_height), image_height - 1)
                    z_px = min(math.floor(i.z * image_width), image_width - 1)
                    pts_3d[index_] = np.array([x_px, y_px, z_px])
        return pts_3d


def calc_face_interact(face0, face1):
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / ((x_max - x_min) * (y_max - y_min))
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / ((x_max - x_min) * (y_max - y_min))
    return min(tmp0, tmp1)


def detect_face(frame):
    # 剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80的
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections or len(results.detections) > 1:
            return -1, None
        rect = results.detections[0].location_data.relative_bounding_box
        out_rect = [rect.xmin, rect.xmin + rect.width, rect.ymin, rect.ymin + rect.height]
        nose_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.NOSE_TIP)
        l_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.LEFT_EYE)
        r_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        if nose_.x > l_eye_.x or nose_.x < r_eye_.x:
            return -2, out_rect

        h, w = frame.shape[:2]
        if rect.xmin < 0 or rect.ymin < 0 or rect.xmin + rect.width > w or rect.ymin + rect.height > h:
            return -3, out_rect
        if rect.width * w < 100 or rect.height * h < 100:
            return -4, out_rect
    return 1, out_rect


def extract_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    total_frames = int(total_frames)
    pts_3d = np.zeros([total_frames, 478, 3])
    face_rect_list = []

    for frame_index in tqdm.tqdm(range(total_frames)):
        ret, frame = cap.read()  # 按帧读取视频
        # #到视频结尾时终止
        if ret is False:
            break
        tag_, rect = detect_face(frame)
        if frame_index == 0 and tag_ != 1:
            print(
                "第一帧人脸检测异常，请剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80")
            pts_3d = -1
            break
        elif tag_ == -1:  # 有时候人脸检测会失败，就用上一帧的结果替代这一帧的结果
            rect = face_rect_list[-1]
        elif tag_ != 1:
            print(
                "第{}帧人脸检测异常，请剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80, tag: {}".format(
                    frame_index, tag_))
            # exit()
        if len(face_rect_list) > 0:
            face_area_inter = calc_face_interact(face_rect_list[-1], rect)
            # print(frame_index, face_area_inter)
            if face_area_inter < 0.6:
                print("人脸区域变化幅度太大，请复查，超出值为{}, frame_num: {}".format(face_area_inter, frame_index))
                pts_3d = -2
                break

        face_rect_list.append(rect)

        x_min = rect[0] * vid_width
        y_min = rect[2] * vid_height
        x_max = rect[1] * vid_width
        y_max = rect[3] * vid_height
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        crop_size = int(max(seq_w * 1.35, seq_h * 1.35))
        x_min = int(max(0, x_mid - crop_size * 0.5))
        y_min = int(max(0, y_mid - crop_size * 0.45))
        x_max = int(min(vid_width, x_min + crop_size))
        y_max = int(min(vid_height, y_min + crop_size))

        frame_face = frame[y_min:y_max, x_min:x_max]
        print(y_min, y_max, x_min, x_max)
        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])
    cap.release()  # 释放视频对象
    return pts_3d


def preprocess_for_wav2lip(work_dir, video_path):
    front_video_path = os.path.join(work_dir, "front.mp4")
    back_video_path = os.path.join(work_dir, "back.mp4")
    circle_video_path = os.path.join(work_dir, "circle.mp4")

    ffmpeg.set_25fps_without_audio(video_path, front_video_path)
    ffmpeg.reverse_video(front_video_path, back_video_path)
    ffmpeg.concat_videos([front_video_path, back_video_path], circle_video_path)

    cap = cv2.VideoCapture(front_video_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    print("正向视频帧数：", frames)
    pts_3d = extract_from_video(front_video_path)
    if type(pts_3d) is np.ndarray and len(pts_3d) == frames:
        print("关键点已提取")
    pts_3d = np.concatenate([pts_3d, pts_3d[::-1]], axis=0)
    output_pkl_path = f"{work_dir}/keypoint_rotate.pkl"
    with open(output_pkl_path, "wb") as f:
        pickle.dump(pts_3d, f)


def wav2lip(work_dir, wav_path):
    output_video_path = os.path.join(work_dir, "wav2lip.mp4")

    audio_model = AudioModel()
    audio_model.loadModel("checkpoint/audio.pkl")
    render_model = RenderModel()
    render_model.loadModel("checkpoint/render.pth")

    pkl_path = os.path.join(work_dir, "keypoint_rotate.pkl")
    video_path = os.path.join(work_dir, "circle.mp4")

    render_model.reset_charactor(video_path, pkl_path)

    mouth_frame = audio_model.interface_wav(wav_path)

    cap_input = cv2.VideoCapture(video_path)
    vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_input.release()

    silence_video_path = os.path.join(work_dir, "silence.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(silence_video_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))

    for frame in tqdm.tqdm(mouth_frame):
        frame = render_model.interface(frame)
        video_writer.write(frame)
    video_writer.release()

    ffmpeg.combine_video_and_audio(silence_video_path, wav_path, output_video_path)


def upscale_video(
        work_dir,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler="realesrgan",
        aligned=False,
        only_center_face=False,
        weight=0.5,
        suffix=None,
        ext="auto",
        bg_tile=400
):
    wav2lip_video_path = os.path.join(work_dir, "wav2lip.mp4")
    wav2lip_frames_dir = os.path.join(work_dir, "wav2lip_frames")
    if os.path.exists(wav2lip_frames_dir):
        shutil.rmtree(wav2lip_frames_dir)
    os.makedirs(wav2lip_frames_dir)
    ffmpeg.extract_25fps_frames(wav2lip_video_path, wav2lip_frames_dir)

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == "realesrgan":
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn("The unoptimized RealESRGAN is slow on CPU. We do not use it. "
                          "If you really want to use it, please modify the corresponding codes.")
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None
    if not torch.cuda.is_available():
        bg_upsampler = None
    restorer = GFPGANer(
        model_path="experiments/pretrained_models/GFPGANv1.3.pth",
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    img_list = sorted(glob.glob(os.path.join(wav2lip_frames_dir, '*')))
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f"Processing {img_name} ...")
        basename, img_ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(work_dir, "cropped_faces", f"{basename}_{idx:02d}.png")
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if suffix is not None:
                save_face_name = f"{basename}_{idx:02d}_{suffix}.png"
            else:
                save_face_name = f"{basename}_{idx:02d}.png"
            save_restore_path = os.path.join(work_dir, "restored_faces", save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(work_dir, "cmp", f"{basename}_{idx:02d}.png"))

        # save restored img
        if restored_img is not None:
            if ext == "auto":
                extension = img_ext[1:]
            else:
                extension = ext

            if suffix is not None:
                save_restore_path = os.path.join(work_dir, 'restored_imgs', f'{basename}_{suffix}.{extension}')
            else:
                save_restore_path = os.path.join(work_dir, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    upscaled_frames_dir = os.path.join(work_dir, "restored_imgs")

    upscaled_silent_video_path = os.path.join(work_dir, "upscaled_silent.mp4")
    upscaled_video_path = os.path.join(work_dir, "upscaled.mp4")
    ffmpeg.assemble_25fps_frames(upscaled_frames_dir, upscaled_silent_video_path)
    ffmpeg.merge_video_and_audio(upscaled_silent_video_path, wav2lip_video_path, upscaled_video_path)
