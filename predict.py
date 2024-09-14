import shutil

import monkeypatch # noqa
import os
import tempfile
import helper

from cog import BasePredictor, Path, Input


def sh(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception(f"Command failed with return code {ret}: {cmd}")


class Predictor(BasePredictor):
    def predict(
            self,
            video: Path = Input(description="Original video", default=None),
            video_data: Path = Input(description="Output file of preprocess stage", default=None),
            audio: Path = Input(description="Wav file of speech", default=None),
            stage: str = Input(description="Stage", choices=["preprocess", "postprocess", "all"], default="all")
    ) -> Path:
        print(f"Running stage {stage} with video={video}, video_data={video_data}, audio={audio}")

        if stage in ["preprocess", "all"] and video is None:
            raise ValueError(f"Video file must be provided for {stage} stage")

        if stage == "postprocess" and video_data is None:
            raise ValueError("Video data file must be provided for postprocess stage")

        if stage in ["postprocess", "all"] and audio is None:
            raise ValueError("Audio file must be provided for postprocess stage")

        with tempfile.TemporaryDirectory() as working_dir:
            if stage in ["preprocess", "all"]:
                helper.preprocess_for_wav2lip(working_dir, video)
            else:
                sh(f"tar xf {video_data} -C {working_dir}")
            if stage == "preprocess":
                sh(f"cd {working_dir} && tar cf video_data.tar circle.mp4 keypoint_rotate.pkl")
                output_path = Path(tempfile.mkdtemp()) / "video_data.tar"
                shutil.copy(f"{working_dir}/video_data.tar", output_path)
                return Path(output_path)

            helper.wav2lip(working_dir, audio)
            helper.upscale_video(working_dir)
            output_path = Path(tempfile.mkdtemp()) / "upscaled.mp4"
            shutil.copy(f"{working_dir}/upscaled.mp4", output_path)
            return Path(output_path)
