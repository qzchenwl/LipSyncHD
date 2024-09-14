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
            video: Path = Input(description="Origin video"),
            audio: Path = Input(description="Wav file of speech"),
            video_data: Path = Input(description="Tar file of preprocessed video data"),
            stage: str = Input(description="Stage of the pipeline", choices=["preprocess", "postprocess", "all"], default="all")
    ) -> Path:
        with tempfile.TemporaryDirectory() as working_dir:
            if stage in ["preprocess", "all"]:
                helper.preprocess_for_wav2lip(working_dir, video)
            else:
                sh(f"tar xf {video_data} -C {working_dir}")
            if stage == "preprocess":
                sh(f"cd {working_dir} && tar cf video_data.tar circle.mp4 keypoint_rotate.pkl")
                return Path(f"{working_dir}/video_data.tar")

            helper.wav2lip(working_dir, audio)
            helper.upscale_video(working_dir)
            return Path(f"{working_dir}/upscaled.mp4")
