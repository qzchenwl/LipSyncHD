import os
import tempfile


def sh(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception(f"Command failed with return code {ret}: {cmd}")


def set_25fps_without_audio(video_path, output_path):
    sh(f"ffmpeg -i {video_path} -r 25 -an -y {output_path}")


def reverse_video(video_path, output_path):
    sh(f"ffmpeg -i {video_path} -vf reverse -y {output_path}")


def concat_videos(video_pathes, output_path):
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
        video_list_path = f.name
        f.write("\n".join([f"file '{os.path.abspath(path)}'" for path in video_pathes]))
        f.flush()
        sh(f"ffmpeg -safe 0 -f concat -i {video_list_path} -c:v copy -y {output_path}")


def combine_video_and_audio(video_path, audio_path, output_path):
    sh(f"ffmpeg -i {video_path} -i {audio_path} -c:v copy -pix_fmt yuv420p {output_path}")


def extract_25fps_frames(video_path, frames_dir):
    sh(f"ffmpeg -i {video_path} -vf fps=25 -y {frames_dir}/%06d.png")


def assemble_25fps_frames(frames_dir, output_path):
    sh(f"ffmpeg -framerate 25 -i {frames_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p -y {output_path}")


def merge_video_and_audio(video_source_path, audio_source_path, output_path):
    sh(f"ffmpeg -i {video_source_path} -i {audio_source_path} -c copy -map 0:v:0 -map 1:a:0 -shortest {output_path}")
