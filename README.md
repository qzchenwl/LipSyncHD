# LipSyncHD

## 准备

```bash
git submodule init
git submodule update

mkdir checkpoint examples experiments/pretrained_models gfpgan/weights
cp libs/DH_live/checkpoint/* checkpoint/
cat checkpoint/render.pth.gz.00* | gzip -d > checkpoint/render.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P gfpgan/weights
wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -P gfpgan/weights

pip install -r requirements.txt --use-pep517
```

## 启动服务

```bash
PORT=8000 python -m cog.server.http
```

## 调用服务

```bash
python lipsync.py http://localhost:8000 --stage=preprocess --video @examples/video.mp4 --output examples/video_data.tar

python lipsync.py http://localhost:8000 --stage=postprocess --video-data @examples/video_data.tar --audio @examples/audio.wav --output examples/upscaled.mp4

python libsync.py http://localhost:8000 --stage=all --video @examples/video.mp4 --audio @examples/audio.wav --output examples/upscaled.mp4
```

## Prompt

```text
写一个Python脚本，作为命令行工具，每次执行将发起HTTP请求。

lipsync http://example.com --stage=preprocess --video @path/to/video.mp4 --output path/to/output.tar
REQUEST
POST http://example.com/predictions
{
  "input": {
    "video": "data url of path/to/video.mp4 if prefix with @ else as it is",
    "stage": "preprocess"
  }
}
RESPONSE
{
  "output": "data url of path/to/output.tar"
}
ACTION
save the output to path/to/output.tar

lipsync http://example.com --stage=postprocess --video-data @path/to/video_data.tar --audio @path/to/audio.wav --output path/to/output.mp4
REQUEST
POST http://example.com/predictions
{
  "input": {
    "video_data": "data url of path/to/video_data.tar if prefix with @ else as it is",
    "audio": "data url of path/to/audio.wav if prefix with @ else as it is",
    "stage": "postprocess"
  }
}
RESPONSE
{
  "output": "data url of path/to/output.mp4"
}
ACTION
save the output to path/to/output.mp4
```