import argparse
import base64
import json

import requests


def encode_file(file_path):
    mime_type = 'text/plain'
    with open(file_path, 'rb') as file:
        encoded_content = base64.b64encode(file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_content}"


def save_file(data, output_path):
    data = base64.b64decode(data.split(',')[1])
    with open(output_path, 'wb') as file:
        file.write(data)


def main():
    parser = argparse.ArgumentParser(description='LipSync CLI Tool')
    parser.add_argument('url', help='URL for the HTTP request')
    parser.add_argument('--stage', choices=['preprocess', 'postprocess', 'all'], required=True, help='Stage of the process')
    parser.add_argument('--video', help='Path to the video file (for preprocess stage)')
    parser.add_argument('--video-data', help='Path to the video data file (for postprocess stage)')
    parser.add_argument('--audio', help='Path to the audio file (for postprocess stage)')
    parser.add_argument('--output', required=True, help='Path for the output file')

    args = parser.parse_args()

    headers = {'Content-Type': 'application/json'}

    if args.stage == 'preprocess':
        video = encode_file(args.video[1:]) if args.video.startswith('@') else args.video
        data = {
            "input": {
                "video": video,
                "stage": "preprocess"
            }
        }
    elif args.stage == 'postprocess':  # postprocess
        video_data = encode_file(args.video_data[1:]) if args.video_data.startswith('@') else args.video_data
        audio = encode_file(args.audio[1:]) if args.audio.startswith('@') else args.audio
        data = {
            "input": {
                "video_data": video_data,
                "audio": audio,
                "stage": "postprocess"
            }
        }
    else:  # all
        video = encode_file(args.video[1:]) if args.video.startswith('@') else args.video
        audio = encode_file(args.audio[1:]) if args.audio.startswith('@') else args.audio
        data = {
            "input": {
                "video": video,
                "audio": audio,
                "stage": "all"
            }
        }

    response = requests.post(f"{args.url}/predictions", headers=headers, data=json.dumps(data))
    response.raise_for_status()

    output = response.json()['output']
    save_file(output, args.output)
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()