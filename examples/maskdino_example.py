#!/usr/bin/env python

import cv2
import argparse
from vision_anything.config.model_config import MaskDINOConfig
from vision_anything.model.model_wrapper import MaskDINOModel


def main(args):
    maskdino_config = MaskDINOConfig.from_args()
    maskdino_model = MaskDINOModel(maskdino_config)
    maskdino_model.set_model()

    # write mp4
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (640, 480),
    )

    # read mp4
    cap = cv2.VideoCapture(args.input)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections, segmentation, visualization = maskdino_model.predict(frame)
        writer.write(cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="maskdino.mp4")
    args = parser.parse_args()
    main(args)
