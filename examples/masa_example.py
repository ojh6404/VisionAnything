#!/usr/bin/env python

import cv2
import argparse
from vision_anything.config.model_config import GroundingDINOConfig, MASAConfig
from vision_anything.model.model_wrapper import GroundingDINOModel, MASAModel


def main(args):
    gd_config = GroundingDINOConfig.from_args()
    masa_config = MASAConfig.from_args()

    gd_model = GroundingDINOModel(gd_config)
    masa_model = MASAModel(masa_config)

    classes = [_class.strip() for _class in args.text.split(";") if _class.strip()]
    gd_model.set_model(classes)
    masa_model.set_model(classes=classes)

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
        detections, visualization = masa_model.predict(frame, gd_model)
        writer.write(cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="masa.mp4")
    parser.add_argument("--text", type=str, default="cup; bottle; cloth")
    args = parser.parse_args()
    main(args)
