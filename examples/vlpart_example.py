#!/usr/bin/env python

import cv2
import argparse
from vision_anything.config.model_config import VLPartConfig
from vision_anything.model.model_wrapper import VLPartModel


def main(args):
    vlpart_config = VLPartConfig.from_args()
    vlpart_model = VLPartModel(vlpart_config)
    classes = [_class.strip() for _class in args.text.split(";") if _class.strip()]
    vlpart_model.set_model(vocabulary="custom", classes=classes, confidence_threshold=args.confidence)

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
        detections, segmentation, visualization = vlpart_model.predict(frame)
        writer.write(cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="vlpart.mp4")
    parser.add_argument("--text", type=str, default="cup handle; hand; bottle cap")
    parser.add_argument("--confidence", type=float, default=0.6)
    args = parser.parse_args()
    main(args)
