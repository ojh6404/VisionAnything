#!/usr/bin/env python

import cv2
import argparse
from vision_anything.config.model_config import GroundingDINOConfig, SAMConfig, CutieConfig
from vision_anything.model.model_wrapper import GroundingDINOModel, SAMModel, CutieModel


def main(args):
    gd_config = GroundingDINOConfig.from_args()
    sam_config = SAMConfig.from_args()
    cutie_config = CutieConfig.from_args()

    gd_model = GroundingDINOModel(gd_config)
    sam_model = SAMModel(sam_config)
    cutie_model = CutieModel(cutie_config)

    classes = [_class.strip() for _class in args.text.split(";") if _class.strip()]
    gd_model.set_model(classes)
    sam_model.set_model()

    # write mp4
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (640, 480),
    )

    # read mp4
    cap = cv2.VideoCapture(args.input)

    track_flag = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not track_flag:
            detections, _ = gd_model.predict(frame)
            segmentation, visualization = sam_model.predict(image=frame, boxes=detections.xyxy)
            cutie_model.set_model(frame, segmentation, labels=detections.class_id, classes=classes)
            track_flag = True
        else:
            segmentation, visualization = cutie_model.predict(frame)
        writer.write(cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="cutie.mp4")
    parser.add_argument("--text", type=str, default="cup; bottle; cloth")
    args = parser.parse_args()
    main(args)
