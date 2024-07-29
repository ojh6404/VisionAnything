#!/usr/bin/env python

import cv2
import argparse
from vision_anything.config.model_config import GroundingDINOConfig, SAMConfig, DEVAConfig
from vision_anything.model.model_wrapper import GroundingDINOModel, SAMModel, DEVAModel


def main(args):
    gd_config = GroundingDINOConfig.from_args()
    sam_config = SAMConfig.from_args()
    deva_config = DEVAConfig.from_args()

    gd_model = GroundingDINOModel(gd_config)
    sam_model = SAMModel(sam_config)
    deva_model = DEVAModel(deva_config)

    classes = [_class.strip() for _class in args.text.split(";") if _class.strip()]
    gd_model.set_model(classes)
    sam_model.set_model()
    deva_model.set_model()

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
        detections, visualization, segmentation = deva_model.predict(frame, sam_model, gd_model)
        writer.write(cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="deva.mp4")
    parser.add_argument("--text", type=str, default="cup; bottle; cloth")
    args = parser.parse_args()
    main(args)
