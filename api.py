import argparse
import time

import cv2
import numpy as np
import torch

from lightweight_openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from lightweight_openpose.modules.keypoints import extract_keypoints, group_keypoints
from lightweight_openpose.modules.load_state import load_state
from lightweight_openpose.modules.pose import Pose, track_poses
from lightweight_openpose.val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def pose_detection(pose):
    #print(Pose.num_kpts)
    #print(Pose.kpt_names)
    #print(pose.keypoints)
    pose_dict = {}
    for i in range(len(pose.keypoints)):
        #print(Pose.kpt_names[i], pose.keypoints[i])
        pose_dict[Pose.kpt_names[i]] = pose.keypoints[i]

    if pose_dict['l_wri'][1] < pose_dict['l_elb'][1] \
            and pose_dict['l_elb'][1] < pose_dict['l_sho'][1]:
        print('\tDETECT: Raise Hand (Left)')
        return 'l_raise_hand'

    elif pose_dict['r_wri'][1] < pose_dict['r_elb'][1] \
            and pose_dict['r_elb'][1] < pose_dict['r_sho'][1]:
        print('\tDETECT: Raise Hand (Right)')
        return 'r_raise_hand'

    else:
        return None



def run_demo(net, image_provider, height_size, cpu, track, smooth, confidence_thres=14.):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 10000
    count = 0
    for img in image_provider:
        t = time.monotonic()

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        print(pose_entries)
        #print(all_keypoints)
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            print('confidence', pose.confidence)
            if pose.confidence > confidence_thres:
                pose.draw(img)
                ret = pose_detection(pose) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if ret is not None:
                    shape = img.shape
                    print(shape)
                    cv2.putText(img, f'{ret}', (0, shape[0]), \
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3, cv2.LINE_AA)

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            if pose.confidence > confidence_thres:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255))

        t = time.monotonic() - t

        print(f"fps: {1/t}")
        cv2.putText(img, f'fps: {1/t:.0f}', (0, 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        cv2.imwrite(f'out/{count:06d}.jpg', img)
        count += 1
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


def get_pose_estimation_net(checkpoint_path):
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    return net

def get_pose(net, img, height_size=256, cpu=True, confidence_thres=14.):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    t = time.monotonic()

    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []

    #print(pose_entries)
    #print(all_keypoints)
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    for pose in current_poses:
        #print('confidence', pose.confidence)
        if pose.confidence > confidence_thres:
            pose.draw(img)
            pose_name = pose_detection(pose) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        if pose.confidence > confidence_thres:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

    t = time.monotonic() - t
    #print(f"fps: {1/t}")

    return img, pose_name

def test():
    input_path = 'test_images/000193.jpg'
    checkpoint_path = 'checkpoint_iter_370000.pth'
    result_path = 'result.jpg'
    img = cv2.imread(input_path)
    net = get_pose_estimation_net(checkpoint_path=checkpoint_path)
    result, pose_name = get_pose(net, img, cpu=True)

    print(f"input image path: {input_path}")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"pose_name: {pose_name}")
    print(f"result: {result_path}")
    cv2.imwrite(result_path, result)


if __name__ == '__main__':
    if True:
        test()
    else:
        parser = argparse.ArgumentParser(
            description='''Lightweight human pose estimation python demo.
                           This is just for quick results preview.
                           Please, consider c++ demo for the best performance.''')
        parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
        parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
        parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
        parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
        parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
        parser.add_argument('--track', type=int, default=1, help='track pose id in video')
        parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
        args = parser.parse_args()

        if args.video == '' and args.images == '':
            raise ValueError('Either --video or --image has to be provided')

        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)

        frame_provider = ImageReader(args.images)
        if args.video != '':
            frame_provider = VideoReader(args.video)
        else:
            args.track = 0

        run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
