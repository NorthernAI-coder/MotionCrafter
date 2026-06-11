# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for MotionCrafter Kubric preprocessing.

"""Generate dense Kubric point tracking files used by gen_kubric_video.py."""

import argparse
import functools
import itertools
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def project_point(cam, point3d, num_frames):
    """Compute the image space coordinates [0, 1] for a set of points.

    Args:
      cam: The camera parameters, as returned by kubric.  'matrix_world' and
        'intrinsics' have a leading axis num_frames.
      point3d: Points in 3D world coordinates.  it has shape [num_frames,
        num_points, 3].
      num_frames: The number of frames in the video.

    Returns:
      Image coordinates in 2D.  The last coordinate is an indicator of whether
        the point is behind the camera.
    """

    homo_transform = tf.linalg.inv(cam["matrix_world"])
    homo_intrinsics = tf.zeros((num_frames, 3, 1), dtype=tf.float32)
    homo_intrinsics = tf.concat([cam["intrinsics"], homo_intrinsics], axis=2)

    point4d = tf.concat([point3d, tf.ones_like(point3d[:, :, 0:1])], axis=2)
    point4d_cam = tf.matmul(point4d, tf.transpose(homo_transform, (0, 2, 1)))
    point3d_cam = tf.identity(point4d_cam[:, :, :3])

    projected = tf.matmul(point4d_cam, tf.transpose(homo_intrinsics, (0, 2, 1)))
    image_coords = projected / projected[:, :, 2:3]
    image_coords = tf.concat([image_coords[:, :, :2], tf.sign(projected[:, :, 2:])], axis=2)
    return image_coords, point3d_cam


def unproject(coord, cam, depth):
    """Unproject points.

    Args:
      coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
        integer (y,x) because of the way meshgrid happens.
      cam: The camera parameters, as returned by kubric.  'matrix_world' and
        'intrinsics' have a leading axis num_frames.
      depth: Depth map for the scene.

    Returns:
      Image coordinates in 3D.
    """
    shp = tf.convert_to_tensor(tf.shape(depth))
    idx = coord[:, 0] * shp[1] + coord[:, 1]
    coord = tf.cast(coord[..., ::-1], tf.float32)
    shp = tf.cast(shp[1::-1], tf.float32)[tf.newaxis, ...]

    # Need to convert from pixel to raster coordinate.
    projected_pt = (coord + 0.5) / shp

    projected_pt = tf.concat(
        [
            projected_pt,
            tf.ones_like(projected_pt[:, -1:]),
        ],
        axis=-1,
    )

    camera_plane = projected_pt @ tf.linalg.inv(tf.transpose(cam["intrinsics"]))
    camera_ball = camera_plane / tf.sqrt(
        tf.reduce_sum(
            tf.square(camera_plane),
            axis=1,
            keepdims=True,
        ),
    )
    camera_ball *= tf.gather(tf.reshape(depth, [-1]), idx)[:, tf.newaxis]

    camera_ball = tf.concat(
        [
            camera_ball,
            tf.ones_like(camera_plane[:, 2:]),
        ],
        axis=1,
    )
    points_3d = camera_ball @ tf.transpose(cam["matrix_world"])
    return points_3d[:, :3] / points_3d[:, 3:]


def reproject(coords, camera, camera_pos, num_frames, bbox=None):
    """Reconstruct points in 3D and reproject them to pixels.

    Args:
      coords: Points in 3D.  It has shape [num_points, 3].  If bbox is specified,
        these are assumed to be in local box coordinates (as specified by kubric),
        and bbox will be used to put them into world coordinates; otherwise they
        are assumed to be in world coordinates.
      camera: the camera intrinsic parameters, as returned by kubric.
        'matrix_world' and 'intrinsics' have a leading axis num_frames.
      camera_pos: the camera positions.  It has shape [num_frames, 3]
      num_frames: the number of frames in the video.
      bbox: The kubric bounding box for the object.  Its first axis is num_frames.

    Returns:
      Image coordinates in 2D and their respective depths.  For the points,
      the last coordinate is an indicator of whether the point is behind the
      camera.  They are of shape [num_points, num_frames, 3] and
      [num_points, num_frames] respectively.
    """
    # First, reconstruct points in the local object coordinate system.
    if bbox is not None:
        coord_box = list(itertools.product([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]))
        coord_box = np.array([np.array(x) for x in coord_box])
        coord_box = np.concatenate([coord_box, np.ones_like(coord_box[:, 0:1])], axis=1)
        coord_box = tf.tile(coord_box[tf.newaxis, ...], [num_frames, 1, 1])
        bbox_homo = tf.concat([bbox, tf.ones_like(bbox[:, :, 0:1])], axis=2)

        local_to_world = tf.linalg.lstsq(tf.cast(coord_box, tf.float32), bbox_homo)
        world_coords = tf.matmul(
            tf.cast(tf.concat([coords, tf.ones_like(coords[:, 0:1])], axis=1), tf.float32)[tf.newaxis, :, :],
            local_to_world,
        )
        world_coords = world_coords[:, :, 0:3] / world_coords[:, :, 3:]
    else:
        world_coords = tf.tile(coords[tf.newaxis, :, :], [num_frames, 1, 1])

    # Compute depths by taking the distance between the points and the camera
    # center.
    depths = tf.sqrt(
        tf.reduce_sum(
            tf.square(world_coords - camera_pos[:, np.newaxis, :]),
            axis=2,
        ),
    )

    # Project each point back to the image using the camera.
    projections, point3d_cam = project_point(camera, world_coords, num_frames)

    return (
        tf.transpose(projections, (1, 0, 2)), # image projections
        tf.transpose(depths), # depth
        tf.transpose(world_coords, (1, 0, 2)),# 3d point coordinates in world coordinates
        tf.transpose(point3d_cam, (1, 0, 2)), # 3d point coordinates in camera coordinates
    )


def estimate_occlusion_by_depth_and_segment(
    data,
    segments,
    x,
    y,
    num_frames,
    thresh,
    seg_id,
):
    """Estimate depth at a (floating point) x,y position.

    We prefer overestimating depth at the point, so we take the max over the 4
    neightoring pixels.

    Args:
      data: depth map. First axis is num_frames.
      segments: segmentation map. First axis is num_frames.
      x: x coordinate. First axis is num_frames.
      y: y coordinate. First axis is num_frames.
      num_frames: number of frames.
      thresh: Depth threshold at which we consider the point occluded.
      seg_id: Original segment id.  Assume occlusion if there's a mismatch.

    Returns:
      Depth for each point.
    """

    # need to convert from raster to pixel coordinates
    x = x - 0.5
    y = y - 0.5

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    shp = tf.shape(data)
    assert len(data.shape) == 3
    x0 = tf.clip_by_value(x0, 0, shp[2] - 1)
    x1 = tf.clip_by_value(x1, 0, shp[2] - 1)
    y0 = tf.clip_by_value(y0, 0, shp[1] - 1)
    y1 = tf.clip_by_value(y1, 0, shp[1] - 1)

    data = tf.reshape(data, [-1])
    rng = tf.range(num_frames)[:, tf.newaxis]
    i1 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
    i2 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
    i3 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
    i4 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

    depth = tf.maximum(tf.maximum(tf.maximum(i1, i2), i3), i4)

    segments = tf.reshape(segments, [-1])
    i1 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
    i2 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
    i3 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
    i4 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

    depth_occluded = tf.less(tf.transpose(depth), thresh)
    seg_occluded = True
    for i in [i1, i2, i3, i4]:
        i = tf.cast(i, tf.int32)
        seg_occluded = tf.logical_and(seg_occluded, tf.not_equal(seg_id, i))

    return tf.logical_or(depth_occluded, tf.transpose(seg_occluded))


def get_camera_matrices(
    cam_focal_length,
    cam_positions,
    cam_quaternions,
    cam_sensor_width,
    input_size,
    num_frames=None,
):
    """Tf function that converts camera positions into projection matrices."""
    intrinsics = []
    matrix_world = []
    assert cam_quaternions.shape[0] == num_frames
    for frame_idx in range(cam_quaternions.shape[0]):
        focal_length = tf.cast(cam_focal_length, tf.float32)
        sensor_width = tf.cast(cam_sensor_width, tf.float32)
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_width * input_size[0] / input_size[1]
        p_x = 0.5
        p_y = 0.5
        intrinsics.append(
            tf.stack(
                [
                    tf.stack([f_x, 0.0, -p_x]),
                    tf.stack([0.0, -f_y, -p_y]),
                    tf.stack([0.0, 0.0, -1.0]),
                ]
            )
        )

        position = cam_positions[frame_idx]
        quat = cam_quaternions[frame_idx]
        rotation_matrix = rotation_matrix_3d.from_quaternion(tf.concat([quat[1:], quat[0:1]], axis=0))
        transformation = tf.concat(
            [rotation_matrix, position[:, tf.newaxis]],
            axis=1,
        )
        transformation = tf.concat(
            [transformation, tf.constant([0.0, 0.0, 0.0, 1.0])[tf.newaxis, :]],
            axis=0,
        )
        matrix_world.append(transformation)

    return (
        tf.cast(tf.stack(intrinsics), tf.float32),
        tf.cast(tf.stack(matrix_world), tf.float32),
    )


def quat2rot(quats):
    """Convert a list of quaternions to rotation matrices."""
    rotation_matrices = []
    for frame_idx in range(quats.shape[0]):
        quat = quats[frame_idx]
        rotation_matrix = rotation_matrix_3d.from_quaternion(tf.concat([quat[1:], quat[0:1]], axis=0))
        rotation_matrices.append(rotation_matrix)
    return tf.cast(tf.stack(rotation_matrices), tf.float32)


def rotate_surface_normals(
    world_frame_normals,
    point_3d,
    cam_pos,
    obj_rot_mats,
    frame_for_query,
):
    """Points are occluded if the surface normal points away from the camera."""
    query_obj_rot_mat = tf.gather(obj_rot_mats, frame_for_query)
    obj_frame_normals = tf.einsum(
        "boi,bi->bo",
        tf.linalg.inv(query_obj_rot_mat),
        world_frame_normals,
    )
    world_frame_normals_frames = tf.einsum(
        "foi,bi->bfo",
        obj_rot_mats,
        obj_frame_normals,
    )
    cam_to_pt = point_3d - cam_pos[tf.newaxis, :, :]
    dots = tf.reduce_sum(world_frame_normals_frames * cam_to_pt, axis=-1)
    faces_away = dots > 0

    # If the query point also faces away, it's probably a bug in the meshes, so
    # ignore the result of the test.
    faces_away_query = tf.reduce_sum(
        tf.cast(faces_away, tf.int32) * tf.one_hot(frame_for_query, tf.shape(faces_away)[1], dtype=tf.int32),
        axis=1,
        keepdims=True,
    )
    faces_away = tf.logical_and(faces_away, tf.logical_not(faces_away_query > 0))
    return faces_away


def single_object_reproject(
    bbox_3d=None,
    pt=None,
    pt_segments=None,
    camera=None,
    cam_positions=None,
    num_frames=None,
    depth_map=None,
    segments=None,
    window=None,
    input_size=None,
    quat=None,
    normals=None,
    frame_for_pt=None,
    trust_normals=None,
):
    """Reproject points for a single object.

    Args:
      bbox_3d: The object bounding box from Kubric.  If none, assume it's
        background.
      pt: The set of points in 3D, with shape [num_points, 3]
      pt_segments: The segment each point came from, with shape [num_points]
      camera: Camera intrinsic parameters
      cam_positions: Camera positions, with shape [num_frames, 3]
      num_frames: Number of frames
      depth_map: Depth map video for the camera
      segments: Segmentation map video for the camera
      window: the window inside which we're sampling points
      input_size: [height, width] of the input images.
      quat: Object quaternion [num_frames, 4]
      normals: Point normals on the query frame [num_points, 3]
      frame_for_pt: Integer frame where the query point came from [num_points]
      trust_normals: Boolean flag for whether the surface normals for each query
        are trustworthy [num_points]

    Returns:
      Position for each point, of shape [num_points, num_frames, 2], in pixel
      coordinates, and an occlusion flag for each point, of shape
      [num_points, num_frames].  These are respect to the image frame, not the
      window.

    """
    # Finally, reproject
    reproj, depth_proj, world_pos, cam_pos = reproject(
        pt,
        camera,
        cam_positions,
        num_frames,
        bbox=bbox_3d,
    )

    occluded = tf.less(reproj[:, :, 2], 0)
    reproj = reproj[:, :, 0:2] * np.array(input_size[::-1])[np.newaxis, np.newaxis, :]
    occluded = tf.logical_or(
        occluded,
        estimate_occlusion_by_depth_and_segment(
            depth_map[:, :, :, 0],
            segments[:, :, :, 0],
            tf.transpose(reproj[:, :, 0]),
            tf.transpose(reproj[:, :, 1]),
            num_frames,
            depth_proj * 0.99,
            pt_segments,
        ),
    )
    obj_occ = occluded
    obj_reproj = reproj

    obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 1], window[0]))
    obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 0], window[1]))
    obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 1], window[2]))
    obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 0], window[3]))

    if quat is not None:
        faces_away = rotate_surface_normals(
            normals,
            world_pos,
            cam_positions,
            quat2rot(quat),
            frame_for_pt,
        )
        faces_away = tf.logical_and(faces_away, trust_normals)
    else:
        # world is convex; can't face away from cam.
        faces_away = tf.zeros([tf.shape(pt)[0], num_frames], dtype=tf.bool)

    return obj_reproj, tf.logical_or(faces_away, obj_occ), depth_proj, cam_pos, world_pos


#  pylint: disable=cell-var-from-loop


def track_points(
    object_coordinates,
    depth,
    depth_range,
    segmentations,
    surface_normals,
    bboxes_3d,
    obj_quat,
    cam_positions,
    intrinsics, 
    matrix_world,
    window,
    max_seg_id=25,
    query_frame=0,
):
    """Track every pixel from one query frame across the full Kubric clip.

    Args:
      object_coordinates: Video of coordinates for each pixel in the object's
        local coordinate frame.  Shape [num_frames, height, width, 3]
      depth: uint16 depth video from Kubric.  Shape [num_frames, height, width]
      depth_range: Values needed to normalize Kubric's int16 depth values into
        metric depth.
      segmentations: Integer object id for each pixel.  Shape
        [num_frames, height, width]
      surface_normals: uint16 surface normal map. Shape
        [num_frames, height, width, 3]
      bboxes_3d: The set of all object bounding boxes from Kubric [num_objects, num_frames, num_corners, 3]
      obj_quat: Quaternion rotation for each object.  Shape
        [num_objects, num_frames, 4]
      cam_positions: Camera positions, with shape [num_frames, 3]
      intrinsics: Camera intrinsic parameters, with shape [num_frames, 3, 3]
      matrix_world: Camera to world matrices, with shape [num_frames, 4, 4]
      window: the window inside which we're sampling points.  Integer valued
        in the format [x_min, y_min, x_max, y_max], where min is inclusive and
        max is exclusive.
      max_seg_id: The maximum segment id in the video.
      query_frame: Frame whose pixels are used as query points.
    Returns:
      Query points for every pixel in the query frame, shaped [num_points, 3].
      Each point is [t, y, x] in pixel/frame coordinates.
      The trajectory for each query point, of shape [num_points, num_frames, 2].
        Each point is [x, y].  Points are in pixel coordinates
      Occlusion flag for each point, of shape [num_points, num_frames].  This is
        a boolean, where True means the point is occluded.
    """
    chosen_points = []
    all_reproj = []
    all_occ = []
    chosen_points_depth = []
    all_reproj_depth = []
    # Convert to metric depth
    depth_range_f32 = tf.cast(depth_range, tf.float32)
    depth_min = depth_range_f32[0]
    depth_max = depth_range_f32[1]
    depth_f32 = tf.cast(depth, tf.float32)
    depth_map = depth_min + depth_f32 * (depth_max - depth_min) / np.iinfo(np.uint16).max

    surface_normal_map = surface_normals / np.iinfo(np.uint16).max * 2.0 - 1.0

    input_size = object_coordinates.shape.as_list()[1:3]
    num_frames = object_coordinates.shape.as_list()[0]

    segmentations_box = segmentations
    object_coordinates_box = object_coordinates

    # If the normal map is very rough, it's often because they come from a normal
    # map rather than the mesh.  These aren't trustworthy, and the normal test
    # may fail (i.e. the normal is pointing away from the camera even though the
    # point is still visible).  So don't use the normal test when inferring
    # occlusion.
    trust_sn = True
    sn_pad = tf.pad(surface_normal_map, [(0, 0), (1, 1), (1, 1), (0, 0)])
    shp = surface_normal_map.shape
    sum_thresh = 0
    for i in [0, 2]:
        for j in [0, 2]:
            diff = sn_pad[:, i : shp[1] + i, j : shp[2] + j, :] - surface_normal_map
            diff = tf.reduce_sum(tf.square(diff), axis=-1)
            sum_thresh += tf.cast(diff > 0.05 * 0.05, tf.int32)
    trust_sn = tf.logical_and(trust_sn, (sum_thresh <= 2))[..., tf.newaxis]
    surface_normals_box = surface_normal_map
    trust_sn_box = trust_sn
    
    def get_camera(fr=None):
        if fr is None:
            return {"intrinsics": intrinsics, "matrix_world": matrix_world}
        return {"intrinsics": intrinsics[fr], "matrix_world": matrix_world[fr]}

    # Construct pixel coordinates for each pixel within the window.
    window = tf.cast(window, tf.float32)
    start_vec = [0, int(window[0]), int(window[1])]
    end_vec = [num_frames, int(window[2]), int(window[3])]
    z, y, x = tf.meshgrid(*[tf.range(st, ed) for st, ed in zip(start_vec, end_vec)], indexing="ij")
    pix_coords = tf.reshape(tf.stack([z, y, x], axis=-1), [-1, 3])

    for i in range(max_seg_id):
        obj_id = i - 1
        mask = tf.equal(tf.reshape(segmentations_box, [-1]), i)
        mask = tf.math.logical_and(mask, tf.equal(pix_coords[:, 0], query_frame))

        pt_coords = tf.boolean_mask(pix_coords, mask)
        pt = tf.boolean_mask(tf.reshape(object_coordinates_box, [-1, 3]), mask)
        normals = tf.boolean_mask(tf.reshape(surface_normals_box, [-1, 3]), mask)
        trust_sn_mask = tf.boolean_mask(tf.reshape(trust_sn_box, [-1, 1]), mask)
        trust_sn_gather = trust_sn_mask

        if obj_id == -1:
            # For the background object, no bounding box is available.  However,
            # this doesn't move, so we use the depth map to backproject these points
            # into 3D and use those positions throughout the video.
            pt_3d = []
            pt_coords_reorder = []
            for fr in range(num_frames):
                pt_coords_chunk = tf.boolean_mask(pt_coords, tf.equal(pt_coords[:, 0], fr))
                pt_coords_reorder.append(pt_coords_chunk)
                pt_3d.append(unproject(pt_coords_chunk[:, 1:], get_camera(fr), depth_map[fr]))
            pt = tf.concat(pt_3d, axis=0)
            chosen_points.append(tf.concat(pt_coords_reorder, axis=0))
            bbox = None
            quat = None
            frame_for_pt = None
        else:
            # For any other object, we just use the point coordinates supplied by kubric.
            pt = pt / np.iinfo(np.uint16).max - 0.5 
            chosen_points.append(pt_coords)
            bbox = tf.cond(obj_id >= tf.shape(bboxes_3d)[0], lambda: bboxes_3d[0, :], lambda: bboxes_3d[obj_id, :]) # bboxes_3d [num_objects, num_frames, num_corners, 3]
            quat = tf.cond(obj_id >= tf.shape(obj_quat)[0], lambda: obj_quat[0, :], lambda: obj_quat[obj_id, :]) # obj_quat [num_objects, num_frames, 4]
            frame_for_pt = pt_coords[..., 0]

        pt_depth = []
        for fr in range(num_frames):
            pt_coords_chunk = tf.boolean_mask(pt_coords, tf.equal(pt_coords[:, 0], fr))
            shp = tf.convert_to_tensor(tf.shape(depth_map[fr]))
            idx = pt_coords_chunk[:, 1] * shp[1] + pt_coords_chunk[:, 2]
            pt_depth.append(tf.gather(tf.reshape(depth_map[fr], [-1]), idx))
        chosen_points_depth.append(tf.concat(pt_depth, axis=0))
        
        # Finally, compute the reprojections for this particular object.
        obj_reproj, obj_occ, reproj_depth, _, _ = tf.cond(
            tf.shape(pt)[0] > 0,
            functools.partial(
                single_object_reproject, 
                bbox_3d=bbox,  # bboxes_3d [1, num_frames, num_corners, 3]
                pt=pt,
                pt_segments=i,
                camera=get_camera(),
                cam_positions=cam_positions,
                num_frames=num_frames,
                depth_map=depth_map,
                segments=segmentations,
                window=window,
                input_size=input_size,
                quat=quat,
                normals=normals,
                frame_for_pt=frame_for_pt,
                trust_normals=trust_sn_gather,
            ),
            lambda: (
                tf.zeros([0, num_frames, 2], dtype=tf.float32),
                tf.zeros([0, num_frames], dtype=tf.bool),
                tf.zeros([0, num_frames], dtype=tf.float32),
                tf.zeros([0, num_frames, 3], dtype=tf.float32),
                tf.zeros([0, num_frames, 3], dtype=tf.float32),
            ),
        )

        all_reproj.append(obj_reproj)
        all_occ.append(obj_occ)
        all_reproj_depth.append(reproj_depth)

    # Points are currently in pixel coordinates of the original video.  We now
    # convert them to coordinates within the window frame, and rescale to
    # pixel coordinates.  Note that this produces the pixel coordinates after
    # the window gets cropped and rescaled to the full image size.
    wd = tf.concat([np.array([0.0]), window[0:2], np.array([num_frames]), window[2:4]], axis=0)
    wd = wd[tf.newaxis, tf.newaxis, :]
    coord_multiplier = [num_frames, input_size[0], input_size[1]]
    all_reproj = tf.concat(all_reproj, axis=0) 
    
    # We need to extract x,y, but the format of the window is [t1,y1,x1,t2,y2,x2]
    window_size = wd[:, :, 5:3:-1] - wd[:, :, 2:0:-1]
    window_top_left = wd[:, :, 2:0:-1]

    all_reproj = (all_reproj - window_top_left) / window_size
    all_reproj = all_reproj * coord_multiplier[2:0:-1]
    all_occ = tf.concat(all_occ, axis=0) 
    chosen_points_depth = tf.concat(chosen_points_depth, axis=0)
    all_reproj_depth = tf.concat(all_reproj_depth, axis=0)

    chosen_points = tf.concat(chosen_points, axis=0)

    pixel_to_raster = tf.constant([0.0, 0.5, 0.5])[tf.newaxis, :]
    chosen_points = tf.cast(chosen_points, tf.float32) + pixel_to_raster

    chosen_points = (chosen_points - wd[:, 0, :3]) / (wd[:, 0, 3:] - wd[:, 0, :3])
    chosen_points = chosen_points * coord_multiplier

    all_relative_depth = all_reproj_depth / chosen_points_depth[..., tf.newaxis] # [num_points, num_frames]/[num_points,1]

    return (
        tf.cast(chosen_points, tf.float32),  # [num_points, (t,y,x)]
        tf.cast(all_reproj, tf.float32),  # [num_points, num_frames, 2]
        all_occ,  # [num_points, num_frames]
        all_relative_depth,  # [num_points, num_frames]
        all_reproj_depth,  # [num_points, num_frames]
    )


def add_tracks(
    data,
    train_size,
    max_seg_id=25,
    start_frame=0,
    end_frame=6,
):
    """Generate dense point tracks for a contiguous query-frame range."""
    shp = data["video"].shape.as_list()
    num_frames = shp[0]
    crop_window = tf.constant([0, 0, shp[1], shp[2]], dtype=tf.int32, shape=[4])

    # RGB frames are written once by the job that starts at frame 0.
    video = tf.identity(data["video"])
    start = tf.tensor_scatter_nd_update([0, 0, 0, 0], [[1], [2]], crop_window[0:2])
    size = tf.tensor_scatter_nd_update(tf.shape(video), [[1], [2]], crop_window[2:4] - crop_window[0:2])
    video = tf.slice(video, start, size)
    video = tf.image.resize(tf.cast(video, tf.float32), train_size)
    video.set_shape([num_frames, train_size[0], train_size[1], 3])

    # Camera parameters are passed through for gen_kubric_video.py.
    input_size = data["object_coordinates"].shape.as_list()[1:3]
    num_frames = data["object_coordinates"].shape.as_list()[0]
    intrinsics, camera2world_matrix = get_camera_matrices(
        data["camera"]["focal_length"],
        data["camera"]["positions"],
        data["camera"]["quaternions"],
        data["camera"]["sensor_width"],
        input_size,
        num_frames=num_frames,
    )

    query_points_all = []
    target_points_all = []
    occluded_all = []
    reproj_depth_all = []
    # Dense tracks are generated for the query frames assigned to this process.
    for i in range(start_frame, end_frame):
        query_points_i, target_points_i, occluded_i, _, reproj_depth_i = track_points(
            data["object_coordinates"],
            data["depth"],
            data["metadata"]["depth_range"],
            data["segmentations"],
            data["normal"],
            data["instances"]["bboxes_3d"],
            data["instances"]["quaternions"],
            data["camera"]["positions"],
            intrinsics, 
            camera2world_matrix,
            crop_window,
            max_seg_id,
            query_frame=i,
        )

        query_points_all.append(query_points_i)         # shape: [N, 3]
        target_points_all.append(target_points_i)       # shape: [N, num_frames, 2]
        occluded_all.append(occluded_i)                 # shape: [N, num_frames]
        reproj_depth_all.append(reproj_depth_i)         # shape: [N, num_frames]

    query_points = tf.stack(query_points_all, axis=0)       # [num_frames, N, 3]
    target_points = tf.stack(target_points_all, axis=0)     # [num_frames, N, num_frames, 2]
    occluded = tf.stack(occluded_all, axis=0)               # [num_frames, N, num_frames]
    reproj_depth = tf.stack(reproj_depth_all, axis=0)       # [num_frames, N, num_frames]
    num_pixels_per_frame = train_size[0] * train_size[1]
    query_points.set_shape([end_frame-start_frame, num_pixels_per_frame, 3])
    target_points.set_shape([end_frame-start_frame, num_pixels_per_frame, num_frames, 2])
    occluded.set_shape([end_frame-start_frame, num_pixels_per_frame, num_frames])
    reproj_depth.set_shape([end_frame-start_frame, num_pixels_per_frame, num_frames])

    res = {
        "video": video / (255.0 / 2.0) - 1.0,  # S,H,W,C range:[-1,1]
        "intrinsics": intrinsics,
        "camera2world_matrix": camera2world_matrix,
        "query_points": query_points,
        "target_points": target_points,
        "reproj_depth": reproj_depth,
        "occluded": occluded,
        "sensor_width": data["camera"]["sensor_width"],
    }
    return res


def create_point_tracking_dataset(
    raw_dir,
    train_size,
    shuffle_buffer_size=256,
    split="train",
    batch_dims=tuple(),
    repeat=True,
    max_seg_id=25,
    num_parallel_point_extraction_calls=16,
    start_frame=0,
    end_frame=6,
):
    """Construct a dataset for point tracking using Kubric.

    Args:
      train_size: Tuple of 2 ints. RGB output is resized to this resolution.
      shuffle_buffer_size: Int. Size of the shuffle buffer
      split: Which split to construct from Kubric.  Can be 'train' or
        'validation'.
      batch_dims: Sequence of ints. Add multiple examples into a batch of this
        shape.
      repeat: Bool. whether to repeat the dataset.
      max_seg_id: Int. The maximum segment id in the video.  The graph size is
        proportional to this value, so prefer small values.
      num_parallel_point_extraction_calls: Int. The num_parallel_calls for the
        map function for point extraction.
    Returns:
      The dataset generator.
    """
    ds = tfds.load(
        f"movi_f/512x512:1.0.0",
        data_dir=raw_dir,
        shuffle_files=shuffle_buffer_size is not None,
        download=False,
    )

    ds = ds[split]
    if repeat:
        ds = ds.repeat()

    ds = ds.map(
        functools.partial(
            add_tracks,
            train_size=train_size,
            max_seg_id=max_seg_id,
            start_frame=start_frame,
            end_frame=end_frame,
        ),
        num_parallel_calls=num_parallel_point_extraction_calls,
    )
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    for bs in batch_dims[::-1]:
        ds = ds.batch(bs)

    return ds



def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Generate dense Kubric tracking .npy files for gen_kubric_video.py.",
        epilog=(
            "Example: python datasets/preprocess/gen_kubric_tracking.py "
            "--raw_dir data/datasets/kubric_tracking/ "
            "--processed_dir data/datasets/kubric --split validation "
            "--start_frame 0 --end_frame 2"
        ),
    )
    parser.add_argument("--raw_dir", type=str, default="./datasets/movi_a/", help="Kubric TFDS data directory.")
    parser.add_argument("--image_size", type=int, default=512, help="Output image size.")
    parser.add_argument("--processed_dir", type=str, default="./datasets/kubric_processed_mix_3d/", help="Output directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Kubric split.")
    parser.add_argument("--start_frame", type=int, default=6, help="First query frame, inclusive.")
    parser.add_argument("--end_frame", type=int, default=12, help="Last query frame, exclusive.")
    return parser


def save_sequence_outputs(data, processed_dir, seq_num, start_frame):
    seq_dir = os.path.join(processed_dir, seq_num)
    os.makedirs(seq_dir, exist_ok=True)

    # Shared per-sequence assets only need to be written by the frame-0 job.
    if start_frame == 0:
        frame_dir = os.path.join(seq_dir, "video_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for frame_idx, frame in enumerate(data["video"]):
            rgb = (((frame + 1) / 2.0) * 255.0).astype("uint8")
            Image.fromarray(rgb).save(os.path.join(frame_dir, f"{frame_idx:03d}.png"))

        camera_info = {
            "intrinsics": data["intrinsics"].astype(np.float16),
            "sensor_width": data["sensor_width"].astype(np.float16),
            "camera2world_matrix": data["camera2world_matrix"].astype(np.float16),
        }
        np.save(os.path.join(seq_dir, f"{seq_num}_camera.npy"), camera_info)

    dense_tracking = {
        # gen_kubric_video.py expects these four keys.
        "coords": data["target_points"].astype(np.float16),
        "queries": data["query_points"].astype(np.float16),
        "reproj_depth": data["reproj_depth"].astype(np.float16),
        "visibility": data["occluded"].astype(bool),
    }
    np.save(os.path.join(seq_dir, f"{seq_num}_dense_tracking_{start_frame}.npy"), dense_tracking)


def main():
    args = get_args_parser().parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError("--end_frame must be greater than or equal to --start_frame")

    processed_dir = args.processed_dir
    if args.split == "validation" and not processed_dir.endswith("_val"):
        processed_dir = processed_dir + "_val"

    dataset = tfds.as_numpy(
        create_point_tracking_dataset(
            raw_dir=args.raw_dir,
            train_size=(args.image_size, args.image_size),
            shuffle_buffer_size=None,
            split=args.split,
            batch_dims=tuple(),
            repeat=False,
            max_seg_id=25,
            num_parallel_point_extraction_calls=4,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    )

    for seq_idx, data in enumerate(dataset):
        seq_num = f"{seq_idx:04d}"
        print(f"Processing {seq_num}, query frames {args.start_frame}-{args.end_frame}")
        save_sequence_outputs(data, processed_dir, seq_num, args.start_frame)

    print("DONE")


if __name__ == "__main__":
    main()
