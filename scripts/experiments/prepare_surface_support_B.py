"""Freeze local planar supports from real COLMAP training tracks, without updates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree, ConvexHull, QhullError
from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary, read_images_binary

ROOT = Path(__file__).resolve().parents[2]
SCENES = ['Curasao', 'IUI3-RedSea', 'JapaneseGradens-RedSea', 'Panama']
POLICY = dict(neighbors=16, minimum_common_train_views=3, max_reprojection_error_px=1.,
              max_normal_to_middle_eigenvalue=.05, min_middle_to_large_eigenvalue=.15,
              uncertainty_spacing_floor=.05, normal_distance_sigmas=3., footprint_sigmas=3.,
              note='Sparse local plane prior, not dense surface truth; no absent-support water labels.')


def prepare(scene, output):
    data = ROOT/'undistorted_data'/('undistorted_'+scene)
    points = read_points3D_binary(data/'sparse/0/points3D.bin')
    images = read_images_binary(data/'sparse/0/images.bin')
    region = json.loads((ROOT/'research_runs/geometry_gradient_A_20261021/regions'/scene/'regions.json').read_text())
    excluded = {v['camera'] for v in region['assets']}
    train = {Path(n).stem for n in (data/'train_list.txt').read_text().splitlines() if n.strip()} - excluded
    camera_ids = sorted(i for i,v in images.items() if Path(v.name).stem in train)
    train_ids = set(camera_ids)
    values = list(points.values())
    tracks = [set(map(int,p.image_ids)) & train_ids for p in values]
    good = np.array([len(t)>=3 and p.error<=1 for t,p in zip(tracks,values)])
    xyz = np.array([p.xyz for p in values])
    selected = np.flatnonzero(good)
    anchors = xyz[selected]
    distances, nearest = cKDTree(anchors).query(anchors,k=16)
    patches=[]; rejects=dict(nonplanar=0, common_views=0, hull=0)
    for j,neighbors in enumerate(nearest):
        ids = selected[neighbors]
        common = set.intersection(*(tracks[i] for i in ids))
        if len(common)<3:
            rejects['common_views']+=1;continue
        cloud = xyz[ids];center=cloud.mean(0)
        eigen,basis=np.linalg.eigh((cloud-center).T@(cloud-center)/len(cloud))
        if eigen[0]> .05*eigen[1] or eigen[1]<.15*eigen[2] or eigen[1]<=1e-16:
            rejects['nonplanar']+=1;continue
        uv=(cloud-center)@basis[:,1:]
        try: hull=ConvexHull(uv)
        except QhullError:
            rejects['hull']+=1;continue
        # Disk strictly inside convex hull; entire 3-sigma sphere must fit at use time.
        radius=float((-hull.equations[:,2]).min())
        if radius<=1e-10:
            rejects['hull']+=1;continue
        sigma=max(float(np.sqrt(max(eigen[0],0))), .05*distances[j,3],1e-8)
        patches.append((center,basis[:,0],radius,sigma,[i in common for i in camera_ids],int(selected[j])))
    output.mkdir(parents=True,exist_ok=False)
    count=len(patches)
    np.savez_compressed(output/'support.npz',centers=np.array([p[0] for p in patches]).reshape(-1,3),
        normals=np.array([p[1] for p in patches]).reshape(-1,3),radii=np.array([p[2] for p in patches]),
        sigma=np.array([p[3] for p in patches]),views=np.array([p[4] for p in patches],dtype=bool).reshape(count,len(camera_ids)),
        camera_names=np.array([Path(images[i].name).stem for i in camera_ids]),raw_xyz=xyz,
        anchor_indices=np.array([p[5] for p in patches],dtype=np.int64))
    report=dict(scene=scene,policy=POLICY,input_points=len(xyz),eligible_track_points=len(anchors),
        patches=count,rejections=rejects,excluded_review_cameras=sorted(excluded),
        points_sha256=hashlib.sha256((data/'sparse/0/points3D.bin').read_bytes()).hexdigest(),
        limitation='COLMAP source reconstruction is inherited; only eligible real training tracks build patches. Plane interpolation is an explicit prior; no dense footprint observation is claimed.')
    (output/'support.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for scene in SCENES:prepare(scene,a.output/scene)
