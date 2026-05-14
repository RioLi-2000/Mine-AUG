import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from mmcv.transforms import BaseTransform
except Exception:
    from mmcv.transforms.base import BaseTransform

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MineClusterCropAug(BaseTransform):
   
    def __init__(self,
                 prob=0.25,
                 min_total_obj_num=15,
                 dbscan_eps_ratio=0.12,
                 dbscan_min_samples=3,
                 min_cluster_size=4,
                 core_k_range=(4, 8),
                 crop_ratio_range=(0.35, 0.50),
                 keep_area_ratio=0.50,
                 max_try=20,
                 debug=False):
        self.prob = prob
        self.min_total_obj_num = min_total_obj_num

        self.dbscan_eps_ratio = dbscan_eps_ratio
        self.dbscan_min_samples = dbscan_min_samples
        self.min_cluster_size = min_cluster_size

        self.core_k_range = core_k_range
        self.crop_ratio_range = crop_ratio_range
        self.keep_area_ratio = keep_area_ratio

        self.max_try = max_try
        self.debug = debug

    def transform(self, results):
        img = results.get('img', None)
        gt_bboxes = results.get('gt_bboxes', None)
        gt_labels = results.get('gt_bboxes_labels', None)

        if img is None or gt_bboxes is None or gt_labels is None:
            return results

        boxes = self._bboxes_to_numpy(gt_bboxes)
        labels = self._labels_to_numpy(gt_labels)

        if boxes is None or len(boxes) == 0:
            return results

        if len(boxes) <= self.min_total_obj_num:
            return results

        if np.random.rand() > self.prob:
            return results

        out = self._apply(img, boxes, labels)
        if out is None:
            return results

        cropped_img, new_boxes, new_labels, keep_inds, debug_info = out

        results['img'] = np.ascontiguousarray(cropped_img)
        results['img_shape'] = cropped_img.shape[:2]
        results['gt_bboxes'] = self._numpy_to_bboxes(new_boxes, gt_bboxes)
        results['gt_bboxes_labels'] = self._numpy_to_labels(new_labels, gt_labels)

        if 'gt_ignore_flags' in results and results['gt_ignore_flags'] is not None:
            results['gt_ignore_flags'] = self._slice_like(results['gt_ignore_flags'], keep_inds)

        results['mine_aug_type'] = 'cluster_crop'
        if self.debug:
            results['cluster_crop_debug'] = debug_info

        return results

    def _apply(self, img, boxes, labels):
        h, w = img.shape[:2]
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

        
        eps = self.dbscan_eps_ratio * min(h, w)
        cluster_labels = self._dbscan(centers, eps=eps, min_samples=self.dbscan_min_samples)

        unique_labels = sorted(set(cluster_labels.tolist()))
        valid_clusters = []
        for lb in unique_labels:
            if lb == -1:
                continue
            inds = np.where(cluster_labels == lb)[0]
            if len(inds) >= self.min_cluster_size:
                valid_clusters.append(inds)

        if len(valid_clusters) == 0:
            return None

        
        chosen_cluster = valid_clusters[np.random.randint(0, len(valid_clusters))]
        chosen_cluster = np.array(chosen_cluster, dtype=np.int64)

        
        seed_idx = int(np.random.choice(chosen_cluster))
        seed_center = centers[seed_idx]

        
        d_cluster = np.sqrt(((centers[chosen_cluster] - seed_center) ** 2).sum(axis=1))

        k_min, k_max = self.core_k_range
        k_max_eff = min(k_max, len(chosen_cluster))
        k_min_eff = min(k_min, k_max_eff)

        if k_max_eff <= 0:
            return None

        if k_min_eff == k_max_eff:
            k = k_max_eff
        else:
            k = np.random.randint(k_min_eff, k_max_eff + 1)

        order = np.argsort(d_cluster)
        core_indices = chosen_cluster[order[:k]]
        core_indices = np.array(sorted(set(core_indices.tolist())), dtype=np.int64)

        if len(core_indices) == 0:
            return None

        core_box = self._union_box(boxes[core_indices])

        valid_crops = []

        
        for _ in range(self.max_try):
            ratio = np.random.uniform(*self.crop_ratio_range)
            crop_w = int(round(ratio * w))
            crop_h = int(round(ratio * h))

            crop_w = int(np.clip(crop_w, 2, w))
            crop_h = int(np.clip(crop_h, 2, h))

            ux1, uy1, ux2, uy2 = core_box
            union_w = ux2 - ux1
            union_h = uy2 - uy1

            
            if union_w > crop_w or union_h > crop_h:
                continue

            
            cx_low = max(crop_w / 2.0, ux2 - crop_w / 2.0)
            cx_high = min(w - crop_w / 2.0, ux1 + crop_w / 2.0)
            cy_low = max(crop_h / 2.0, uy2 - crop_h / 2.0)
            cy_high = min(h - crop_h / 2.0, uy1 + crop_h / 2.0)

            if cx_low > cx_high or cy_low > cy_high:
                continue

            crop_cx = np.random.uniform(cx_low, cx_high)
            crop_cy = np.random.uniform(cy_low, cy_high)

            x1 = int(round(crop_cx - crop_w / 2.0))
            y1 = int(round(crop_cy - crop_h / 2.0))
            x1 = int(np.clip(x1, 0, w - crop_w))
            y1 = int(np.clip(y1, 0, h - crop_h))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            patch = np.array([x1, y1, x2, y2], dtype=np.float32)

            keep_mask, core_keep_mask, clipped_boxes = self._filter_boxes_for_crop(
                boxes, patch, core_indices
            )

            keep_inds = np.where(keep_mask)[0]
            kept_core_inds = np.where(core_keep_mask)[0]

            
            if seed_idx not in keep_inds:
                continue
            if len(kept_core_inds) < 2:
                continue

            valid_crops.append({
                'patch': patch.copy(),
                'keep_inds': keep_inds.copy(),
                'clipped_boxes': clipped_boxes.copy(),
            })

        if len(valid_crops) == 0:
            return None

        chosen_crop = valid_crops[np.random.randint(0, len(valid_crops))]
        patch = chosen_crop['patch']
        keep_inds = chosen_crop['keep_inds']
        clipped_boxes = chosen_crop['clipped_boxes']

        x1, y1, x2, y2 = [int(round(v)) for v in patch]
        cropped_img = img[y1:y2, x1:x2].copy()

        new_boxes = clipped_boxes[keep_inds].copy()
        new_boxes[:, [0, 2]] -= x1
        new_boxes[:, [1, 3]] -= y1
        new_labels = labels[keep_inds].copy()

        debug_info = {
            'cluster_labels': cluster_labels.copy(),
            'chosen_cluster': chosen_cluster.copy(),
            'seed_idx': int(seed_idx),
            'core_indices': core_indices.copy(),
            'crop_box': patch.copy(),
            'keep_indices': keep_inds.copy(),
            'num_valid_crops': len(valid_crops),
            'eps': float(eps),
        }
        return cropped_img, new_boxes, new_labels, keep_inds, debug_info

    def _dbscan(self, centers, eps, min_samples):
        """A lightweight DBSCAN implementation without sklearn.
        Returns:
            labels: np.ndarray of shape [N], cluster id or -1 for noise
        """
        n = len(centers)
        labels = np.full(n, -99, dtype=np.int32)  # -99 means unvisited
        visited = np.zeros(n, dtype=bool)

        dist_mat = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))

        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = np.where(dist_mat[i] <= eps)[0]

            # noise
            if len(neighbors) < min_samples:
                labels[i] = -1
                continue

            # start new cluster
            labels[i] = cluster_id
            seeds = list(neighbors.tolist())

            j = 0
            while j < len(seeds):
                idx = seeds[j]

                if not visited[idx]:
                    visited[idx] = True
                    idx_neighbors = np.where(dist_mat[idx] <= eps)[0]
                    if len(idx_neighbors) >= min_samples:
                        for nb in idx_neighbors.tolist():
                            if nb not in seeds:
                                seeds.append(nb)

                if labels[idx] in (-99, -1):
                    labels[idx] = cluster_id

                j += 1

            cluster_id += 1

        labels[labels == -99] = -1
        return labels

    def _filter_boxes_for_crop(self, boxes, patch, core_indices):
        x1, y1, x2, y2 = patch

        inter_x1 = np.maximum(boxes[:, 0], x1)
        inter_y1 = np.maximum(boxes[:, 1], y1)
        inter_x2 = np.minimum(boxes[:, 2], x2)
        inter_y2 = np.minimum(boxes[:, 3], y2)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        orig_area = np.maximum(1e-6, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        kept_ratio = inter_area / orig_area

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        center_inside = (
            (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
            (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
        )

        keep_mask = center_inside & (kept_ratio >= self.keep_area_ratio)

        core_flag = np.zeros(len(boxes), dtype=bool)
        core_flag[core_indices] = True
        core_keep_mask = keep_mask & core_flag

        clipped_boxes = np.stack(
            [inter_x1, inter_y1, inter_x2, inter_y2], axis=1
        ).astype(np.float32)

        return keep_mask, core_keep_mask, clipped_boxes

    @staticmethod
    def _union_box(boxes):
        return np.array([
            boxes[:, 0].min(),
            boxes[:, 1].min(),
            boxes[:, 2].max(),
            boxes[:, 3].max(),
        ], dtype=np.float32)

    @staticmethod
    def _bboxes_to_numpy(gt_bboxes):
        if gt_bboxes is None:
            return None
        if hasattr(gt_bboxes, 'tensor'):
            return gt_bboxes.tensor.detach().cpu().numpy().astype(np.float32)
        return np.asarray(gt_bboxes, dtype=np.float32)

    @staticmethod
    def _labels_to_numpy(gt_labels):
        if gt_labels is None:
            return None
        if torch is not None and torch.is_tensor(gt_labels):
            return gt_labels.detach().cpu().numpy()
        return np.asarray(gt_labels)

    @staticmethod
    def _numpy_to_bboxes(boxes_np, ref_bboxes):
        if hasattr(ref_bboxes, 'tensor') and torch is not None:
            tensor = torch.as_tensor(
                boxes_np,
                dtype=ref_bboxes.tensor.dtype,
                device=ref_bboxes.tensor.device
            )
            return ref_bboxes.__class__(tensor)
        return boxes_np.astype(np.float32)

    @staticmethod
    def _numpy_to_labels(labels_np, ref_labels):
        if torch is not None and torch.is_tensor(ref_labels):
            return torch.as_tensor(labels_np, dtype=ref_labels.dtype, device=ref_labels.device)
        return labels_np.astype(ref_labels.dtype if hasattr(ref_labels, 'dtype') else np.int64)

    @staticmethod
    def _slice_like(arr, inds):
        if torch is not None and torch.is_tensor(arr):
            inds_t = torch.as_tensor(inds, dtype=torch.long, device=arr.device)
            return arr[inds_t]
        return np.asarray(arr)[inds]

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'prob={self.prob}, '
                f'min_total_obj_num={self.min_total_obj_num}, '
                f'dbscan_eps_ratio={self.dbscan_eps_ratio}, '
                f'crop_ratio_range={self.crop_ratio_range}, '
                f'keep_area_ratio={self.keep_area_ratio})')