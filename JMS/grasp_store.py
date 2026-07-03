# grasp_store.py
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np
import json
import csv
import os

@dataclass
class GraspData:
    """
    Data structure：
      TCP: List[float] -> [x, y, z]
      pose: Tuple[Tuple[3], Tuple[3], Tuple[3]] -> ((Ra...), (Rb...), (Rc...))
      contact_area: float
      grasp_width: float
      score: float
      pair_num: Optional[int]
      edge_num: Optional[int]
      point_num: Optional[int]
    """
    TCP: List[float]
    pose: Tuple[Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float]]
    contact_area: float
    grasp_width: float
    score: float
    pair_num: Optional[int] = None   
    edge_num: Optional[int] = None  
    point_num: Optional[int] = None

    def __post_init__(self):
        """
        Automatically converts each vector within a pose into a tuple to ensure JSON serializability
        Supports lists, tuples, or np.ndarray
        """
        new_pose = []
        for v in self.pose:
            if isinstance(v, (np.ndarray, list, tuple)):
                if len(v) != 3:
                    raise ValueError(f"Each vector must have a length of 3, but {v} is received.")
                new_pose.append(tuple(v))
            else:
                raise TypeError(f"Each element in the pose must be a length-3 vector. Current type: {type(v)}, value: {v}")
        self.pose = tuple(new_pose)

    def to_flat(self):
        """Expand to a single line for CSV (using JSON string for pose)"""
        return [ 
            self.TCP[0], self.TCP[1], self.TCP[2],
            json.dumps(self.pose),
            self.contact_area,
            self.grasp_width,
            self.score,
            self.pair_num if self.pair_num is not None else "",
            self.edge_num if self.edge_num is not None else "",
            self.point_num if self.point_num is not None else ""
            ]

    @staticmethod
    def csv_header():
        return ["x", "y", "z", "pose_json", "contact_area", "grasp_width", "score","pair_num","edge_num","point_num"]


class GraspStore:
    def __init__(self):
        self._records: List[GraspData] = []

    def add(self, rec: GraspData):
        self._records.append(rec)

    def extend(self, recs: List[GraspData]):
        self._records.extend(recs)

    def __len__(self):
        return len(self._records)

    def all(self) -> List[GraspData]:
        return self._records
    
    def filter_by_indices(
        self,
        pair_num: Optional[int] = None,
        edge_num: Optional[int] = None,
        point_num: Optional[int] = None,
    ) -> List[GraspData]:
        res = []
        for r in self._records:
            if pair_num is not None and r.pair_num != pair_num:
                continue
            if edge_num is not None and r.edge_num != edge_num:
                continue
            if point_num is not None and r.point_num != point_num:
                continue
            res.append(r)
        return res

    def get_by_index(self, pair_num: int, edge_num: int, point_num: int) -> Optional[GraspData]:
        for r in self._records:
            if (
                r.pair_num == pair_num
                and r.edge_num == edge_num
                and r.point_num == point_num
            ):
                return r
        return None

    def group_by(self, level: str = "pair_edge") -> dict:
        """
        Grouped by index level:
          level="pair" -> { pair_num: [rec,...] }
          level="pair_edge" -> { (pair_num, edge_num): [rec,...] }
          level="full" -> { (pair_num, edge_num, point_num): [rec,...] }
        """
        groups = {}
        if level == "pair":
            for r in self._records:
                key = r.pair_num
                groups.setdefault(key, []).append(r)
        elif level == "pair_edge":
            for r in self._records:
                key = (r.pair_num, r.edge_num)
                groups.setdefault(key, []).append(r)
        elif level == "full":
            for r in self._records:
                key = (r.pair_num, r.edge_num, r.point_num)
                groups.setdefault(key, []).append(r)
        else:
            raise ValueError("Unsupported level: use 'pair', 'pair_edge', or 'full'")
        return groups

    # ---------------- NPZ ----------------
    def save_npz(self, filepath: str, compress: bool = True):
        """
        save as NPZ:
        - xyz: shape (n,3)
        - pose: shape (n,3,3) where pose[i] = [Ra, Rb, Rc]
        - contact_area, grasp_width, score : shape (n,)
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        n = len(self._records)
        xyz = np.zeros((n, 3), dtype=float)
        pose = np.zeros((n, 3, 3), dtype=float)
        contact = np.zeros((n,), dtype=float)
        grasp = np.zeros((n,), dtype=float)
        score = np.zeros((n,), dtype=float)
        pair_idx = np.full((n,), -1, dtype=int)
        edge_idx = np.full((n,), -1, dtype=int)
        point_idx = np.full((n,), -1, dtype=int)

        for t, r in enumerate(self._records):
            xyz[t] = np.array(r.TCP, dtype=float)
            pose[t, 0, :] = r.pose[0]
            pose[t, 1, :] = r.pose[1]
            pose[t, 2, :] = r.pose[2]
            contact[t] = r.contact_area
            grasp[t] = r.grasp_width
            score[t] = r.score
            pair_idx[t] = -1 if r.pair_num is None else int(r.pair_num)
            edge_idx[t] = -1 if r.edge_num is None else int(r.edge_num)
            point_idx[t] = -1 if r.point_num is None else int(r.point_num)

        if compress:
            np.savez_compressed(
                filepath,
                xyz=xyz,
                pose=pose,
                contact=contact,
                grasp=grasp,
                score=score,
                pair_num=pair_idx,
                edge_num=edge_idx,
                point_num=point_idx,
            )
        else:
            np.savez(
                filepath,
                xyz=xyz,
                pose=pose,
                contact=contact,
                grasp=grasp,
                score=score,
                pair_num=pair_idx,
                edge_num=edge_idx,
                point_num=point_idx,
            )

    @classmethod
    def load_npz(cls, filepath: str) -> "GraspStore":
        data = np.load(filepath, allow_pickle=False)
        store = cls()
        xyz = data["xyz"]
        pose = data["pose"]
        contact = data["contact"]
        grasp = data["grasp"]
        score = data["score"]
        pair_idx = data["pair_num"]
        edge_idx = data["edge_num"]
        point_idx = data["point_num"]

        n = xyz.shape[0]
        for t in range(n):
            tcp = list(xyz[t])
            Ra = tuple(pose[t, 0, :].tolist())
            Rb = tuple(pose[t, 1, :].tolist())
            Rc = tuple(pose[t, 2, :].tolist())
            rec = GraspData(
                TCP=tcp,
                pose=(Ra, Rb, Rc),
                contact_area=float(contact[t]),
                grasp_width=float(grasp[t]),
                score=float(score[t]),
                pair_num=int(pair_idx[t]) if pair_idx[t] >= 0 else None,
                edge_num=int(edge_idx[t]) if edge_idx[t] >= 0 else None,
                point_num=int(point_idx[t]) if point_idx[t] >= 0 else None,
            )
            store.add(rec)
        return store

    # ---------------- CSV ----------------
    def save_csv(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(GraspData.csv_header())
            for r in self._records:
                w.writerow(r.to_flat())

    @classmethod
    def load_csv(cls, filepath: str) -> "GraspStore":
        store = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tcp = [float(row["x"]), float(row["y"]), float(row["z"])]
                pose = json.loads(row["pose_json"])
                Ra = tuple(float(v) for v in pose[0])
                Rb = tuple(float(v) for v in pose[1])
                Rc = tuple(float(v) for v in pose[2])
                pair_val = int(row["pair_num"]) if row.get("pair_num", "") != "" else None
                edge_val = int(row["edge_num"]) if row.get("edge_num", "") != "" else None
                point_val = int(row["point_num"]) if row.get("point_num", "") != "" else None
                rec = GraspData(
                    TCP=tcp,
                    pose=(Ra, Rb, Rc),
                    contact_area=float(row["contact_area"]),
                    grasp_width=float(row["grasp_width"]),
                    score=float(row["score"]),
                    pair_num=pair_val,
                    edge_num=edge_val,
                    point_num=point_val,
                )
                store.add(rec)
        return store

    # ---------------- Convenience ----------------
    def save_both(self, basepath: str, compress_npz: bool = True):

        csv_path = basepath + ".csv"
        npz_path = basepath + ".npz"
        self.save_csv(csv_path)
        self.save_npz(npz_path, compress=compress_npz)

    @staticmethod
    def load(filepath: str) -> "GraspStore":
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npz":
            return GraspStore.load_npz(filepath)
        elif ext == ".csv":
            return GraspStore.load_csv(filepath)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def top_k(self, k: int = 1) -> List[GraspData]:
        if k <= 0:
            return []
        return sorted(self._records, key=lambda r: r.score, reverse=True)[:k]

    def best(self) -> Optional[GraspData]:
        if not self._records:
            return None
        return max(self._records, key=lambda r: r.score)
    
    def worst(self) -> Optional[GraspData]:
        if not self._records:
            return None
        return min(self._records, key=lambda r: r.score)


    @staticmethod
    def grasp_to_transform(grasp: GraspData, origin_offset: Optional[List[float]] = None) -> np.ndarray:
        """
        Returns a 4x4 matrix: The right-handed coordinate system is defined by the three pose vectors Ra, Rb, Rc (as row vectors).
        origin_offset: If TCP requires an offset in the local coordinate system (e.g., offset from gripper tip to tool0), pass [dx, dy, dz].
        Note: Adjust the position of Ra/Rb/Rc (rows vs. columns) according to your robot's coordinate system convention.
        """
        Ra, Rb, Rc = grasp.pose
        R = np.column_stack((np.array(Ra), np.array(Rb), np.array(Rc)))  # shape (3,3)
        t = np.array(grasp.TCP, dtype=float).reshape(3)
        if origin_offset is not None:
            t = t + np.array(origin_offset, dtype=float)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
