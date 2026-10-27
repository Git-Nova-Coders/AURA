"""
Unit tests for AURA Multi-Object Tracking Subsystem (Milestone 5).
"""

import unittest
from vision.detector import Detection
from vision.tracker import ObjectTracker, TrackedObject, compute_iou


class TestObjectTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker(max_age=3, min_hits=1, iou_threshold=0.3)

    def test_compute_iou(self):
        """Verify IoU calculation for identical, disjoint, and partial overlap boxes."""
        boxA = [10.0, 10.0, 50.0, 50.0]  # area = 1600
        boxB = [10.0, 10.0, 50.0, 50.0]  # area = 1600
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        # Disjoint
        boxC = [100.0, 100.0, 150.0, 150.0]
        self.assertAlmostEqual(compute_iou(boxA, boxC), 0.0)

        # Partial overlap (half width overlap)
        boxD = [30.0, 10.0, 70.0, 50.0]
        iou = compute_iou(boxA, boxD)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_tracked_object_lifecycle(self):
        """Verify TrackedObject updates history, velocity, age, and temporal persistence."""
        det = Detection(class_id=0, class_name="person", confidence=0.9, bbox=[100.0, 100.0, 200.0, 200.0])
        track = TrackedObject(
            track_id=1,
            class_id=det.class_id,
            class_name=det.class_name,
            bbox=list(det.bbox),
            confidence=det.confidence,
        )

        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.age, 1)
        self.assertEqual(track.hits, 1)
        self.assertEqual(track.temporal_persistence, 1.0)

        # Frame 2: Object moves right by 10px
        det2 = Detection(class_id=0, class_name="person", confidence=0.88, bbox=[110.0, 100.0, 210.0, 200.0])
        track.update(det2)

        self.assertEqual(track.age, 2)
        self.assertEqual(track.hits, 2)
        self.assertEqual(track.time_since_update, 0)
        self.assertAlmostEqual(track.velocity[0], 6.5)
        self.assertAlmostEqual(track.velocity[1], 0.0)
        self.assertAlmostEqual(track.motion_speed, 6.5)

        # Frame 3: Object missed
        track.mark_missed()
        self.assertEqual(track.age, 3)
        self.assertEqual(track.hits, 2)
        self.assertEqual(track.time_since_update, 1)
        self.assertAlmostEqual(track.temporal_persistence, 2.0 / 3.0)

    def test_tracker_id_persistence(self):
        """Verify tracker maintains consistent track IDs across consecutive moving frames."""
        # Frame 1: 2 objects
        f1_dets = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=[50.0, 50.0, 150.0, 150.0]),
            Detection(class_id=63, class_name="laptop", confidence=0.85, bbox=[300.0, 200.0, 400.0, 300.0]),
        ]
        tracked1 = self.tracker.update(f1_dets)
        id_person = tracked1[0].track_id
        id_laptop = tracked1[1].track_id

        self.assertIsNotNone(id_person)
        self.assertIsNotNone(id_laptop)
        self.assertNotEqual(id_person, id_laptop)

        # Frame 2: Objects shift slightly
        f2_dets = [
            Detection(class_id=0, class_name="person", confidence=0.89, bbox=[55.0, 52.0, 155.0, 152.0]),
            Detection(class_id=63, class_name="laptop", confidence=0.86, bbox=[302.0, 201.0, 402.0, 301.0]),
        ]
        tracked2 = self.tracker.update(f2_dets)

        self.assertEqual(tracked2[0].track_id, id_person)
        self.assertEqual(tracked2[1].track_id, id_laptop)

    def test_tracker_track_expiration(self):
        """Verify tracks are purged when missing for more than max_age frames."""
        # Frame 1: Object present
        det = Detection(class_id=0, class_name="person", confidence=0.9, bbox=[50.0, 50.0, 150.0, 150.0])
        self.tracker.update([det])
        self.assertEqual(len(self.tracker.all_tracks), 1)

        # Frames 2, 3, 4: No detections (max_age = 3)
        self.tracker.update([])
        self.tracker.update([])
        self.tracker.update([])
        # Frame 5: One more empty frame -> track exceeded max_age and is deleted
        self.tracker.update([])

        self.assertEqual(len(self.tracker.all_tracks), 0)


if __name__ == "__main__":
    unittest.main()
