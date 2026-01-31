from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import numpy as np
import os

class ViolenceDetector:
    def __init__(self, model_path="yolo26n-seg.pt", output_dir="intersection_outputs"):
        self.output_dir = output_dir
        self.frame_count = 0
        self.fps = 30
        
        # --- ROBUST MODEL LOADING ---
        print(f"🔄 Loading model: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Successfully loaded {model_path}")
        except Exception as e:
            print(f"\n⚠️  WARNING: Could not load custom model '{model_path}'.")
            print("   👉 Falling back to standard 'yolov8n-seg.pt' for demonstration.")
            self.model = YOLO("yolov8n-seg.pt")
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created output directory: {output_dir}")
        
    def masks_intersect(self, mask1, mask2):
        """Check if two segmentation masks intersect"""
        intersection = cv2.bitwise_and(mask1, mask2)
        return np.any(intersection > 0)
    
    def calculate_mask_iou(self, mask1, mask2):
        """Calculate IoU between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def get_mask_centroid(self, mask):
        """Calculate centroid of a segmentation mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] == 0:
            return (0, 0)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_bbox_from_mask(self, mask):
        """Extract bounding box from mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [int(cmin), int(rmin), int(cmax), int(rmax)]
    
    def assess_lighting(self, frame):
        """Analyze lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:
            return "low_shadows"
        elif mean_brightness > 180:
            return "bright"
        else:
            return "normal"
    
    def assess_camera_angle(self, frame, detections):
        """Estimate camera angle based on detection positions"""
        if len(detections) == 0:
            return "unknown"
        
        heights = [det['bbox'][3] - det['bbox'][1] for det in detections]
        avg_height = np.mean(heights) if heights else 0
        
        frame_height = frame.shape[0]
        if avg_height > frame_height * 0.6:
            return "close_up"
        elif avg_height > frame_height * 0.3:
            return "medium_shot"
        else:
            return "wide_shot"
    
    def draw_masks(self, frame, masks_list, intersecting_pairs):
        """Draw segmentation masks on frame (no boxes)"""
        overlay = frame.copy()
        
        # Define colors for different people
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw each mask with different color
        for i, mask in enumerate(masks_list):
            color = colors[i % len(colors)]
            
            # Create colored mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            
            # Blend with original frame
            cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0, overlay)
            
            # Draw mask contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # Add person ID label at centroid
            centroid = self.get_mask_centroid(mask)
            cv2.putText(overlay, f"P{i+1}", (centroid[0]-15, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Highlight intersecting pairs
        for pair in intersecting_pairs:
            i, j = pair
            # Draw intersection area in bright red
            intersection = cv2.bitwise_and(masks_list[i], masks_list[j])
            overlay[intersection > 0] = (0, 0, 255)  # Bright red for intersection
        
        return overlay
    
    def save_intersection_event(self, json_data, annotated_frame):
        """Save JSON and frame to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Save JSON
        json_filename = os.path.join(self.output_dir, f"intersection_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save frame image
        img_filename = os.path.join(self.output_dir, f"intersection_{timestamp}.jpg")
        cv2.imwrite(img_filename, annotated_frame)
        
        return json_filename, img_filename
    
    def analyze_frame(self, frame):
        """Main analysis function using segmentation"""
        self.frame_count += 1
        
        # Run YOLO segmentation
        results = self.model(frame, classes=[0], verbose=False)
        
        # Check if masks exist
        if results[0].masks is None or len(results[0].masks) == 0:
            return None, frame.copy(), False
        
        boxes = results[0].boxes
        masks_data = results[0].masks.data.cpu().numpy()
        
        # Build detection list with masks
        detections = []
        masks_list = []
        
        for i in range(len(boxes)):
            box = boxes[i]
            conf = float(box.conf[0])
            
            # Get the mask for this detection
            mask = masks_data[i]
            
            # Resize mask to frame size
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Get bounding box from mask
            bbox = self.get_bbox_from_mask(mask_binary)
            
            detections.append({
                "object_id": 100 + i,
                "class": "person",
                "role": "unknown",
                "confidence": round(conf, 2),
                "bbox": bbox,
                "mask_area": int(np.sum(mask_binary))
            })
            
            masks_list.append(mask_binary)
        
        # Check for mask intersections between ALL pairs
        intersection_found = False
        decision = "NORMAL"
        risk_category = "none"
        max_iou = 0.0
        min_distance = float('inf')
        intersecting_pairs = []
        intersection_area = 0
        
        if len(detections) >= 2:
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    mask1 = masks_list[i]
                    mask2 = masks_list[j]
                    
                    # Check if masks intersect
                    if self.masks_intersect(mask1, mask2):
                        intersection_found = True
                        intersecting_pairs.append((i, j))
                        
                        # Calculate IoU between masks
                        iou = self.calculate_mask_iou(mask1, mask2)
                        max_iou = max(max_iou, iou)
                        
                        # Calculate intersection area
                        intersection = cv2.bitwise_and(mask1, mask2)
                        intersection_area = int(np.sum(intersection))
                        
                        # Calculate distance between centroids
                        centroid1 = self.get_mask_centroid(mask1)
                        centroid2 = self.get_mask_centroid(mask2)
                        distance = self.euclidean_distance(centroid1, centroid2)
                        min_distance = min(min_distance, distance)
                        
                        # Assign roles based on position
                        if centroid1[0] < centroid2[0]:
                            detections[i]['role'] = "aggressor_potential"
                            detections[j]['role'] = "victim_potential"
                        else:
                            detections[i]['role'] = "victim_potential"
                            detections[j]['role'] = "aggressor_potential"
        
        # Only flag if intersection detected
        if intersection_found:
            decision = "FLAGGED"
            risk_category = "physical_contact"
        
        # Draw custom visualization (masks only, no boxes)
        annotated_frame = self.draw_masks(frame, masks_list, intersecting_pairs)
        
        # Build output JSON with RICH METADATA
        output = {
            "frame_id": f"frame_{self.frame_count}",
            "timestamp": f"{self.frame_count / self.fps:.1f}s",
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "decision": decision,
            "risk_category": risk_category,
            "metadata": {
                "lighting_condition": self.assess_lighting(frame),
                "camera_angle": self.assess_camera_angle(frame, detections),
                "scene_density": "crowded" if len(detections) > 5 else "moderate" if len(detections) > 2 else "sparse"
            },
            "detections": detections,
            "geometry": {
                "mask_overlap_iou": round(max_iou, 2),
                "distance_between_centroids": f"{int(min_distance)}px" if min_distance != float('inf') else "N/A",
                "contact_detected": intersection_found,
                "intersecting_pairs": len(intersecting_pairs),
                "intersection_area_pixels": intersection_area
            }
        }
        
        return output, annotated_frame, intersection_found


# =============================================================================
# MAIN LOGIC (WITH CAMERA SWITCHING)
# =============================================================================

def open_camera(index):
    """Simple wrapper to safely open a camera index"""
    print(f"\n🎥 Attempting to open Camera Index: {index}...")
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"✅ Camera {index} opened successfully!")
        return cap
    else:
        print(f"❌ Failed to open Camera {index}.")
        return None

def main():
    detector = ViolenceDetector("yolo26n-seg.pt", output_dir="intersection_outputs")
    
    # --- CAMERA INIT LOGIC ---
    # Start at Index 1 (External Webcam usually)
    current_cam_index = 1
    cap = open_camera(current_cam_index)
    
    # Fallback to Index 0 if Index 1 fails
    if cap is None:
        print("⚠️ Index 1 failed. Trying Index 0...")
        current_cam_index = 0
        cap = open_camera(current_cam_index)
    
    paused = False
    current_frame = None
    intersection_count = 0
    
    print("\n" + "="*60)
    print("AUTO-SAVE SEGMENTATION MASK INTERSECTION DETECTOR")
    print("="*60)
    print(f"📁 Output: {detector.output_dir}/")
    print("\nCONTROLS:")
    print("   's'  - SWITCH CAMERA (Cycle 0 -> 1 -> 2...)")
    print("   'p'  - Pause/Resume")
    print("   'q'  - Quit")
    print("="*60 + "\n")
    
    while True:
        if not paused:
            if cap is None or not cap.isOpened():
                print("❌ Camera disconnected.")
                break

            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame read failed. Auto-switching...")
                cap.release()
                current_cam_index = (current_cam_index + 1) % 4
                cap = open_camera(current_cam_index)
                continue
            
            # Analyze
            current_json, annotated_frame, intersection_detected = detector.analyze_frame(frame)
            if annotated_frame is not None:
                current_frame = annotated_frame.copy()
            
            # Auto-Save Logic
            if intersection_detected and current_json:
                json_file, img_file = detector.save_intersection_event(current_json, annotated_frame)
                intersection_count += 1
                paused = True
                print(f"🚨 INTERSECTION DETECTED (Total: {intersection_count})")
                print(f"   Saved to: {json_file}")
        else:
            if current_frame is not None:
                annotated_frame = current_frame

        if current_frame is None: continue

        # UI
        if paused:
            # Semi-transparent overlay
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 80), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame, "PAUSED - Press 'p' to Resume", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        status = f"Cam: {current_cam_index} | Events: {intersection_count}"
        cv2.putText(annotated_frame, status, (10, annotated_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Detector", annotated_frame)
        
        # KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):  # Manual Switch
            print("\n🔄 Switching Camera...")
            if cap: cap.release()
            current_cam_index = (current_cam_index + 1) % 4
            cap = open_camera(current_cam_index)

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()