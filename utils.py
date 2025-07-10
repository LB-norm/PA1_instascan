import cv2 
import numpy as np

def draw_polygon(image_path, annotation):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    coords = annotation[1:]  # Skip the class label
    points = np.array([(float(coords[i]) * w, float(coords[i+1]) * h) for i in range(0, len(coords), 2)], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Test", 1000, 1000)
    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Example usage
annotation_line = "0 0.74306778619701 0.06842105263157894 0.7541611592604316 0.4014045099616225 0.7592286420481434 0.6392432176245445 0.762318246979316 0.8973684210526316 0.48235648953231797 0.9124245270878909 0.3272578332992013 0.9157894736842106 0.31599343060309315 0.4315697304927021 0.3128199877124718 0.07894736842105263 0.491915792065558 0.0715609798851834"
annotation = annotation_line.split()
draw_polygon(r"C:\Python Projects\datasets\Exponat_img\test\images\0b9cf526-exponat_img_091.jpg", annotation)