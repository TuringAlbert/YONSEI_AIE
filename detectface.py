import cv2
import dlib
import numpy as np

def detect_face(image_path):
  detector = dlib.get_frontal_face_detector()

  img = cv2.imread(image_path)
  faces = detector(img, 1)
  for face in faces:
    left_top = (face.left(), face.top())
    right_bottom = (face.right(), face.bottom())
    cv2.rectangle(img, left_top, right_bottom, (0,0,255),2) #bgr
  
  cv2.imshow("img",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def face_detect(image):
  left_top = (0,0)
  right_bottom = (0,0)

  detector = dlib.get_frontal_face_detector()
  face_rect_list = detector(image, 1)

  if len(face_rect_list) > 0:
    face_rect = face_rect_list[0]
    left_top = (face_rect.left(), face_rect.top())
    right_bottom = (face_rect.right(), face_rect.bottom())

  return left_top, right_bottom

def face_sample(img):
  image = cv2.imread(img)
  lt, rb = face_detect(image)
  cv2.rectangle(image, lt, rb, (0,0,255), thickness=1)
  cv2.imshow("img",image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
def face_detect_multi(image):
  detector = dlib.get_frontal_face_detector()
  face_rect_list = detector(image, 1)

  results = []

  for face_rect in face_rect_list:
    left_top = (face_rect.left(), face_rect.top())
    right_bottom = (face_rect.right(), face_rect.bottom())
    results.append((left_top, right_bottom))

  return results

def face_sample1(img):
  image = cv2.imread(img)
  for lt, rb in face_detect_multi(image):
    print(lt, rb)
    cv2.rectangle(image, lt, rb, (0,0,255), thickness=1)
    # cv2.circle(image, ((lt + rb)/2), ((rb-lt)/2), (0,0,255), thickness=1)
    #cv2.circle(img, center, radius, color, thickness)
  cv2.imshow("img",image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def draw_landmark_point(image_path): #point 추가
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/shape_predictor_68_face_landmarks.dat')
  img = image_path
#   img = cv2.imread(image_path)
  faces = detector(img, 1)
  for face in faces:
    shape = predictor(img, face)
    for x in range(68):
      pts = (shape.part(x).x, shape.part(x).y)
      cv2.circle(img, pts, 1, (255,0,0), cv2.FILLED, cv2.LINE_AA)
    cv2.imwrite('new_image.jpg', image) #image 저장
  cv2.imshow("img",img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def draw_landmark(image_path):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/shape_predictor_68_face_landmarks.dat')
  image = image_path
  #image = cv2.imread(image_path)

  for face_rect in detector(image, 0):
    shape = predictor(image, face_rect)
    for x in range(68):
      pts = (shape.part(x).x, shape.part(x).y)
      cv2.circle(image, pts, 1, (255,0,0),cv2.FILLED, cv2.LINE_AA)
      cv2.putText(image, f"{x}", pts, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite('new_image.jpg', image) #image 저장
  
  cv2.imshow("img",image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  print("hello!")

def detect_faces(img):
  result = []
  for face_rect in detector(img, 0):
    shape = predictor(img, face_rect)
    result.append(shape)
  return result

def blend_glasses(image, glasses, faces):
  for index, face in enumerate(faces):
    glasses_width = int((face.part(45).x - face.part(36).x) * 1.4)
    left = int(face.part(36).x - glasses_width * 0.15)
    glass_height = glasses_width / 240 * 112
    top = int((face.part(36).y + face.part(45).y) / 2 - glass_height / 2.5)

    glasses = cv2.resize(glasses, (glasses_width, int(glasses_width / 240 * 112)))
    alpha = cv2.cvtColor(glasses[:,:,3], cv2.COLOR_GRAY2BGR) / 255.0
    image[top:top+glasses.shape[0], left:left+glasses.shape[1],:3] = \
    (1.0 - alpha) * image[top:top+glasses.shape[0], left:left+glasses.shape[1], :3] + alpha * glasses[:,:,:3]
    cv2.imwrite(f'glasses_image_{index}.jpg', image) #image 저장
  cv2.imshow("img",image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
def rotate(img, angle, scale = 1.0):
  center  = (img.shape[1] // 2, img.shape[0] // 2)
  M = cv2.getRotationMatrix2D(center, angle, scale)

  rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
  return rotated_img

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/shape_predictor_68_face_landmarks.dat")
img_list = ['/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/image/사진3.jpeg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/image/sung.jpeg',"/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/image/you.jpeg","/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/image/nobel.jpeg"]
# img_list = ['/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_54.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_56.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_47.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_45.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_30.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_74.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_72.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/just girls/just girls_70.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/ugly love/ugly love_69.jpg', '/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/ugly love/ugly love_60.jpg']
for i in img_list:
    image = cv2.imread(i)
    faces = detect_faces(image)
    print(faces)
    draw_landmark_point(image)
    glasses = cv2.imread("/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/image/glasses.png", cv2.IMREAD_UNCHANGED)
    blend_glasses(image, glasses, faces)
