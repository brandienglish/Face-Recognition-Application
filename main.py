import cv2
import face_recognition

#loads name and faces
known_face_encodings = []
known_face_names=[]

#known faces ######
known_person1_image = face_recognition.load_image_file("/Users/brandienglish/Desktop/Face Recognition/person1.jpg")
known_person2_image = face_recognition.load_image_file("/Users/brandienglish/Desktop/Face Recognition/person2.jpg")
known_person3_image = face_recognition.load_image_file("/Users/brandienglish/Desktop/Face Recognition/person3.jpg")

known_person1_face_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_face_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_face_encoding = face_recognition.face_encodings(known_person3_image)[0]

known_face_encodings.append(known_person1_face_encoding)
known_face_encodings.append(known_person2_face_encoding)
known_face_encodings.append(known_person3_face_encoding)

known_face_names.append("Issa Rae")
known_face_names.append("Jacob Elordi")
known_face_names.append("Jennette Mccurdy")
######known faces

# create webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    #capture frames
    ret, frame = video_capture.read()

    #return an error for no webcam
    if not ret:
        print("Error: Could not read frame.")
        break
    
    #gathers all face locations in the frame 
    face_locations =face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    #focuses on each face in frame 
    for (top, right, bottom,left), face_encoding in zip(face_locations, face_encodings):

        #checks to see if face is familar, if not it is unknown 
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "UNKNOWN"

        if True in matches:
            first_match_index = matches.index(True)
            name= known_face_names[first_match_index]

        #creates a box around face
        cv2.rectangle(frame, (left, top),(right, bottom),(0,0,255),2)
        cv2.putText(frame, name, (left,top-10),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255),2)

    #display results
    cv2.imshow("Video", frame)
   
    #press "q" to quit the program 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

