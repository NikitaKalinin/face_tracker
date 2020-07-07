import numpy as np
import cv2
import dlib
import face_recognition

# Set basics
video_capture = cv2.VideoCapture(0)
window = 'Video'
font = cv2.FONT_HERSHEY_SIMPLEX
tolerance = 0.6
model = 'hog'

# Set variables
known_face_encodings = []
known_face_names = []

# Scan face(s)
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # found face locations
    face_locations = face_recognition.face_locations(rgb_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Draw box around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 0, 255), 2)  # red color for any found face

    cv2.imshow(window, frame)

    key_code = cv2.waitKey(1)

    # Hit 'q' on the keyboard to exit
    if key_code & 0xFF == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        exit(0)

    # Hit 's' on the keyboard to scan face
    if key_code & 0xFF == ord('s'):

        # print count of detected faces 
        print('found {} face(s)'.format(len(face_locations)))
        if len(face_locations) == 0:
            video_capture.release()
            cv2.destroyAllWindows()
            exit(0)

        # remember face encodings with unique name
        print('enter face name or press return to skip')
        face_encoding_index = 0
        for (top, right, bottom, left) in face_locations:
            face_encoding_index += 1

            # draw selected face
            selected_face_frame = np.copy(frame)
            cv2.rectangle(
                selected_face_frame,
                (left, top), (right, bottom),
                (0, 165, 255), 2)  # orange color for selected face
            cv2.imshow(window, selected_face_frame)
            cv2.waitKey(5)

            # give face a name
            print(f'who is face # {face_encoding_index} : ', end='')
            face_name = str(input())

            if face_name is '':
                print('unknown')
                continue

            # remember face name and encoding
            known_face_names.append(face_name)
            known_face_encodings.append(face_encodings[face_encoding_index - 1])

            # draw remembered face box
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 180, 0), 2)  # green color

            # draw a label
            cv2.putText(frame, face_name, (left + 6, bottom - 6),
                        font, 0.5, (255, 255, 255), 1)
            cv2.imshow(window, frame)

        break

if len(known_face_encodings) == 0:
    print('no faces')
    exit(0)
else:
    print('\nfaces in memory:')

for face_encoding_index in range(len(known_face_encodings)):
    print(f'face # {face_encoding_index + 1}, name \"{known_face_names[face_encoding_index]}\", encoding {type(known_face_encodings[face_encoding_index])}')
print()

process_this_frame = True
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        is_found_known_face = False
        print(f'found {len(face_encodings)} faces')

        # draw any found face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 0, 255), 2)  # red color for any found face

        face_encoding_index = -1
        for face_encoding in face_encodings:
            face_encoding_index += 1

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,
                                                     tolerance=tolerance)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:

                is_found_known_face = True

                name = known_face_names[best_match_index]
                print(f'    found \"{name}\", distance = {face_distances[best_match_index]}')
                top, right, bottom, left = face_locations[face_encoding_index]

                # draw remembered face box
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 180, 0), 2)  # green color

                # draw a label
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            font, 0.5, (255, 255, 255), 1)

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
