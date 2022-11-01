import cv2

webCamera = cv2.VideoCapture(0)
classificadorVideoFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classificadorOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

while True:
    camera, frame = webCamera.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = classificadorVideoFace.detectMultiScale(cinza)

    for(x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
        pegaOlho = frame[y:y + a, x:x + y]
        OlhoCinza = cv2.cvtColor(pegaOlho, cv2.COLOR_BGR2GRAY)
        localizaOlho = classificadorOlho.detectMultiScale(OlhoCinza)
        for(ox, oy, ol, oa) in localizaOlho:
            radius = ol // 2
            eye_ox = int(ox + 0.5*ol)
            eye_oy = int(oy + 0.5*oa)
            cv2.circle(pegaOlho, (eye_ox, eye_oy), radius, (0, 255, 0), 2)
            #cv2.rectangle(pegaOlho, (ox, oy), (ox + ol, oy + oa), (0, 0, 255), 2)

    cv2.imshow("Imagem WebCamera", frame)

    if cv2.waitKey(1) == ord('f'):
        break

webCamera.release()
cv2.destroyAllWindows()
