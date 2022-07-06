import cv2
import os
import shutil
import glob
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime


def selesai1():
    intructions.config(text="Rekam Data Telah Selesai!")


def selesai2():
    intructions.config(text="Training Wajah Telah Selesai!")


def selesai3():
    intructions.config(text="Absensi Telah Dilakukan")


def rekamDataWajah():
    if nama_input.get() == '' or nim_input.get() == '' or kelas_input.get() == '':
        print(nama_input.get(), nim_input.get(), kelas_input.get())
        intructions.config(text="Ada kolom yang masih kosong!")
    else:
        if os.path.exists("DataWajah.csv"):
            df = pd.read_csv("DataWajah.csv", dtype=str)
            lookup = df[df['NIM'] == nim_input.get()]
        else:
            df = pd.DataFrame()
            lookup = []
        if len(lookup) > 0:
            intructions.config(text="Anda sudah merekam wajah. Anda dapat absen langsung")
        else:
            wajahDir = 'datawajah'
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
            # faceID = nim_input.get()
            nama = nama_input.get()
            nim = nim_input.get()
            kelas = kelas_input.get()
            ambilData = 1
            while True:
                retV, frame = cam.read()
                abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)
                for (x, y, w, h) in faces:
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    namaFile = str(nim) + '_'+str(nama) + '_' + str(kelas) + '_' + str(ambilData) + '.jpg'
                    cv2.imwrite(wajahDir + '/' + namaFile, frame)
                    ambilData += 1
                    roiabuabu = abuabu[y:y + h, x:x + w]
                    roiwarna = frame[y:y + h, x:x + w]
                    eyes = eyeDetector.detectMultiScale(roiabuabu)
                    for (xe, ye, we, he) in eyes:
                        cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)
                cv2.imshow('webcamku', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
                    break
                elif ambilData > 30:
                    break
            df = df.append({
                "Nama": nama_input.get(),
                "NIM": nim_input.get(),
                "Kelas": kelas_input.get(),
                "Waktu": datetime.now().strftime('%H:%M:%S')
            }, ignore_index=True)
            df.to_csv("DataWajah.csv", index=False)
            selesai1()
            cam.release()
            cv2.destroyAllWindows()  # untuk menghapus data yang sudah dibaca


def trainingWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'
    imagePaths = glob.glob(f'{wajahDir}/*')
    if len(imagePaths) == 0:
        intructions.config(text="Harap rekam data wajah")
    else:
        # if os.path.exists("DataWajah.csv"):
        df = pd.read_csv("DataWajah.csv", dtype=str)
        lookup = df[df['NIM'] == nim_input.get()]
        # else:
        #     df = pd.DataFrame()

        def getImageLabel(path):
            imagePaths = glob.glob(f'{path}/*')
            faceSamples = []
            faceIDs = []
            for imagePath in imagePaths:
                PILimg = Image.open(imagePath).convert('L')
                imgNum = np.array(PILimg, 'uint8')
                faceID = lookup.index[0]
                # faceID = int(os.path.split(imagePath)[-1].split('_')[0])
                faces = faceDetector.detectMultiScale(imgNum)
                for (x, y, w, h) in faces:
                    faceSamples.append(imgNum[y:y + h, x:x + w])
                    faceIDs.append(faceID)
                return faceSamples, faceIDs

        faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
        faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces, IDs = getImageLabel(wajahDir)
        if os.path.exists(f'{latihDir}/training.xml'):
            faceRecognizer.read(f'{latihDir}/training.xml')
        # faceRecognizer.train(faces, np.array(IDs))
        faceRecognizer.update(faces, np.array(IDs))
        # simpan
        faceRecognizer.write(latihDir + '/training.xml')
        shutil.rmtree(wajahDir)
        os.mkdir(wajahDir)
        df.to_csv("DataWajah.csv", index=False,)
        selesai2()


def markAttendance(id):
    if os.path.exists("Kehadiran.csv"):
        df = pd.read_csv("Kehadiran.csv", dtype=str)
    else:
        df = pd.DataFrame()
    row_data = {
        "id_wajah": str(id),
        "Nama": nama_input.get(),
        "NIM": nim_input.get(),
        "Kelas": kelas_input.get(),
        "Waktu": datetime.now().strftime('%H:%M:%S')
    }
    df = df.append(row_data, ignore_index=True)
    df.to_csv("Kehadiran.csv", index=False)
    # with open("Attendance.csv",'r+') as f:
    #     namesDatalist = f.readlines()
    #     namelist = []
    #     yournim = nim_input.get()
    #     yourclass = kelas_input.get()
    #     for line in namesDatalist:
    #         entry = line.split(',')
    #         namelist.append(entry[0])
    #     if name not in namelist:
    #         now = datetime.now()
    #         dtString = now.strftime('%H:%M:%S')
    #         f.writelines(f'\n{name},{yourclass},{yournim},{dtString}')


def absensiWajah():
    # model.predict()
    if nama_input.get() == '' or nim_input.get() == '' or kelas_input.get() == '':
        print(nama_input.get(), nim_input.get(), kelas_input.get())
        intructions.config(text="Ada kolom yang masih kosong!")
    else:
        wajahDir = 'datawajah'
        latihDir = 'latihwajah'
        if os.path.exists("DataWajah.csv"):
            df = pd.read_csv("DataWajah.csv", dtype=str)
            lookup = df[df['NIM'] == nim_input.get()]
        else:
            df = pd.DataFrame()
            lookup = []
        if len(lookup) == 0:
            intructions.config(text="Anda belum merekam wajah")
        else:
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
            faceRecognizer.read(latihDir + '/training.xml')
            font = cv2.FONT_HERSHEY_PLAIN

            #id = 0
            yourname = nama_input.get()
            names = []
            names.append(yourname)
            minWidth = 0.1 * cam.get(3)
            minHeight = 0.1 * cam.get(4)

            while True:
                retV, frame = cam.read()
                # frame = cv2.flip(frame, 1)
                #
                abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)), )
                for (x, y, w, h) in faces:
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    id, confidence = faceRecognizer.predict(abuabu[y:y+h, x:x+w])
                    if (confidence < 100):
                        nama_hasil = df.loc[id, 'Nama']
                        confidence = "  {0}%".format(round(150 - confidence))
                    elif confidence < 50:
                        nama_hasil = df.loc[id, 'Nama']
                        confidence = "  {0}%".format(round(170 - confidence))

                    elif confidence > 70:
                        nama_hasil = "Tidak Diketahui"
                        confidence = "  {0}%".format(round(150 - confidence))

                    cv2.putText(frame, f'{nama_hasil}', (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    cv2.putText(frame, str(confidence), (x + 5, y + h + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                cv2.imshow('ABSENSI WAJAH', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
                    break
            markAttendance(id)
            selesai3()
            cam.release()
            cv2.destroyAllWindows()

# LOAD MODEL


# GUI
root = tk.Tk()
root.title('Smart Absensi')
root.iconbitmap('cctv.ico')
# mengatur canvas (window tkinter)
canvas = tk.Canvas(root, width=700, height=400)
canvas.grid(columnspan=3, rowspan=8)
canvas.configure(bg="black")
# judul
judul = tk.Label(root, text="Face Attendance - Smart Absensi", font=("Roboto", 34), bg="#242526", fg="white")
canvas.create_window(350, 80, window=judul)
# credit
made = tk.Label(root, text="Powered by Iv4n", font=("Roboto", 10), bg="black", fg="white")
canvas.create_window(360, 20, window=made)


# for entry data nama
nama_input = tk.Entry(root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=nama_input)
label1 = tk.Label(root, text="Nama Siswa", font="Roboto", fg="white", bg="black")
canvas.create_window(90, 170, window=label1)
# for entry data nim
nim_input = tk.Entry(root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=nim_input)
label2 = tk.Label(root, text="NIM", font="Roboto", fg="white", bg="black")
canvas.create_window(60, 210, window=label2)
# for entry data kelas
kelas_input = tk.Entry(root, font="Roboto")
canvas.create_window(457, 250, height=25, width=411, window=kelas_input)
label3 = tk.Label(root, text="Kelas", font="Roboto", fg="white", bg="black")
canvas.create_window(65, 250, window=label3)

global intructions

# tombol untuk rekam data wajah
intructions = tk.Label(root, text="Welcome", font=("Roboto", 15), fg="white", bg="black")
canvas.create_window(370, 300, window=intructions)
Ambil_Gambar_text = tk.StringVar()
Rekam_btn = tk.Button(root, textvariable=Ambil_Gambar_text, font="Roboto", bg="#20bebe",
                      fg="white", height=1, width=15, command=rekamDataWajah)
Ambil_Gambar_text.set("Take Images")
Rekam_btn.grid(column=0, row=7)

# tombol untuk training wajah
Train_wajah_text = tk.StringVar()
Rekam_btn1 = tk.Button(root, textvariable=Train_wajah_text, font="Roboto", bg="#20bebe",
                       fg="white", height=1, width=15, command=trainingWajah)
Train_wajah_text.set("Training")
Rekam_btn1.grid(column=1, row=7)

# tombol absensi dengan wajah
Absensi_text = tk.StringVar()
Rekam_btn2 = tk.Button(root, textvariable=Absensi_text, font="Roboto", bg="#20bebe",
                       fg="white", height=1, width=20, command=absensiWajah)
Absensi_text.set("Automatic Attendance")
Rekam_btn2.grid(column=2, row=7)

root.mainloop()
