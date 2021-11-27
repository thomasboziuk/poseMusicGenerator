import cv2
import mediapipe as mp
import os
from mingus.midi import pyfluidsynth
from time import sleep, time
import tkinter, tkinter.ttk
from PIL import Image
from PIL import ImageTk
import numpy as np
from kalmanStructs import kalmanFilt, genNXYVelMats
import sympy

# Initialize GUI
window = tkinter.Tk()
window.title('Dance to (make) the music')
window.geometry('1200x800')
lbl = tkinter.Label(window, text="We can adjust some parameters here.")    
lbl.grid(column=0, row=0)
style = tkinter.ttk.Style(window)
style.theme_use('clam')

inst_dict= {'Piano': 1, 'Guitar': 26, 'Synth Bass': 39, 'Strings': 49, 'Orchestra Hit': 56, 'Pan Flute': 76, 'Sitar': 105, 'Bag Pipe': 110}
inst_options = inst_dict.keys()
current_inst = tkinter.StringVar(window)
current_inst.set("Select an Instrument")

w1 = tkinter.Scale(window, from_=0, to=20)
w1.set(15)
w1.grid(column=2, row=1)

def setCurrentInst(*args):
  # NEED TO FIX: if prog select set to 0, ALL instruments change???
  # but if set to 1, only the note on channel 1 changes. What gives???
  synth.program_select(0, sfid, 0, inst_dict[current_inst.get()])
  synth.program_select(1, sfid, 0, inst_dict[current_inst.get()])
  synth.program_select(2, sfid, 0, inst_dict[current_inst.get()])
  synth.program_select(3, sfid, 0, inst_dict[current_inst.get()])
  synth.noteon(0,40,80)
  synth.noteon(1,48,80)
  synth.noteon(2,40,80)
  synth.noteon(3,48,80)


inst_menu = tkinter.OptionMenu(
            window,
            current_inst,
            *inst_options, command=setCurrentInst)
inst_menu.grid(column=0, row=1)

gui_image_label = tkinter.Label(window, image=None)
gui_image_label.grid(column=0, row=2)

def breakLoop():
  cap.release()

stop_button = tkinter.Button(window, text="stop", command = breakLoop)
stop_button.grid(column=1,row=1)


# initialize what points we'll be tracking in mediapipe. dictionary of dictionaries will let us track what we want efficiently
trackedPoints = {"left_shoulder": {"mpIndex": 11}, "right_shoulder": {"mpIndex": 12}, "left_wrist": {"mpIndex": 15}, "right_wrist": {"mpIndex": 16}}

# Initialize synth stuff
synth = pyfluidsynth.Synth()
sfid = synth.sfload('/usr/share/sounds/sf2/FluidR3_GM.sf2')
synth.start()
synth.program_select(0, sfid, 0, 49)
synth.program_select(1, sfid, 0, 49)
synth.program_select(2, sfid, 0, 49)
synth.program_select(3, sfid, 0, 49)

# some drum channels, using the synth drum, # 118
# TODO
synth.program_select(4, sfid, 0, 118)
synth.program_select(5, sfid, 0, 118)
sleep(1)

# for getting everything set on ubuntu's sound management (may be different for other OSs)
os.system('jack-matchmaker -e fluidsynth:left system:playback_1 fluidsynth:right system:playback_2 &')
sleep(1)

# define our base notes
synth.noteon(0,40,80)
sleep(1)
synth.noteon(1,48,80)
sleep(1)
synth.noteon(2,40,80)
sleep(1)
synth.noteon(3,48,80)
sleep(1)

# define the pitch wheel sensitivity, in semi-tones.
# had to manually add this to the library! There's two locations where it appears.
synth.pitch_wheel_sens(0,20)
synth.pitch_wheel_sens(1,20)
synth.pitch_wheel_sens(2,20)
synth.pitch_wheel_sens(3,20)

# Based on mediapipe example: https://google.github.io/mediapipe/solutions/pose#python-solution-api
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize the Kalman filter settings
'''
old, manual way. new way follows

x = np.array([0.5,0.5,0.1,0.1])
P = np.array([[.1**2,0,0,0],[0,.1**2,0,0],[0,0,.05**2,0],[0,0,0,.05**2]])
R = np.array([[.1**2,0],[0,.1**2]])
Q = np.array([[.5,0,0,0],[0,.5,0,0],[0,0,.1,0],[0,0,0,.1]])
dt = sympy.symbols('dt')
A = sympy.Matrix([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
H = np.array([[1,0,0,0],[0,1,0,0]])
'''
x, P, R, Q, A, H = genNXYVelMats(len(trackedPoints))

kalmFilt = kalmanFilt(x,P,R,Q,A,H)




# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  lastTime = time()
  leftHandTime = time()
  rightHandTime = time()
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


    # get x, y, z coordinates in handier form
    if results.pose_landmarks:
      keypoints = []
      for data_point in results.pose_landmarks.landmark:
          keypoints.append({
                              'X': data_point.x,
                              'Y': data_point.y,
                              'Z': data_point.z,
                              'Visibility': data_point.visibility,
                              })

      shoulderx1 = keypoints[11]['X']
      shoulderx2 = keypoints[12]['X']
      shouldery1 = keypoints[11]['Y']
      shouldery2 = keypoints[12]['Y']
      leftHandx = keypoints[15]['X']
      leftHandy = keypoints[15]['Y']
      rightHandx = keypoints[16]['X']
      rightHandy = keypoints[16]['Y']

      # Update Kalman Filter using new measurement
      currTime = time()
      timestep = currTime - lastTime
      lastTime = currTime
      kalm_pos = kalmFilt.update(np.array([shoulderx1,shouldery1,shoulderx2,shouldery2,leftHandx,leftHandy,rightHandx,rightHandy]), timestep)


    else:
      # Use Kalman Filter to propogate state anyways
      print('using kalman to propogate instead')
      currTime = time()
      timestep = currTime - lastTime
      # we don't update last time because we don't have a measurement. that will only be updated with a measurement,
      # to match the expected behavior for kalmanFilter.update versus kalmanFilter.propogate
      kalm_pos = kalmFilt.propogate(timestep)
    
    # adjust synth value and bend tone based on x/y locations of shoulders
    #UPDATE TO KALMAN VALUES
      
    shoulderx1 = kalm_pos[0]
    shoulderx2 = kalm_pos[2]
    shouldery1 = kalm_pos[1]
    shouldery2 = kalm_pos[3]
    

    v_shoulder_1 = kalm_pos[8:10]
    v_shoulder_2 = kalm_pos[10:12]
    leftHandVel = (kalm_pos[12]**2 + kalm_pos[13]**2)**.5
    rightHandVel = (kalm_pos[14]**2 + kalm_pos[15]**2)**.5
    print(shoulderx1, shouldery1, v_shoulder_1, shoulderx2, shouldery2, v_shoulder_2, leftHandVel, rightHandVel)
    
    height,width,channels = image.shape
    # kalman filter can propogate beyond desired range, so we will cap it here
    if shoulderx1 > 1:
      shoulderx1 = 1
    if shouldery1 > 1:
      shouldery1 = 1
    if shoulderx2 > 1:
      shoulderx2 = 1
    if shouldery2 > 1:
      shouldery2 = 1
    image = cv2.circle(image,(int(kalm_pos[0]*width),int(kalm_pos[1]*height)),3,thickness=1,color=(0,255,0))
    image = cv2.circle(image,(int(kalm_pos[2]*width),int(kalm_pos[3]*height)),3,thickness=1,color=(0,255,0))
    image = cv2.circle(image,(int(kalm_pos[4]*width),int(kalm_pos[5]*height)),3,thickness=1,color=(0,255,0))
    image = cv2.circle(image,(int(kalm_pos[6]*width),int(kalm_pos[7]*height)),3,thickness=1,color=(0,255,0))
    
    # bends happen here
    synth.pitch_bend(0,int(shoulderx1*16000-8000))
    synth.pitch_bend(1,int(shouldery1*16000-8000))
    synth.pitch_bend(2,int(shoulderx2*16000-8000))
    synth.pitch_bend(3,int(shouldery2*16000-8000))

    # velocity (from kalman estimator) is used to scale volume
    synth.cc(0,11,int(abs(v_shoulder_1[0])*600+30))
    synth.cc(1,11,int(abs(v_shoulder_1[1])*600+30))
    synth.cc(2,11,int(abs(v_shoulder_2[0])*600+30))
    synth.cc(3,11,int(abs(v_shoulder_2[1])*600+30))

    # drum channels play if the time has been over one second and the velocity is above threshold
    # TODO: ONLY HAPPENS IF HAND IN FRAME
    percussion_sensitivity = w1.get() / 100
    if leftHandVel > percussion_sensitivity and currTime > leftHandTime + 0.5:
      synth.noteon(4,40,80)
      leftHandTime = currTime
    if rightHandVel > percussion_sensitivity and currTime > rightHandTime + 0.25:
      synth.noteon(5,80,80)
      rightHandTime = currTime

    # update GUI with new picture
    guiImage = Image.fromarray(np.flip(cv2.flip(image,1), axis=-1))
    guiImage = ImageTk.PhotoImage(guiImage)
    gui_image_label.configure(image = guiImage)
    window.update()
    # break on esc key, this no longer works because I don't draw with opencv. TOREMOVE
    if cv2.waitKey(5) & 0xFF == 27:
      break

  # release camera and synth
  cap.release()
  synth.noteoff(0,40)
  synth.noteoff(1,48)
  synth.noteoff(2,40)
  synth.noteoff(3,48)
  synth.delete()