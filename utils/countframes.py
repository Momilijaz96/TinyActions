from imutils.video import count_frames
import os
import matplotlib.pyplot as plt

# count the total number of frames in the video file
path = '/home/mo926312/Documents/datasets/TinyVIRAT-v2/videos/train'

frame_count=[]
# display the frame count to the terminal
for x in os.walk(path):
    for f in x[2]:
        file=x[0]+'/'+f
        print(file)
        total = count_frames(file, override=True)
        print("Total frames: ",total)
        frame_count.append(total)
        
#Get frame count distribution plot
plt.figure()
plt.plot(frame_count)
plt.ylabel("Frame count")
plt.xlabel("Frequency")
plt.title("TinyVIRAT-v2 training Framecount distribution")
plt.savefig('framecount.png')
