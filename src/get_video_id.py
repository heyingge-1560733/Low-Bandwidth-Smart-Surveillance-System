mypath = '/Users/freddie/Homeworks/Capstone/data/video_split/burglary_combined'

import os
import re

filenames = next(os.walk(mypath), (None, None, []))[2]  # [] if no file

for file in sorted(filenames):
	m = re.match("Burglary([0-9]+)_frame_([0-9]+).jpg", file)
	if m:
		print(m[1] + "_" + str(int(m[2]) / 3))
	else:
		print("Error")