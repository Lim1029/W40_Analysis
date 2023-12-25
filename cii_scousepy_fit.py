# import scousepy
from scousepy import scouse

# create pointers to input, output, data
datadir = "/home/mingkang/Desktop/NARIT_Internship/"
outputdir = "/home/mingkang/Desktop/NARIT_Internship/scousepy_fit/try2/"
filename = "FEEDBACK_W40_CII_OC9N.lmv"

# run scousepy
config_file = scouse.run_setup(filename, datadir, outputdir=outputdir)
s = scouse.stage_1(config=config_file, interactive=True)
s = scouse.stage_2(config=config_file)
s = scouse.stage_3(config=config_file)
s = scouse.stage_4(config=config_file, bitesize=True)

# output the ascii table
from scousepy.io import output_ascii_indiv
output_ascii_indiv(s, outputdir)