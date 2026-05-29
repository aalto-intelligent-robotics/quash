import argparse
import os

namedict = {}
namedict['a'] = "bathroom" # (should have a toilet and a sink)
namedict['b'] = "bedroom"
namedict['c'] = "closet"
namedict['d'] = "dining room" # (includes “breakfast rooms” other rooms people mainly eat in)
namedict['e'] = "entryway" # /foyer/lobby" # (should be the front door, not any door)
namedict['f'] = "familyroom" # (should be a room that a family hangs out in, not any area with couches)
namedict['g'] = "garage"
namedict['h'] = "hallway"
namedict['i'] = "library" #(should be room like a library at a university, not an individual study)
namedict['j'] = "laundryroom"#/mudroom" # (place where people do laundry, etc.)
namedict['k'] = "kitchen"
namedict['l'] = "living room" # (should be the main “showcase” living room in a house, not any area with couches)
namedict['m'] = "meetingroom" #/conferenceroom
namedict['n'] = "lounge" # (any area where people relax in comfy chairs/couches that is not the family room or living room
namedict['o'] = "office" # (usually for an individual, or a small set of people)
namedict['p'] = "porch" # /terrace/deck/driveway # (must be outdoors on ground level)
namedict['r'] = "rec"#/game # (should have recreational objects, like pool table, etc.)
namedict['s'] = "stairs"
namedict['t'] = "toilet" # (should be a small room with ONLY a toilet)
namedict['u'] = "utilityroom"#/toolroom
namedict['v'] = "tv" # (must have theater-style seating)
namedict['w'] = "workout"#/gym/exercise
namedict['x'] = "outdoor areas" # containing grass, plants, bushes, trees, etc.
namedict['y'] = "balcony" # (must be outside and must not be on ground floor)
namedict['z'] = "other" # room # (it is clearly a room, but the function is not clear)
namedict['B'] = "bar"
namedict['C'] = "classroom"
namedict['D'] = "dining booth"
namedict['S'] = "spa" #/sauna
namedict['Z'] = "junk" # (reflections of mirrors, random points floating in space, etc.)
namedict['-'] = "no label"


def main(input, output):
    inf = open(input, "r")
    ouf = open(output, "w")
    ouf.write("id; name\n")

    lines = inf.readlines()
    first = True
    for line in lines:
        if(first):
            first = False
            continue
        line = line.replace("  ", " ")
        parts = line.split(" ")
        if(parts[0] == "R"):
            label = parts[5]
            name = namedict[label]
            print("region", parts[1], label, name)
            ouf.write(parts[1] + "; " + name + "\n")

    inf.close()
    ouf.close()


# H name label #images #panoramas #vertices #surfaces #segments #objects #categories #regions #portals #levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
# L level_index #regions label  px py pz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
# R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
# P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi  0 0 0 0
# S surface_index region_index 0 label px py pz  nx ny nz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
# V vertex_index surface_index label  px py pz  nx ny nz  0 0 0
# P name  panorama_index region_index 0  px py pz  0 0 0 0 0
# I image_index panorama_index  name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
# C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name 0 0 0 0 0
# O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0
# E segment_index object_index id area px py pz xlo ylo zlo xhi yhi zhi  0 0 0 0 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./parse_house.py")
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/matterport3d/v1/tasks/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.house"
    )
    parser.add_argument(
        '--output', '-o',
        dest="output",
        type=str,
        required=False,
        default="../cfg/house-0.cfg"
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input
    output = FLAGS.output
    main(input, output)