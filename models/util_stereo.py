import math
import png
import struct

from util import *


# Reads a .txt file containing the reported runtime in seconds for a Middlebury
# result as a float value as text. Returns the time as a float.
def ReadMiddlebury2014TimeFile(path):
    time = -1
    with open(path, 'rb') as time_file:
        text = time_file.read().decode('UTF-8').strip()
        try:
            time = float(text)
        except ValueError:
            raise Exception('Cannot parse time file: ' + path)
    return time


# Returns a dict which maps the parameters to their values. The values (right
# side of the equal sign) are all returned as strings (and not parsed).
def ReadMiddlebury2014CalibFile(path):
    result = dict()
    with open(path, 'rb') as calib_file:
        for line in calib_file.readlines():
            line = line.decode('UTF-8').rstrip('\n')
            if len(line) == 0:
                continue
            eq_pos = line.find('=')
            if eq_pos < 0:
                raise Exception('Cannot parse Middlebury 2014 calib file: ' + path)
            result[line[:eq_pos]] = line[eq_pos + 1:]
    return result


# Writes a calib.txt file according to the Middlebury format, given the required
# values.
def WriteMiddlebury2014CalibFile(path,
                                 left_fx, left_fy, left_cx, left_cy,
                                 right_fx, right_fy, right_cx, right_cy,
                                 baseline_in_mm,
                                 width,
                                 height,
                                 ndisp):
    with open(path, 'wb') as calib_file:
        calib_file.write(StrToBytes('cam0=[' + str(left_fx) + ' 0 ' + str(left_cx) + '; 0 ' + str(left_fy) + ' ' + str(left_cy) + '; 0 0 1]\n'))
        calib_file.write(StrToBytes('cam1=[' + str(right_fx) + ' 0 ' + str(right_cx) + '; 0 ' + str(right_fy) + ' ' + str(right_cy) + '; 0 0 1]\n'))
        calib_file.write(StrToBytes('doffs=' + str(right_cx - left_cx) + '\n'))
        calib_file.write(StrToBytes('baseline=' + str(baseline_in_mm) + '\n'))
        calib_file.write(StrToBytes('width=' + str(width) + '\n'))
        calib_file.write(StrToBytes('height=' + str(height) + '\n'))
        calib_file.write(StrToBytes('ndisp=' + str(ndisp) + '\n'))


# Reads a .pfm file containing a disparity image in Middlebury format.
# Returns a 3-tuple (width, height, pixels), where pixels is a tuple of floats,
# ordered as in the PFM file (i.e., the bottommost row comes first).
def ReadMiddlebury2014PfmFile(path):
    with open(path, 'rb') as pfm_file:
        state = 0
        word = ''
        width = -1
        height = -1
        little_endian = True
        while True:
            character = pfm_file.read(1).decode('UTF-8')
            if not character:
                raise Exception('Cannot parse pfm file: unexpected end of file')
            elif character == '#' or character == ' ' or character == '\n' or character == '\r' or character == '\t':
                # Parse word
                if word != '':
                    if state == 0:
                        if word != 'Pf':
                            raise Exception('Cannot parse pfm file: header is not "Pf"')
                        state = 1
                    elif state == 1:
                        width = int(word)
                        state = 2
                    elif state == 2:
                        height = int(word)
                        state = 3
                    elif state == 3:
                        little_endian = float(word) < 0
                        break
                
                word = ''
                
                if character == '#':
                    # Skip comment.
                    pfm_file.readline()
                else:
                    # Skip whitespace
                    continue
            
            word += character
        
        # Read float buffer
        pixel_count = width * height
        endian_character = '<' if little_endian else '>'
        pixels = struct.unpack(endian_character + str(pixel_count) + 'f', pfm_file.read(4 * pixel_count))
    
    return (width, height, pixels)


# Writes a .pfm file containing a disparity image according to Middlebury format.
# Expects pixels as a list of floats
def WriteMiddlebury2014PfmFile(path, width, height, pixels):
    with open(path, 'wb') as pfm_file:
        pfm_file.write(StrToBytes('Pf\n'))
        pfm_file.write(StrToBytes(str(width) + ' ' + str(height) + '\n'))
        pfm_file.write(StrToBytes('-1\n'))  # negative number means little endian
        pfm_file.write(struct.pack('<' + str(len(pixels)) + 'f', *pixels))  # < means using little endian


# Converts a Middlebury .pfm disparity image to a Kitti .png disparity image.
def ConvertMiddlebury2014PfmToKitti2015Png(src_path, dest_path):
    (pfm_width, pfm_height, pfm_pixels) = ReadMiddlebury2014PfmFile(src_path)
    
    png_disp = []  # list of rows
    for y in range(pfm_height - 1, -1, -1):  # iterate in reverse order according to pfm format
        in_row = pfm_pixels[y * pfm_width : (y + 1) * pfm_width]
        out_row = []
        for value in in_row:
            if math.isinf(value):
                out_row.append(0)  # invalid value
            else:
                converted_value = max(1, int(round(256.0 * value)))
                if converted_value > 65535:
                    print('Warning: A disparity value of 256 or larger needed to be clamped in the conversion from PFM to Kitti PNG. File: ' + src_path)
                    converted_value = 65535
                out_row.append(converted_value)
        png_disp.append(out_row)
    
    with open(dest_path, 'wb') as dest_png_file:
        png_writer = png.Writer(width=pfm_width, height=pfm_height, bitdepth=16, compression=9, greyscale=True)
        png_writer.write(dest_png_file, png_disp)


# Converts a Kitti .png disparity image to a Middlebury .pfm disparity image.
def ConvertKitti2015PngToMiddlebury2014Pfm(src_path, dest_path):
    # Read .png file.
    disp_reader = png.Reader(src_path)
    disp_data = disp_reader.read()
    if disp_data[3]['bitdepth'] != 16:
        raise Exception('bitdepth of ' + src_path + ' is not 16')
    
    width = disp_data[0]
    height = disp_data[1]
    
    # Get list of rows.
    disp_rows = list(disp_data[2])
    
    # Convert to Middlebury's PFM format.
    disp_float = []
    for y in range(len(disp_rows) - 1, -1, -1):  # iterate in reverse order according to pfm format
        input_line = disp_rows[y]
        for value in input_line:
            if value > 0:
                disp_float.append(float(value) / 256.0)
            else:
                disp_float.append(float('inf'))  # invalid value
    
    WriteMiddlebury2014PfmFile(dest_path, width, height, disp_float)
    
    disp_reader.close()
