function [img, scaleFactor] = parsePfm(filePath)
%parsePfm Parses .pfm images.  The pfm format is described here: http://gl.ict.usc.edu/HDRShop/PFM/PFM_Image_File_Format.html
%   Bring a .pfm image into matlab.  Return image buffer and scale factor from .pfm file.  May throw.

    %Constants
    ALLOW_COMMENTS_AND_BLANK_LINES = true; %Some programs(aka The Gimp) insert 
    %comment lines into .pfm headers.  These lines start with '#'.  This is 
    %*not* part of the .pfm standard.  Keep ALLOW_COMMENTS set to true to 
    %allow this behavior, or set to false and an exception will be thrown when a
    %comment is encountered.  This setting also allows blank lines in the
    %header.


    %Open the file and check for errors
    fid = fopen(filePath, 'r');
    if fid == -1
        error = MException('pfm:IOError', 'Could not open file!');
        throw(error);
    end

    %File opened OK
    %
    %.pfm headers have 3 ASCII lines
    %Line 1: The text 'PF' or 'Pf' where the latter denotes grayscale and
    %the former denotes 3 channel color.
    %Line 2: Two integers, width then height.
    %Line 3: A single signed decimal number S
    %    is S < 0 then the file is little endian
    %    otherwise the file is big endian
    %    |S| is a scale factor to relate pixel samples to a physical
    %    quantity(like radiance for example).
    %

    %Info to determine during header parse
    numChannels = 0; %1 = grayscale, 3 = RGB
    imWidth     = 0;
    imHeight    = 0;
    isBigEndian = 0;
    scaleFactor = 1; %Described above, ignored by this code for now

    %Read the whole 3 line header
    if ALLOW_COMMENTS_AND_BLANK_LINES
        COMMENT_CHAR = '#';
        line1 = fgetl(fid);
        while numel(line1) == 0 || line1(1) == COMMENT_CHAR
            line1 = fgetl(fid);
        end
        line2 = fgetl(fid);
        while numel(line2) == 0 || line2(1) == COMMENT_CHAR
            line2 = fgetl(fid);
        end
        line3 = fgetl(fid);
        while numel(line3) == 0 || line3(1) == COMMENT_CHAR
            line3 = fgetl(fid);
        end
    else
        line1 = fgetl(fid);
        line2 = fgetl(fid);
        line3 = fgetl(fid);
    end
    if ~(ischar(line1) && ischar(line2) && ischar(line3))
        fclose(fid);
        error = MException('pfm:IOError', 'Header was incomplete!');
        throw(error);
    end

    %Parse line 1, determine color or BW
    if strcmp(line1, 'PF') == 1 %Color
        numChannels = 3;
    elseif strcmp(line1, 'Pf ') == 1 | strcmp(line1, 'Pf') == 1 %Gray
        numChannels = 1;
    else %Invalid header
        fclose(fid);
        error = MException('pfm:IOError', 'Invalid .pfm header!');
        throw(error);
    end

    %Parse line 2, get image dims
    [dims, foundCount, errMsg] = sscanf(line2, '%u %u');
    if numel(dims) ~= 2 || strcmp(errMsg,'') ~= 1 || foundCount ~= 2
        fclose(fid);
        error = MException('pfm:IOError', 'Dimensions line was malformed!');
        throw(error);
    end
    imWidth  = dims(1);
    imHeight = dims(2);

    %Line 3, the endianness+scale line
    [scale, matchCount, errMsg] = sscanf(line3, '%f');
    if matchCount ~= 1 || strcmp(errMsg,'') ~= 1
        fclose(fid);
        error = MException('pfm:IOError', 'Endianness+Scale line was malformed!');
        throw(error);
    end
    scaleFactor = abs(scale);
    endianChar = 'n';
    if scale < 0.0
        isBigEndian = 0;
        endianChar = 'l';
    else
        isBigEndian = 1;
        endianChar = 'b';
    end
    
    %Allocate image buffer
    img = zeros(imHeight, imWidth, numChannels);
    totElems = numel(img);

    %Now at last parse in the pixel raster
    %the raster is a 4 byte valeues arranged left to right, starting in the
    %upper left corner of the image.  In the case of a color image,
    %channels are interleaved
    [rawData, numFloatsRead] = fread(fid, totElems, 'single', 0, endianChar);
    if numFloatsRead ~= totElems
        fclose(fid);
        error = MException('pfm:IOError', 'Raster data did not match header description!');
        throw(error);
    end
    fclose(fid);

    %Put the data into the output buffer
    if numChannels == 1
        img = rot90(reshape(rawData, imWidth, imHeight, numChannels));
    else
        rBuf = rot90(reshape(rawData(1:numChannels:numel(rawData)), imWidth, imHeight));
        gBuf = rot90(reshape(rawData(2:numChannels:numel(rawData)), imWidth, imHeight));
        bBuf = rot90(reshape(rawData(3:numChannels:numel(rawData)), imWidth, imHeight));
        img = cat(3, rBuf, gBuf, bBuf);
    end
end


