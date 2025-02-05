%%% Created by Dr. Benjamin Vien (Monash Unversity)
% Function- Updated 6/2024 Version 1
% Content - Updated 6/2/2025

% Requirements
%	Version '23.2'	'MATLAB' (R2023b)

%% 1. Preliminary
filename='IMG_7296.mp4';
vidObj=VideoReader(filename);
vidReader=vision.VideoFileReader(filename);
vidReader.VideoOutputDataType='double';
hblobanalysis=vision.BlobAnalysis('MinimumBlobArea',30,'MaximumBlobArea',1000,'LabelMatrixOutputPort',1500,'MaximumCount',1500);
vidPlayer = vision.DeployableVideoPlayer;

%Save statistics of dots
saved_objC=NaN(1500, 2, vidObj.NumFrames);
saved_objA=NaN(1500, 1, vidObj.NumFrames);

%Save Video of Outputs.
outputV=1;
if outputV==1
    outputVideo = VideoWriter('Type_CAXXX_featureshow2.mp4', 'MPEG-4');
    open(outputVideo);
end


%% 2. Read Original Video and Extracts Dots
counter0=1;
while ~isDone(vidReader)

    sampleframe = step(vidReader);
    k=read(vidObj,counter0);
    sampleframe=k; %Odd

    %% Convert RGB image to chosen color space
    I = rgb2ycbcr(sampleframe);

    % Define thresholds for channels 1-3 based on histogram settings
    channel1Min = 0.000;
    channel1Max = 255.000;

    channel2Min = 0.000;
    channel2Max = 118.000;

    channel3Min = 138.000;
    channel3Max = 255.000;

    % Create mask based on chosen histogram thresholds
    sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
        (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
        (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
    BW = sliderBW;
    BW = ~BW; % Invert mask

    %% Largest connecting component

    CC0 = bwconncomp(BW);
    stats_BIG = regionprops(CC0, 'Area');
    [~, idxBIG] = max([stats_BIG.Area]);
    % Create a new binary mask that keeps only the largest component
    BW_largest = false(size(BW));      % Initialise a blank mask
    BW_largest(CC0.PixelIdxList{idxBIG}) = true;  % Assign the largest component's pixels
    BW=BW_largest;

    figure(8)
    subplot(1,3,1)
    imshow(BW)
    title(counter0)

    % Initialize output masked image based on input image.
    maskedRGBImage = sampleframe;
    maskedRGBImage(repmat(~BW,[1 1 3])) = 255;

    figure(8)
    subplot(1,3,2)
    imshow(maskedRGBImage)
    bw_image=im2bw(maskedRGBImage);

    %% Constraint Dot Size
    factorRes=0.20;
    c = bwconncomp(~bw_image);

    stats = regionprops(cc, 'Area');
    stats2= regionprops("table",~bw_image,'Centroid','MajorAxisLength','MinorAxisLength');

    majorR=stats2.MajorAxisLength;
    minorR=stats2.MinorAxisLength;

    dot_idx2a = find([stats2.MajorAxisLength] >= 30*factorRes & [stats2.MajorAxisLength] <= 90*factorRes);
    dot_idx2b = find([stats2.MinorAxisLength] >= 30*factorRes & [stats2.MinorAxisLength] <= 90*factorRes);
    dot_idx2T = intersect(dot_idx2a , dot_idx2b );

    filtered_dots = ismember(labelmatrix(cc), dot_idx2T);
    infostat2=stats2(dot_idx2T,:);

    figure(8)
    subplot(1,3,3)
    imshow(filtered_dots)

    %%

    [objA,objC,boxOUT]=step(hblobanalysis,filtered_dots);
    saved_objC(1:length(objC),:,counter0)=objC;
    saved_objA(1:length(objA),counter0)=objA;

    % Circling
    radius = min(boxOUT(:,3), boxOUT(:,4)) / 2; 
    ishape = insertShape(sampleframe, "filled-circle", [objC(:,1),objC(:,2), radius], ...
        'LineWidth', 1, 'Color', 'm');

    numObj = length(objA);
    texty = sprintf('%d', numObj);
    Icombined = insertText(ishape, [20 20], texty, 'FontSize', 40, 'TextColor', 'white');
    numbercount(counter0)=numObj;

    %Play in the video player
    step(vidPlayer,Icombined);

    if outputV==1
        writeVideo(outputVideo, Icombined);
    end
    counter0=counter0+1;
end
%% Cleanup
release(vidReader)
release(hblobanalysis)

release(vidPlayer)
if outputV==1
    close(outputVideo);
end
%% Saving h5
h5create('saved_objC.h5', '/saved_objC', size(saved_objC));
h5write('saved_objC.h5', '/saved_objC', saved_objC);
