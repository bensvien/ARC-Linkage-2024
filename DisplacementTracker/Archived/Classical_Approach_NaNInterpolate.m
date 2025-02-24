%%% Created by Dr. Benjamin Vien (Monash Unversity)
% Function- Updated 2/2025 Version 1
% Content - Updated 2/2025

% Requirements
%	Version '23.2'	'MATLAB' (R2023b)

%% Ensure the ordered code is performed first!

load('saved_objC_ordered.mat')

data=ordered_data;
% data=data_interpolated;
nan_mask = isnan(data);  % same size as data
frames_with_nan_logical = squeeze(any(any(nan_mask, 1), 2));
frames_with_nan = find(frames_with_nan_logical);
disp(frames_with_nan);

%%
% Preallocate the output array
data_interpolated = data;  % Copy original data

nPoints = size(data, 1);  % 1024
nCoords = size(data, 2);  % 2
nFrames = size(data, 3);  % 483

% Loop over each point and each coordinate
for i = 1:nPoints
    disp(i)
    for j = 1:nCoords
        % Extract the time series (across frames) for the current (point, coordinate)
        timeSeries = squeeze(data(i, j, :));
        
        % Find the indices where data is valid (not NaN)
        validIdx = ~isnan(timeSeries);
        frameIdx = (1:nFrames)';
        
        % If there are valid points, perform interpolation
        if any(validIdx)
            % interp1 with 'linear' will interpolate between valid points
            % 'extrap' is used to handle cases at the boundaries where needed.
            timeSeries_interp = interp1(frameIdx(validIdx), timeSeries(validIdx), frameIdx, 'linear', 'extrap');
            
            % Store the interpolated data back
            data_interpolated(i, j, :) = timeSeries_interp;
        end
    end
end
