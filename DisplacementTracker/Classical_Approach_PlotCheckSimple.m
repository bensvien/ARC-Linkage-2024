% %%% Created by Dr. Benjamin Vien (Monash Unversity)
% Function- Updated 2/2025 Version 1
% Content - Updated 2/2025

% Requirements
%	Version '23.2'	'MATLAB' (R2023b) 

% Specimen 32x32 Grib
% Assuming your variable is named 'data' with size 1024x2x483

data=ordered_data;
frame_index = 1; % Select the frame to visualize

% Extract x and y coordinates for the selected frame
x = squeeze(data(:,1,frame_index));
y = squeeze(data(:,2,frame_index));

% Scatter plot
figure;
scatter(x, y, 'filled');
hold on;

for n = 1:length(x)
    text(x(n), y(n), num2str(n), 'FontSize', 8, 'Color', 'r', 'HorizontalAlignment', 'left');
end

% Formatting
xlabel('X Coordinate');
ylabel('Y Coordinate');
title(['Scatter Plot of Frame ', num2str(frame_index)]);
axis equal;
grid on;
hold off;
