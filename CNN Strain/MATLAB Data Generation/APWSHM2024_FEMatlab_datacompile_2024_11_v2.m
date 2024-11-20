%% APWSHM 2024 Conference Data Compile Code
% Created by Dr. Benjamin Vien (Monash Unversity)
% Function- Updated 1/12/2022 Version 1
% Content - Updated 20/11/2024

% Requirements
%	Version '23.2'	'MATLAB' (R2023b)

%% Load MATLAB file with ztotal saved
load('datafile.mat')
%%
tic
for i=1:1:length(ztotal)

    zval=ztotal{i};
    x=zval.xval;
    y=zval.yval;
    vx=zval.Displacement.x;
    vy=zval.Displacement.y;
    vz=zval.Displacement.z;
    vexx=zval.Strain.xx;
    veyy=zval.Strain.yy;
    vexy=zval.Strain.xy;

    veq3=vexx; %Dummy Check

    %Create uniform array of solutions (No data augmentation)
    [xq,yq] = meshgrid(0:1/100:31/100,0:1/100:31/100);

    vxq=griddata(x,y,vx,xq,yq);
    vyq=griddata(x,y,vy,xq,yq);
    vzq=griddata(x,y,vz,xq,yq);
    veqxx=griddata(x,y,vexx,xq,yq);
    veqxy=griddata(x,y,vexy,xq,yq);
    veqyy=griddata(x,y,veyy,xq,yq);

    datax(i,:,:,:)=cat(3,vxq,vyq,vzq);
    datay_xx(i,:,:)=veqxx;
    datay_xy(i,:,:)=veqxy;
    datay_yy(i,:,:)=veqyy;

    disp(i)
end
toc
%% Optional Check Visualisation (Internal notes)

showpic=1;
if showpic==1
    [xqw,yqw] = meshgrid(0:1/100:50/100);
    veq3b=griddata(x,y,veq3,xqw,yqw);
    for i=1:1:1
        figure(1)
        surf(xqw,yqw,veq3b)
        hold on
        plot3(x,y,veq3,'o')
        set(gcf,'color','w')
        view(2)
        colormap(jet)
        hold off
    end
    set(gcf,'color','w')
    axis off
end

%%
save('compiled_data_forPYTHON.mat','datax','datay_xx','datay_xy','datay_yy')
%%