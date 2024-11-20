%% APWSHM 2024 Conference FE-Data Generation Code
% Created by Dr. Benjamin Vien (Monash Unversity)
% Function- Updated 1/12/2022 Version 1
% Content - Updated 20/11/2024

% Requirements 
%	Version '23.2'	'MATLAB' (R2023b)
%	Version '23.2'	'Partial Differential Equation Toolbox'


%% 1. Preliminary Initiationalisation
model = createpde("structural","static-solid");
g=importGeometry(model,'50by50by1flatvm.stl'); %% Geometry 50x50x1 to 500mm x 500mm x 10mm
scale(model.Geometry,10^-2) %% Rescale to metres 0.5m x 0.5m x 0.01m

% Commment Out/In
rng(2) % Random Seed for Training
% rng(3) % Random Seet for Validation

%% Create discretiseation- Vertex Points for applied forcing points
xstep=0.100;
th=0;
for i=xstep:xstep:(0.500-xstep)
    for j=xstep:xstep:(0.500-xstep)
        v0=[i j th];
        if i==xstep && j==xstep
            V=v0;
        else
            V=[V;v0];
        end
    end
end

%% Generates Model
VertexID = addVertex(g,"Coordinates",V);
structuralProperties(model,"YoungsModulus",30E6,"PoissonsRatio",0.33);
generateMesh(model);

% Visualsation Check
figure(1)
subplot(2,1,1)
pdegplot(model,"EdgeLabel","off","VertexLabels","on","FaceLabels","on","FaceAlpha",0.1);
title("Geometry with Labels")
grid on
axis off
set(gcf,'color','w')
subplot(2,1,2)
pdemesh(model)


%% 2. Data Generation and Saving
% Initial array empty list
ztotal={};
listtotal={};

tic
for i=(length(ztotal)+1):1:10000 %Change number of data required here (Default) 10,000 samples
    [zval list0,R]=strain_gen(model);
    fprintf(['Solved ' num2str(i) ' \n'])
    ztotal{i}=zval;
    listtotal{i}=list0;
end
toc

%% Saves Data, if too large use -v7.3 (Comment out/in)
% save('test_data.mat','listtotal','ztotal')
save('test_data.mat','listtotal', 'ztotal', '-v7.3')

%% Optinal visualisation check
showpicture=1;
if showpicture==1
    [xq,yq] = meshgrid(0:1/100:50/100);
    ztemp=ztotal{1};
    xvalu=ztemp.xval;
    yvalu=ztemp.yval;
    zvalu=ztemp.Displacement.Magnitude;
    vq = griddata(xvalu,yvalu,zvalu,xq,yq);
    surf(xq,yq,vq)
    colormap(jet)
    view(2)
end

%% Functions

function [zval,list0,R]=strain_gen(model)

%Limitations for max/min/steps
imin=-100;
imax=100; 
dx0=10; 

%Random Number of Forces up to 5 for Vertices.
tempvertex=8+randperm(16,randi(5));
tempforce2 =[];

for ii=1:length(tempvertex)
    tempforce=[randi([imin,imax])*dx0,randi([imin,imax])*dx0,randi([imin,imax])*dx0];
    tempforce2=[tempforce2;tempforce];
end

%Random - No Repeat of Faces up to 4 Faces
facelist=[1,2,4,6];
randface=randi(4);
num_pos_face=randperm(4,4);
tempface= facelist(num_pos_face(1:randface));
if randface<4
    faceforces=randi(4-randface+1)-1;
    if faceforces>0
        for ii=1:1:faceforces
            facetemp(ii)=facelist(num_pos_face(randface+ii));
            tempforceface(ii,:)=[randi([imin,imax])*10,randi([imin,imax])*10,randi([imin,imax])*10];
        end
    else
        facetemp=[];
        tempforceface=[];
    end
else
    facetemp=[];
    tempforceface=[];
end

list0={tempvertex tempforce2, tempface, facetemp,tempforceface}; %List of conditions
delete(model.BoundaryConditions) %Resets

%Fixed Constraints
structuralBC(model,"Face",tempface,"constraint","fixed");

%Forcing on Vertices and Fraces
for ii=length(tempvertex)
    structuralBoundaryLoad(model,"vertex",tempvertex(ii),"Force",tempforce2(ii,:));
end

if randface<4
    if faceforces>0
        for ii=1:1:faceforces
            structuralBoundaryLoad(model,"Face",facetemp,"SurfaceTraction",tempforceface(ii,:));
        end
    end
end

R = solve(model);

% Outputs List
[inda indab]=find(R.Mesh.Nodes(3,:)==0.01);
zval.xval=R.Mesh.Nodes(1,indab);
zval.yval=R.Mesh.Nodes(2,indab);
zval.Displacement.z=R.Displacement.z(indab);
zval.Displacement.x=R.Displacement.x(indab);
zval.Displacement.y=R.Displacement.y(indab);
zval.Displacement.Magnitude=R.Displacement.Magnitude(indab);
zval.Strain.xx=R.Strain.xx(indab);
zval.Strain.yy=R.Strain.yy(indab);
zval.Strain.zz=R.Strain.zz(indab);
zval.Strain.hoop=R.Strain.hoop(indab);
zval.Strain.yz=R.Strain.yz(indab);
zval.Strain.xz=R.Strain.xz(indab);
zval.Strain.xy=R.Strain.xy(indab);
end
