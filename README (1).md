# Facial-recognition-matlab
This project aims to develop a system which will automatically Capture image(150 images) in few seconds and then uses those image dataset it will detect the person in fromt of the camera.

# Steps
1. Data Collection
2. Train model
3. Test Model

# Code
    
## DATA COLLECTION:
    clc
    clear all
    close all
    warning off;
    cao=webcam;
    faceDetector=vision.CascadeObjectDetector;
    c=150;
    temp=0;
    while true
        e=cao.snapshot;
        bboxes =step(faceDetector,e);
        if(sum(sum(bboxes))~=0)
        if(temp>=c)
            break;
        else
        es=imcrop(e,bboxes(1,:));
        es=imresize(es,[227 227]);
        filename=strcat(num2str(temp),'.bmp');
        imwrite(es,filename);
        temp=temp+1;
        imshow(es);
        drawnow;
        end
        else
            imshow(e);
            drawnow;
        end
    end
## Train model
    clc
    clear all
    close all
    warning off
    g=alexnet;
    layers=g.Layers;
    layers(23)=fullyConnectedLayer(2);
    layers(25)=classificationLayer;
    allImages=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames');
    opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
    myNet1=trainNetwork(allImages,layers,opts);
    save myNet1;

## Test model
    clc;
    close;
    clear
    c=webcam;
    load myNet1;
    faceDetector=vision.CascadeObjectDetector;
    while true
        e=c.snapshot;
        bboxes =step(faceDetector,e);
        if(sum(sum(bboxes))~=0)
         es=imcrop(e,bboxes(1,:));
        es=imresize(es,[227 227]);
        label=classify(myNet1,es);
        image(e);
        title(char(label));
        drawnow;
        else
            image(e);
            title('No Face Detected');
        end
    end

# Output
<img width="421" alt="default" src="https://github.com/user-attachments/assets/4ee1c96f-413e-40f1-9338-18e9b04e3ff2" />
<img width="422" alt="msd" src="https://github.com/user-attachments/assets/ab9c92ad-c6ac-4bc2-95d5-48e09d8f5165" />
<img width="420" alt="kohli" src="https://github.com/user-attachments/assets/96df4a5c-7586-4e5c-9158-7edf66dc4927" />
