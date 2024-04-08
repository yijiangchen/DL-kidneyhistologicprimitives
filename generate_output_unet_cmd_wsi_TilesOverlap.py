# %%

# coding: utf-8

# %%


import argparse
import os
import glob
import numpy as np
import cv2
import torch
import sys
import time
import matplotlib.pyplot as plt
from WSI_handling import wsi
import math
import copy
# %%

from unet import UNet

# %%
#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


# %%

# ----- parse command line arguments
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")

# %%
parser.add_argument('-r', '--resolution', help="image resolution in microns per pixel", default=1, type=float)
parser.add_argument('-c', '--color', help="annotation color to use, default None", default='green', type=str)
parser.add_argument('-a', '--annotation', help="annotation index to use, default largest", default='wsi', type=str)

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-m', '--model', help="model", default="best_model.pth", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)

# %%
#args = parser.parse_args()
args = parser.parse_args(['-s3','-i0','-p256',
                          '-o/newscratch/scratch/Jaidip2/Test20x_Nephro/Output/glomunit_Testset0',
                          '-r2','-awsi','-m./ROI_example/pas-normal-glomerular-unit-5X_unet_best_model.pth',
                          '/newscratch/scratch/Jaidip2/Test20x_Nephro/Test/*.svs'])

#r0.25 capillaries_PAS_03_05_unet_03_01_adam.pth     r2 pas-normal-glomerular-unit-5X_unet_best_model.pth
# %%


if not (args.input_pattern):
    parser.error('No images selected with input pattern')

# %%
OUTPUT_DIR = args.outdir

# %%

batch_size = args.batchsize
patch_size = args.patchsize
base_stride_size = patch_size//2

# %%

# ----- load network
device = torch.device(args.gpuid if torch.cuda.is_available() else 'cpu')


# %%

checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
             padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
             up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
model.load_state_dict(checkpoint["model_dict"])
model.eval()

# %%
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# ----- get file list

# %%

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %%

files = []
basepath = args.basepath  #
basepath = basepath + os.sep if len(
    basepath) > 0 else ""  # if the user supplied a different basepath, make sure it ends with an os.sep

# %%


if len(args.input_pattern) > 1:  # bash has sent us a list of files
    files = args.input_pattern
elif args.input_pattern[0].endswith("tsv"):  # user sent us an input file
    # load first column here and store into files
    with open(args.input_pattern[0], 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            files = line.split("\t")
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.basepath + args.input_pattern[0])


# %%

Missed_fles=[]
# ------ work on files
for fname in files:
    
    fname = fname.strip()
    # TODO: provide class name
    newfname_class = "%s/%s_glomunit5xmask.png" % (OUTPUT_DIR, os.path.basename(fname)[0:fname.rfind('.')])

    print(f"working on file: \t {fname}")
    print(f"saving to : \t {newfname_class}")

    if not args.force and os.path.exists(newfname_class):
        print("Skipping as output file exists")
        continue
    start_time = time.time()
    cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))                                            
    
    xml_dir = fname[0:fname.rfind(os.path.sep)]+'_xml'
    xml_fname = xml_dir + os.path.sep + os.path.basename(fname)[0:os.path.basename(fname).rfind('.')] + '.xml'

    img = wsi(fname,xml_fname)
    
    stride_size = int(base_stride_size * (args.resolution/img["mpp"]))
    stride_size=math.ceil(stride_size/2)*2
    if(args.annotation.lower() == 'wsi'):
        img_dims0 = [0,0,img["img_dims"][0][0],img["img_dims"][0][1]]
    # else:
        img_dims = img.get_dimensions_of_annotation(colors_to_use=args.color,annotation_idx=0)#args.annotation
    
    output = None
    output_batch = None
    im_mask = None
    if img_dims:
    
        x_start = int(img_dims[0])
        y_start = int(img_dims[1])
        w_orig = int(img_dims[2]) - x_start # width/columns of cortex region
        h_orig = int(img_dims[3]) - y_start # height/rows of cortex region only.

        w = int(w_orig + (stride_size - (w_orig % (stride_size)))) # next when remove padding, convert mask to oroginal 'mpp' resolution and remove added pixel at the end.
        h = int(h_orig + (stride_size - (h_orig % (stride_size))))

        x_points = range(x_start-stride_size//2,x_start+w+stride_size//2,stride_size) 
        y_points = range(y_start-stride_size//2,y_start+h+stride_size//2,stride_size) 

        grid_points = [(x,y) for x in x_points for y in y_points]
        points_split = divide_batch(grid_points,batch_size)

        #in case we have a large network, lets cut the list of tiles into batches
        output = np.zeros((len(grid_points),base_stride_size,base_stride_size),np.uint8) # shape (tiles,128,128)
        for i,batch_points in enumerate(points_split):
            if i %25 == 0:
                print(i,'of batch-groups ', len(grid_points)/batch_size)
                # print(i,batch_points)
                # break

            batch_arr1 = np.array([img.get_tile(img["mpp"],coords,(stride_size*2,stride_size*2)) for coords in batch_points]) #img["mpp"]
            batch_arr=np.zeros((batch_arr1.shape[0],patch_size,patch_size, batch_arr1.shape[3]),np.uint8)
            for ith in range(batch_arr1.shape[0]):
                batch_arr[ith,:,:,:] = cv2.resize(batch_arr1[ith,:,:,:], (patch_size,patch_size), interpolation = cv2.INTER_CUBIC)
                
            # im_concat=np.concatenate((batch_arr[0,64:192,64:192,:],batch_arr[1,64:192,64:192,:],batch_arr[2,64:192,64:192,:]),axis=0)
            im_batch=batch_arr[:,base_stride_size//2:-base_stride_size//2,base_stride_size//2:-base_stride_size//2,:]
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
           
            # ---- get results
            output_batch = model(arr_out_gpu)
            # --- pull from GPU and append to rest of output 
            output_batch = output_batch.detach().cpu().numpy()
            output_batch = output_batch.argmax(axis=1)            
            #remove the padding from each tile, we only keep the center
            output_batch = output_batch[:,base_stride_size//2:-base_stride_size//2,base_stride_size//2:-base_stride_size//2] #base stride size//2=64 to get only center (128,128) pixels
                            
            output[((i+1)*batch_size - batch_size):((i+1)*batch_size),:,:] = output_batch
            
        #turn from a single list into a matrix of tiles       
        output = output.reshape(len(x_points),len(y_points),base_stride_size,base_stride_size)
        output = np.concatenate(np.concatenate(output.transpose(1,0,2,3),1),1)
        output1=copy.deepcopy(output)
        # RESIZE to (w,h) and then remove padding
        output=cv2.resize(output, (batch_points[-1][0]+int(stride_size*1.5)-x_start, batch_points[-1][1]+int(stride_size*1.5)-y_start), interpolation=cv2.INTER_CUBIC)
        output=output[:-(stride_size - (h_orig % (stride_size))), :-(stride_size - (w_orig % (stride_size)))]
       
        if(args.annotation.lower() == 'wsi'):
        #     cv2.imwrite(newfname_class, output);

        # else:
            # Add padding for orginal size image
            print('Raw Image size=',img_dims0)
            Whole_mask= np.zeros((img_dims0[3]+2000, img_dims0[2]+2000), np.uint8) # Predicted mask has unremoved padding cause large size image than wholse mask so added padding to whole mask
            try:
                Whole_mask[img_dims[1]:(img_dims[1]+output.shape[0]),img_dims[0]:(img_dims[0]+output.shape[1])] =output#remove added 2000 padding to whole mask
                Whole_mask1=Whole_mask[:-2000,:-2000]
                cv2.imwrite(newfname_class, Whole_mask1)
            except ValueError:
                Missed_fles.append(fname)

    else:
        print('No annotation of color')

    print('Elapsed time = ' + str(time.time()-start_time))
print(Missed_fles)





# %% Overlay image with predicted mask


'''
import matplotlib.pyplot as plt
import numpy as np

oimg,msk11=img.get_annotated_region(img['mpp'],args.color, annotation_idx=0)# img.get_wsi(desired_mpp=2) >>gives whole image
width=oimg.shape[1]; height=oimg.shape[0]
x3=np.asarray(output,'uint8')
mask = cv2.threshold(x3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
oimg[mask==255] = (36,255,12)
plt.figure()
plt.imshow(oimg)

plt.savefig('MaskOverlay.png')

'''

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
