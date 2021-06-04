# Measuring Poverty 

  

Measuring poverty in rural parts of underdeveloped countries can be extremely difficult because their institutions do not have the capacity for regular intake and maintenance of survey and administrative data. A common solution to address the absence of data is to use proxies such as nighttime light radiance captured in satellite imagery: Greater radiance of nighttime lights indicates greater access to electricity, presence of technology, and often higher income in a region. However, this proxy is imperfect, and there is much information missed that would be available in a daytime satellite image of the same region (e.g. tin roofs, tractors, or greater proportion of plowed vs unplowed fields). Because there is an abundance of free (or nearly free) satellite imagery covering the world, the task is to generate features from the combination of _both_ daytime and nighttime satellite imagery that can best predict local poverty.

  

# Our Solution

  

We propose an unsupervised framework for poverty mapping using a Convolutional Variational Encoder-Decoder (inspired by Variational Autoencoders from [Kingma and Welling 2014](https://arxiv.org/abs/1312.6114)), or ConvVED, that encodes an RGB Landsat-8 daytime image to a low-dimension latent space and decodes from the latent space to the corresponding nighttime light image from the Visible Infrared Imaging Radiometer Suite (VIIRS) dataset. By this method, our model learns to identify the features from a daytime satellite image that are predictive of greater nighttime light intensity, and these features are used as input to a downstream supervised model that predicts local average daily household consumption. We compare our results to the previous work of [Jean et al (2016)]([https://science.sciencemag.org/content/353/6301/790](https://science.sciencemag.org/content/353/6301/790)) that used a transfer learning approach for a similar task. Our previous attempts at training a ConvVED for this task (in a previous class) used only ~30k image pairs and a shallow 4-layer convolutional encoder. In the latest iteration, we expand the training data size to 1 million image pairs (reducing the risk of model overfit) and were therefore able to use a larger encoder: Resnet18 (an 18-layer convolutional residual neural network pretrained on ImageNet courtesy of [He et al 2015](https://arxiv.org/pdf/1512.03385.pdf)) treating the 1000-dimension output as the final layer of the encoder. This dramatic expansion of both model size and training data requires a new set of strategies for data acquisition, preprocessing, and training that leverage large-scale computing methods.

  

Note: because much of this repo was developed before this class, below we explicitly identify those modules that were written or modified for this project.

  
  

# Use of Large-Scale Methods

  

## Data Generation / Preprocessing

  

### 1.  Points generation: `preprocessing/generate-points/`
    

Large-scale tools: Numba, mpi4py

Contributors: Matt (though Matt and Mike jointly discussed/agreed on methodology beforehand)

  

To improve the ability of our ConvVED model to decode VIIRS images from Landsat images, we first want to train our model on large numbers of VIIRS and Landsat images from many different locations. Because at the present time, our model is focused on detecting poverty in Sub-Saharan Africa, we focus solely on this area in our training. This module generates the points for which images will later be extracted and trained on.

  

-   `create_subsaharan-africa-shape.ipynb`: Extract polygon for Sub-Saharan Africa from global shapefile.
    
-   `generate_points.py`: Using Numba and mpi4py, randomly sample points in parallel from Sub-Saharan Africa polygon and save points to CSV.
    
-   `generate_points.sh`: Execute `generate_points.py`, generating 1,000,000 total points. (Note that this script is currently being performed on a local machine, but can easily be adapted to be run as a job on the Midway cluster.)
    

  

### 2. Linking points to Landsat pathrows: `preprocessing/get-pathrows/`
    

Large-scale tools: mpi4py, AWS ParallelCluster

Contributors: Mike wrote initial code to match points to pathrows and incorporate MPI. Matt refined (incorporating spatial index) and wrote the final script included in the current repo.

  

Once we have generated our 1,000,000 points, we must retrieve the Landsat and VIIRS images that correspond to those points. The Landsat 8 images are available in a [public S3 bucket](https://registry.opendata.aws/landsat-8/), while the VIIRS images are hosted by the [Earth Observation Group](https://eogdata.mines.edu/nighttime_light/). To retrieve the appropriate Landsat 8 images corresponding to the points we have generated, we must map each pair of coordinates to an appropriate ‘path’ and ‘row’. Path refers to the path of the satellite as it moves along its orbit, while the row refers to the latitudinal center line of a frame of imagery ([USNA](https://www.usna.edu/Users/oceano/pguth/md_help/html/landsat_path_row.html)). Each coordinate corresponds to only one path-row, so to identify the correct Landsat scene that includes the point, we must map the point to a path-row. (Note that each Landsat ‘scene’ is a 185km x 180km digital image of the Earth, and is uniquely identified by its corresponding path and row, as well as the date on which it was taken.)

  

-   `get_pathrows.py`: Get shapefile containing all path-rows.
    
-   `match_pathrows.py`: Use mpi4py to match 1,000,000 coordinates to pathrows in parallel. To make this process more efficient, we first subset the global path-rows shapefile to include only the path-rows in Sub-Saharan Africa, then construct a [spatial index](https://geopandas.org/docs/reference/sindex.html) from the remaining path-rows. As a result, each matching runs in O(log n) time. We then save the coordinate/path-row mappings to a CSV.
    
-   `run.sbatch`: Run `match_pathrows.py` module using MPI on [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/pcluster.html) (instead of Midway).
    
-   `pathrow_match_example.ipynb`: Serially match points in a Jupyter notebook before moving to pcluster.
    

  

### 3. Linking points/pathrows to 2016 Landsat scenes: `preprocessing/get-landsat-scenes/`
    

Large-scale tools: Dask, Amazon EMR, S3
    
Contributors: Mike
    

  

Now that we have 1,000,000 distinct points in Sub-Saharan Africa mapped to their  corresponding path-row, we must generate the links to the TIF files where we can find the appropriate Landsat image corresponding to the path-row. As discussed above, the Landsat scenes are available in a publicly available S3 bucket. This S3 bucket includes a gzipped CSV file that includes metadata on all of the scenes available in the bucket, which is inclusive of the path and row of the scene, the date the scene was taken and processed, and a URL for where the scene can be downloaded.

  

Note: by using S3 as our data source for the Landsat images, we ultimately only need to store the TIFs for those path-rows that we need for training.

  

-   `get-landsat-scenes.ipynb`: Using Dask in an EMR notebook, filter down and join scene list CSV with coordinate/path-rows CSV generated in step (2) above. Add links to bands 2, 3, and 4 (blue, green, and red) TIFs corresponding to appropriate path-row for every image. Note that we only pull links for scenes in 2016, as our VIIRS images are from 2016, as well.
    

  

### 4. Build dataloaders for model training: `detecting_poverty/data_loaders.py`
    

Large-scale tools: S3, multi-worker Data Loaders in PyTorch
    
Contributors: Initial code for dataloaders and dataset objects was written last quarter, when this project was first launched. This quarter, Mike wrote the code adapting the LandsatTransform, ViirsTransform, and LandsatVIIRs objects to the new data being generated in steps (1) - (3) above. Matt subsequently tweaked and refined code.
    

  

To build dataloaders for Landsat images, we open the TIFs for each band (corresponding to one image) directly from S3 using RasterIO, and write the bands into a merged RGB image. Then for each coordinate pair, we extract only that data within a 10km bounding box centered on the coordinate, and perform the necessary image transformations (i.e. normalizing pixel values and resizing the image to 224 x 224) before feeding it into the model. Loading and processing the images in this way is an opportunity for SIMD parallelism which we implemented using the PyTorch DataLoader class.

  

To build dataloaders for VIIRS images, we simply hold the TIF for Sub-Saharan Africa in memory, and then extract a 21x21 image from the TIF corresponding to the appropriate coordinate pair.

  
  

## Model Training

Large-scale tools: GPU, CUDA, (to come: distributed CUDA on AWS ParallelCluster)

Contributors: Matt

Module: `detecting_poverty/conv_ved.py`


  

Development and initial model testing was performed on Google Colab making use of both the increased RAM (25GB) and Tesla T4 GPU that are available with the upgraded Colab account. The ConvVED was both developed and trained using the PyTorch deep learning library. PyTorch comes with CUDA capabilities for both training and running deep neural networks. Neural networks at their core are a series of matrix multiplications — an operation that benefits from the immense parallelization and high throughput of modern GPUs.

However, even the most advanced GPU is not sufficient to handle training a large neural network on 1 million image pairs for several epochs in a reasonable amount of time. Our future work will be to perform model training with distributed data in parallel on a cluster of GPUs using PyTorch’s distributed communication library and a Slurm job scheduler for requesting the necessary compute resources in the AWS environment.
