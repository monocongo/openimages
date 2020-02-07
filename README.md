# openimages
Tools for downloading images and corresponding annotations from Google's 
[OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset.

## Download images and annotations
The `openimages` package contains a `download` module which provides an API with 
two download functions and a corresponding CLI (command line interface) including 
script entry points that can be used to perform downloading of images and 
corresponding annotations from the OpenImages dataset.

##### Public API
*  `openimages.download.download_images` for downloading images only

    For example, to download all images for the two classes "Hammer" and "Scissors" 
    into the directories "/dest/dir/Hammer/images" and "/dest/dir/Scissors/images":
    ```python
    from openimages.download import download_images
    download_images("/dest/dir", ["Hammer", "Scissors",])
    ```
* `openimages.download.download_dataset` for downloading images and corresponding 
annotations
    For example, to download all images and corresponding annotations in PASCAL 
    VOC format for the two classes "Hammer" and "Scissors" into the directories 
    "/dest/dir/Hammer/[images|pascal]" and "/dest/dir/Scissors/[images|pascal]":
    ```python
    from openimages.download import download_dataset
    download_dataset("/dest/dir", ["Hammer", "Scissors",], annotation_format="pascal")
    ```
##### Command Line Interface
Two Python script entry points are installed when the package is installed into 
a Python environment, corresponding to the public API functions described above: 
`oi_download_dataset` and `oi_download_images`. These commands use the follwing 
options:

Option              | Required | Description
--------------------|----------|-------------
--base_dir \<dir\>  | yes      | directory into which images and annotations will be downloaded, with each class label having a separate subdirectory containing an "images" subdirectory for image files and (for annotated datasets) an \<annotation_format\> subdirectory for annotation files
--labels \<label1\> [\<label_2\> ...] | yes      | space-separated list of class labels, at least one required, multi-word labels with spaces must be quoted
--format \<annotation_format\> | for annotated dataset yes, not applicable for images only  | required for downloading an annotated dataset, currently supported format specifiers are "darknet" and "pascal"
--csv_dir \<dir\> | no, but usually recommended | directory into which the CSV files specifying annotations and class labels are downloaded (if not already present) or read from (if present)
--exclusions \<file\> | no | text file containing image file IDs, one per line, for images to be excluded from the final dataset, useful in cases when images have been identified as problematic
--limit \<int\> | no | the upper limit on the number of images to be downloaded per label class
###### NOTE:
If you'll use these commands more than once then it's imperative to utilize the 
`--csv_dir` option that specifies where to save the (rather large) CSV file containing 
bounding box information etc., as this will save you from having to redownload this 
large file in subsequent usages.

###### Usage examples
Download images and PASCAL format annotations for the class labels "Scissors" and 
"Hammer", limiting the number of images to 200 and storing the CSV files under 
`~/openimages` (reading the CSV files from there if they already exist):
```bash
$ oi_download_dataset --csv_dir ~/openimages --base_dir ~/openimages --labels Scissors Hammer --format pascal --limit 100
```
Download images only for the class label "Scissors", limiting the number of images 
to 100 and storing the CSV files under `~/openimages` (reading the CSV files from 
there if they already exist):
```bash
$ oi_download_images --csv_dir ~/openimages --base_dir ~/openimages --labels Scissors --limit 100
```
