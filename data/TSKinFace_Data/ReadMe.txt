TSKinFace Dataset

Overview
The TSKinFace Dataset contains the child-parents images used in the paper "Tri-Subject Kinship Verification: Understanding the Core of A Family"  by Xiaoqian Qin, Xiaoyang Tan,and Songcan Chen.

The "TSKinFace" dataset is the first large-scale dataset of families for one-versus-two kin relation. It contains 1015 different family with distinct family names, including 2,589 individuals, with 787 images. 

All images in the dataset are harvested from the internet based on knowledge of public figures family and photo-sharing social network such as flickr.com. Each family contains one child and two parents. 

The final dataset includes 274, 285 and 228 family photos for Father-Mother-Daughter (FM-D), Father-Mother-Son (FM-S) and Father-Mother-Son- Daughter (FM-SD), respectively.

Two kinds of family-based kinship relations are constructed in the TSKinFace database: Father-Mother-Son(FM-S) and Father-Mother-Daughter(FM-D). The FM-S and the FM-D contain 513 and 502 groups of tri-subject kinship relations, respectively. Hence we have 1015 tri-subject groups in our database totally.

The families included in our database are diverse in terms of races as well. For FM-S relation, there are 343 and 170 groups of tri-subject kinship relations for Asian and non-Asian, respectively. And for FM-D relation, the numbers for Asian and non-Asian groups are respectively 331 and 171.

For pair-wise relationships, there are 513 father-son relations, 502 father-daughter relations, 513 mother-son relations, and 502 mother-daughter relations.




Terms of Use
Please adhere to the following terms of use of this dataset. 

This dataset is for non-commercial reseach purposes (such as academic research) only. The images are not allowed to be redistributed (do not pass copies of any part of this collection to others, or post any images on the Internet). 




If you use any part of this image collection in your research, please cite the paper below.

People
Xiaoqian Qin, Xiaoyang Tan,and Songcan Chen


Citation


Bibtex





Dataset and Family Structure
If you spot any mismatch between the face image and the person, please report to us.

The file folder "TSKinFace_source"  is explained below:
 - The file folder FMD,FMS and FMSD contains each kind of family images in original resolution. The name of the image is "relation-image index". 
 - The file folder FMD_information,FMS_information and FMSD_information contains the FaceList and EyeList information for each family.
   The format of the FaceList is: "the kind of the family\the image name"  "the role of the person in a family" "the position of the face in a family"  and "the radius of the face region"  
   The format of the EyeList is : "the kind of the family\the image name"  "the role of the person in a family" "the position of left eye" and "the position of right eye"


The file folder "TSKinFace_cropped"  is explained below:
 - The file folder FMD,FMS and FMSD contains each kind of group images warped. The name of the image is "relation-image index-the role of the person in a family". 
 

 Note, father is expressed as "F" , mother is expressed as "M", son is "S" and daughter is "D" here.


The Evaluation protocal
   We design a verification protocol for our database: the database is equally divided into five folds such that each fold contains nearly
the same number of face groups with kinship relation, which facilitates five-fold cross validation experiments. The following Table lists
the face number index for the five folds of our TSKinFace database. 
   Note, the images of each family in FMSD are decomposed and numbered behind the FMD or FMS relation, respectively.  

     FACE NUMBER INDEX OF EACH FOLD OF THE TSKINFACE DATABASE
     Fold      1          2         3          4          5
     FM-D   [1,100]  [101,200]  [201,300]  [301,400]  [401,502]
     FM-S   [1,102]  [103,204]  [205,306]  [307,408]  [409,513]









