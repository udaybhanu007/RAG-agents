# PDF Content Extraction Report

## Document Text

**FAQ** (Frequently Asked Questions):


Q01: I keep getting errors when I uncompressed the tar.gz files. Could you double check the integrity of
files?


_A: Those errors may be due to the incompleteness of downloaded files. Try to download the files one by_
_one from different computers and places._


Q02: Are the radiological reports included in the publicly accessible chest X-ray dataset?


_A: No. We currently do not have a plan to release the radiological reports._


Q03: Is the source code or pre-trained model available for the unified classification and localization
framework introduced in the paper?


_A: Currently, we do not plan to release the code or pre-trained model._


Q04: Are there any restrictions in using this dataset?


_A: The usage of the data set is unrestricted. But you should provide the link to our original download site,_
_acknowledge the NIH Clinical Center and provide a citation to our CVPR 2017 paper._


Q05: What does 'no finding' label mean?


_A: ‘No finding’ means the 14 listed disease patterns are not found in the image._


Q06: Could you have the X-ray images available in DICOM format?


_A: We are only able to provide the png images._


Q07: Do you have MD5s for the compressed files?


_A: We thank Utku Ozbulak for providing those MD5s._


_fe8ed0a6961412fddcbb3603c11b3698 images_001.tar.gz_


_ab07a2d7cbe6f65ddd97b4ed7bde10bf images_002.tar.gz_


_2301d03bde4c246388bad3876965d574 images_003.tar.gz_


_9f1b7f5aae01b13f4bc8e2c44a4b8ef6 images_004.tar.gz_


_1861f3cd0ef7734df8104f2b0309023b images_005.tar.gz_


_456b53a8b351afd92a35bc41444c58c8 images_006.tar.gz_


_1075121ea20a137b87f290d6a4a5965e images_007.tar.gz_


_b61f34cec3aa69f295fbb593cbd9d443 images_008.tar.gz_


_442a3caa61ae9b64e61c561294d1e183 images_009.tar.gz_


_09ec81c4c31e32858ad8cf965c494b74 images_010.tar.gz_


_499aefc67207a5a97692424cf5dbeed5 images_011.tar.gz_


_dc9fda1757c2de0032b63347a7d2895c images_012.tar.gz_


Q08: General concerns about the image label accuracy.


_A: There are several things about the published image labels that we want to clarify:_


_1._ _Different terms and phrases might be used for the same finding: The image labels are mined from_

_the radiology reports using NLP techniques. Those disease keywords are purely extracted from_
_the reports. The radiologists often described the findings and impressions by using their own_
_preferred terms and phrases for each particular disease pattern or a group of patterns, where the_
_chance of using all possible terms in the description is small._
_2._ _Which terms should be used: We understand it is hard if not impossible to distinguish certain_

_pathologies solely based on the findings in the images. However, other information from multiple_
_sources may be also available to the radiologists (e.g. reason for exam, patients’ previous studies_
_and other clinical information) when he/she reads the study. The diagnostic terms used in the_
_report (like ‘pneumonia’) come from a decision based on all of the available information, not just_
_the imaging findings._
_3._ _Entity extraction using NLP is not perfect: we try to maximize the recall of finding accurate_

_disease findings by eliminating all possible negations and uncertainties of disease mentions._
_Terms like ‘It is hard to exclude …’ will be treated as uncertainty cases and then the image will_
_be labeled as ‘No finding’._
_4._ _‘No finding’ is not equal to ‘normal’. Images labeled with ‘No finding’ could contain disease_

_patterns other than the listed 14 or uncertain findings within the 14 categories._
5. _We encourage others to share their own labels, ideally from a group of radiologists so that_

_observer variability can also be assessed. The published image labels are a first step at enabling_
_other researchers to start looking at the problem of ‘automated reading a chest X-ray’ on a very_
_large dataset, and the labels are meant to be improved by the community._


Q09: Will you publish the data split files?


_A: Yes, two split files (train_val_list.txt and test_list.txt) are now provided and ready for downloading._
_Images in the ChestX-ray dataset are divided into two sets on the patient level. All studies from the same_
_patient will only appear in either training/validation or testing set. This data split is adopted to generate_
_the classification and localization results reported in our latest arxiv paper_
_(ARXIV_V5_CHESTXRAY.pdf)._




