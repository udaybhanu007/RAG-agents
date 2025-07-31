# PDF Content Extraction Report

## Document Text

# **ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on** **Weakly-Supervised Classification and Localization of Common Thorax Diseases**

Xiaosong Wang [1], Yifan Peng [2], Le Lu [1], Zhiyong Lu [2], Mohammadhadi Bagheri [1], Ronald M. Summers [1]

1 Department of Radiology and Imaging Sciences, Clinical Center,
2 National Center for Biotechnology Information, National Library of Medicine,
National Institutes of Health, Bethesda, MD 20892


_{_ xiaosong.wang,yifan.peng,le.lu,luzh,mohammad.bagheri,rms _}_ @nih.gov


**Abstract**



_The chest X-ray is one of the most commonly accessi-_
_ble radiological examinations for screening and diagnosis_
_of many lung diseases._ _A tremendous number of X-ray_
_imaging studies accompanied by radiological reports are_
_accumulated and stored in many modern hospitals’ Pic-_
_ture Archiving and Communication Systems (PACS). On_
_the other side, it is still an open question how this type_
_of hospital-size knowledge database containing invaluable_
_imaging informatics (i.e., loosely labeled) can be used to fa-_
_cilitate the data-hungry deep learning paradigms in build-_
_ing truly large-scale high precision computer-aided diagno-_
_sis (CAD) systems._
_In this paper, we present a new chest X-ray database,_
_namely “ChestX-ray8”, which comprises 108,948 frontal-_
_view X-ray images of 32,717 unique patients with the text-_
_mined eight disease image labels (where each image can_
_have multi-labels), from the associated radiological reports_
_using natural language processing. Importantly, we demon-_
_strate that these commonly occurring thoracic diseases can_
_be detected and even spatially-located via a unified weakly-_
_supervised multi-label image classification and disease lo-_
_calization framework, which is validated using our proposed_
_dataset. Although the initial quantitative results are promis-_
_ing as reported, deep convolutional neural network based_
_“reading chest X-rays” (i.e., recognizing and locating the_
_common disease patterns trained with only image-level la-_
_bels) remains a strenuous task for fully-automated high pre-_
_cision CAD systems._


**1** **Introduction**

The rapid and tremendous progress has been evidenced
in a range of computer vision problems via deep learning
and large-scale annotated image datasets [26, 38, 13, 28].
Drastically improved quantitative performances in object
recognition, detection and segmentation are demonstrated in



Figure 1. _Eight common thoracic diseases observed in chest X-rays_
_that validate a challenging task of fully-automated diagnosis._


comparison to previous shallow methodologies built upon
hand-crafted image features. Deep neural network representations further make the joint language and vision
learning tasks more feasible to solve, in image captioning

[49, 24, 33, 48, 23], visual question answering [2, 46, 51, 55]
and knowledge-guided transfer learning [4, 34], and so
on. However, the intriguing and strongly observable performance gaps of the current state-of-the-art object detection and segmentation methods, evaluated between using
PASCAL VOC [13] and employing Microsoft (MS) COCO

[28], demonstrate that there is still significant room for performance improvement when underlying challenges (represented by different datasets) become greater. For example, MS COCO is composed of 80 object categories from
200k images, with 1.2M instances (350k are people) where
every instance is segmented and many instances are small
objects. Comparing to PASCAL VOC of only 20 classes
and 11,530 images containing 27,450 annotated objects with
bounding-boxes (BBox), the top competing object detection
approaches achieve in 0.413 in MS COCO versus 0.884 in
PASCAL VOC under mean Average Precision (mAP).
Deep learning yields similar rises in performance in the
medical image analysis domain for object (often human
anatomical or pathological structures in radiology imaging)



1


detection and segmentation tasks. Recent notable work includes (but do not limit to) an overview review on the future
promise of deep learning [14] and a collection of important
medical applications on lymph node and interstitial lung disease detection and classification [37, 43]; cerebral microbleed detection [11]; pulmonary nodule detection in CT images [40]; automated pancreas segmentation [36]; cell image segmentation and tracking [35], predicting spinal radiological scores [21] and extensions of multi-modal imaging
segmentation [30, 16]. The main limitation is that all proposed methods are evaluated on some small-to-middle scale
problems of (at most) several hundred patients. It remains
unclear how well the current deep learning techniques will
scale up to tens of thousands of patient studies.


In the era of deep learning in computer vision, research efforts on building various annotated image datasets

[38, 13, 28, 2, 33, 55, 23, 25] with different characteristics
play indispensably important roles on the better definition
of the forthcoming problems, challenges and subsequently
possible technological progresses. Particularly, here we focus on the relationship and joint learning of image (chest Xrays) and text (X-ray reports). The previous representative
image caption generation work [49, 24] utilize Flickr8K,
Flickr30K [53] and MS COCO [28] datasets that hold 8,000,
31,000 and 123,000 images respectively and every image is
annotated by five sentences via Amazon Mechanical Turk
(AMT). The text generally describes annotator’s attention
of objects and activity occurring on an image in a straightforward manner. Region-level ImageNet pre-trained convolutional neural networks (CNN) based detectors are used
to parse an input image and output a list of attributes or
“visually-grounded high-level concepts” (including objects,
actions, scenes and so on) in [24, 51]. Visual question answering (VQA) requires more detailed parsing and complex
reasoning on the image contents to answer the paired natural
language questions. A new dataset containing 250k natural
images, 760k questions and 10M text answers [2] is provided to address this new challenge. Additionally, databases
such as “Flickr30k Entities” [33], “Visual7W” [55] and “Visual Genome” [25, 23] (as detailed as 94,000 images and
4,100,000 region-grounded captions) are introduced to construct and learn the spatially-dense and increasingly difficult semantic links between textual descriptions and image
regions through the object-level grounding.


Though one could argue that the high-level analogy exists between image caption generation, visual question answering and imaging based disease diagnosis [42, 41], there
are three factors making truly large-scale medical image
based diagnosis (e.g., involving tens of thousands of patients) tremendously more formidable. **1**, Generic, openended image-level anatomy and pathology labels cannot be
obtained through crowd-sourcing, such as AMT, which is
prohibitively implausible for non-medically trained annota


tors. Therefore we exploit to mine the per-image (possibly multiple) common thoracic pathology labels from the
image-attached chest X-ray radiological reports using Natural Language Processing (NLP) techniques. Radiologists
tend to write more abstract and complex logical reasoning
sentences than the plain describing texts in [53, 28]. **2**, The
spatial dimensions of an chest X-ray are usually 2000 _×_ 3000
pixels. Local pathological image regions can show hugely
varying sizes or extents but often very small comparing to
the full image scale. Fig. 1 shows eight illustrative examples
and the actual pathological findings are often significantly
smaller (thus harder to detect). Fully dense annotation of
region-level bounding boxes (for grounding the pathological findings) would normally be needed in computer vision
datasets [33, 55, 25] but may be completely nonviable for
the time being. Consequently, we formulate and verify a
weakly-supervised multi-label image classification and disease localization framework to address this difficulty. **3**,
So far, all image captioning and VQA techniques in computer vision strongly depend on the ImageNet pre-trained
deep CNN models which already perform very well in a
large number of object classes and serves a good baseline
for further model fine-tuning. However, this situation does
not apply to the medical image diagnosis domain. Thus we
have to learn the deep image recognition and localization
models while constructing the weakly-labeled medical image database.
To tackle these issues, we propose a new chest X-ray
database, namely “ChestX-ray8”, which comprises 108,948
frontal-view X-ray images of 32,717 (collected from the
year of 1992 to 2015) unique patients with the text-mined
eight common disease labels, mined from the text radiological reports via NLP techniques. In particular, we
demonstrate that these commonly occurred thoracic diseases can be detected and even spatially-located via a unified weakly-supervised multi-label image classification and
disease localization formulation. Our initial quantitative results are promising. However developing fully-automated
deep learning based “reading chest X-rays” systems is still
an arduous journey to be exploited. Details of accessing the
ChestX-ray8 dataset can be found via the website [1] .

**1.1** **Related Work**

There have been recent efforts on creating openly available annotated medical image databases [50, 52, 37, 36]
with the studied patient numbers ranging from a few hundreds to two thousands. Particularly for chest X-rays, the
largest public dataset is OpenI [1] that contains 3,955 radiology reports from the Indiana Network for Patient Care
and 7,470 associated chest x-rays from the hospitals picture
archiving and communication system (PACS). This database
is utilized in [42] as a problem of caption generation but


1
[https://nihcc.app.box.com/v/ChestXray-NIHCC,](https://nihcc.app.box.com/v/ChestXray-NIHCC)
[more details: https://www.cc.nih.gov/drd/summers.html](https://www.cc.nih.gov/drd/summers.html)


no quantitative disease detection results are reported. Our
newly proposed chest X-ray database is at least one order
of magnitude larger than OpenI [1] (Refer to Table 1). To
achieve the better clinical relevance, we focus to exploit
the quantitative performance on weakly-supervised multilabel image classification and disease localization of common thoracic diseases, in analogy to the intermediate step
of “detecting attributes” in [51] or “visual grounding” for

[33, 55, 23].


**2** **Construction of Hospital-scale Chest X-ray**
**Database**

In this section, we describe the approach for building a hospital-scale chest X-ray image database, namely
“ChestX-ray8”, mined from our institute’s PACS system.
First, we short-list eight common thoracic pathology keywords that are frequently observed and diagnosed, i.e., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia and Pneumathorax (Fig. 1), based on radiologists’ feedback. Given those 8 text keywords, we search
the PACS system to pull out all the related radiological reports (together with images) as our target corpus. A variety of Natural Language Processing (NLP) techniques are
adopted for detecting the pathology keywords and removal
of negation and uncertainty. Each radiological report will
be either linked with one or more keywords or marked
with ’Normal’ as the background category. As a result, the
ChestX-ray8 database is composed of 108,948 frontal-view
X-ray images (from 32,717 patients) and each image is labeled with one or multiple pathology keywords or “Normal”
otherwise. Fig. 2 illustrates the correlation of the resulted
keywords. It reveals some connections between different
pathologies, which agree with radiologists’ domain knowledge, e.g., Infiltration is often associated with Atelectasis
and Effusion. To some extend, this is similar with understanding the interactions and relationships among objects or
concepts in natural images [25].
**2.1** **Labeling Disease Names by Text Mining**
Overall, our approach produces labels using the reports
in two passes. In the first iteration, we detected all the disease concept in the corpus. The main body of each chest
X-ray report is generally structured as “Comparison”, “Indication”, “Findings”, and “Impression” sections. Here, we
focus on detecting disease concepts in the Findings and Impression sections. If a report contains neither of these two
sections, the full-length report will then be considered. In
the second pass, we code the reports as “Normal” if they
do not contain any diseases (not limited to 8 predefined
pathologies).
**Pathology Detection:** We mine the radiology reports
for disease concepts using two tools, DNorm [27] and
MetaMap [3]. DNorm is a machine learning method for
disease recognition and normalization. It maps every mention of keywords in a report to a unique concept ID in the



Figure 2. _The circular diagram shows the proportions of images_
_with multi-labels in each of 8 pathology classes and the labels’_

_co-occurrence statistics._

Systematized Nomenclature of Medicine Clinical Terms
(or SNOMED-CT), which is a standardized vocabulary of
clinical terminology for the electronic exchange of clinical
health information.

MetaMap is another prominent tool to detect bioconcepts from the biomedical text corpus. Different from
DNorm, it is an ontology-based approach for the detection of Unified Medical Language System _⃝_ [R] (UMLS _⃝_ R )
Metathesaurus. In this work, we only consider the semantic types of Diseases or Syndromes and Findings (namely
‘dsyn’ and ‘fndg’ respectively). To maximize the recall
of our automatic disease detection, we merge the results
of DNorm and MetaMap. Table 1 (in the supplementary material) shows the corresponding SNOMED-CT concepts that are relevant to the eight target diseases (these
mappings are developed by searching the disease names in
the UMLS _⃝_ [R] terminology service 2, and verified by a boardcertified radiologist.
**Negation and Uncertainty:** The disease detection algorithm locates every keyword mentioned in the radiology report no matter if it is truly present or negated. To eliminate
the noisy labeling, we need to rule out those negated pathological statements and, more importantly, uncertain mentions of findings and diseases, e.g., “suggesting obstructive
lung disease”.
Although many text processing systems (such as [6]) can
handle the negation/uncertainty detection problem, most of
them exploit regular expressions on the text directly. One
of the disadvantages to use regular expressions for nega

2 [https://uts.nlm.nih.gov/metathesaurus.html](https://uts.nlm.nih.gov/metathesaurus.html)


Figure 3. _The dependency graph of text: “clear of focal airspace_
_disease, pneumothorax, or pleural effusion”._


tion/uncertainty detection is that they cannot capture various syntactic constructions for multiple subjects. For example, in the phrase of “clear of A and B”, the regular expression can capture “A” as a negation but not “B”, particularly
when both “A” and “B” are long and complex noun phrases
(“clear of focal airspace disease, pneumothorax, or pleural
effusion” in Fig. 3).
To overcome this complication, we hand-craft a number
of novel rules of negation/uncertainty defined on the syntactic level in this work. More specifically, we utilize the
syntactic dependency information because it is close to the
semantic relationship between words and thus has become
prevalent in biomedical text processing. We defined our
rules on the dependency graph, by utilizing the dependency
label and direction information between words.

As the first step of preprocessing, we split and tokenize
the reports into sentences using NLTK [5]. Next we parse
each sentence by the Bllip parser [7] using David McCloskys biomedical model [29]. The syntactic dependencies are then obtained from “CCProcessed” dependencies
output by applying Stanford dependencies converter [8] on
the parse tree. The “CCProcessed” representation propagates conjunct dependencies thus simplifies coordinations.
As a result, we can use fewer rules to match more complex constructions. For an example as shown in Fig. 3, we
could use “clear _→_ prep of _→_ DISEASE” to detect three
negations from the text _⟨_ neg, focal airspace disease _⟩_, _⟨_ neg,
pneumothorax _⟩_, and _⟨_ neg, pleural effusion _⟩_ .
Furthermore, we label a radiology report as “normal” if
it meets one of the following criteria:


_•_ If there is no disease detected in the report. Note that
here we not only consider 8 diseases of interest in this
paper, but all diseases detected in the reports.


_•_ If the report contains text-mined concepts of “normal”
or “normal size” (CUIs C0205307 and C0332506 in
the SNOMED-CT concepts respectively).


**2.2** **Quality Control on Disease Labeling**
To validate our method, we perform the following experiments. Given the fact that no gold-standard labels exist for
our dataset, we resort to some existing annotated corpora as
an alternative. Using the OpenI API [1], we retrieve a total
of 3,851 unique radiology reports where each OpenI report
is assigned with its key findings/disease names by human
annotators [9]. Given our focus on the eight diseases, a subset of OpenI reports and their human annotations are used as



|Item #|OpenI|Ov.|ChestX-ray8|Ov.|
|---|---|---|---|---|
|Report<br>Annotations<br>Atelectasis<br>Cardiomegaly<br>Effusion<br>Inﬁltration<br>Mass<br>Nodule<br>Pneumonia<br>Pneumothorax<br>Normal|2,435<br>2,435<br>315<br>345<br>153<br>60<br>15<br>106<br>40<br>22<br>1,379|-<br>-<br>122<br>100<br>94<br>45<br>4<br>18<br>15<br>11<br>0|108,948<br>-<br>5,789<br>1,010<br>6,331<br>10,317<br>6,046<br>1,971<br>1,062<br>2,793<br>84,312|-<br>-<br>3,286<br>475<br>4,017<br>4,698<br>3,432<br>1,041<br>703<br>1,403<br>0|


Table 1. _Total number (#) and # of Overlap (Ov.) of the corpus in_
_both OpenI and ChestX-ray8 datasets._


MetaMap Our Method
Disease
P / R / F P / R / F

Atelectasis 0.95 / 0.95 / 0.95 0.99 / 0.85 / 0.91

Cardiomegaly 0.99 / 0.83 / 0.90 1.00 / 0.79 / 0.88
Effusion 0.74 / 0.90 / 0.81 0.93 / 0.82 / 0.87

Infiltration 0.25 / 0.98 / 0.39 0.74 / 0.87 / 0.80
Mass 0.59 / 0.67 / 0.62 0.75 / 0.40 / 0.52

Nodule 0.95 / 0.65 / 0.77 0.96 / 0.62 / 0.75

Normal 0.93 / 0.90 / 0.91 0.87 / 0.99 / 0.93

Pneumonia 0.58 / 0.93 / 0.71 0.66 / 0.93 / 0.77

Pneumothorax 0.32 / 0.82 / 0.46 0.90 / 0.82 / 0.86

_Total_ 0.84 / 0.88 / 0.86 0.90 / 0.91 / 0.90


Table 2. _Evaluation of image labeling results on OpenI dataset._
_Performance is reported using P, R, F1-score._


the gold standard for evaluating our method. Table 1 summarizes the statistics of the subset of OpenI [1, 20] reports.
Table 2 shows the results of our method using OpenI, measured in precision (P), recall (R), and F1-score. Higher precision of 0.90, higher recall of 0.91, and higher F1-score
of 0.90 are achieved compared to the existing MetaMap approach (with NegEx enabled). For all diseases, our method
obtains higher precisions, particularly in “pneumothorax”
(0.90 vs. 0.32) and “infiltration” (0.74 vs. 0.25). This indicates that the usage of negation and uncertainty detection
on syntactic level successfully removes false positive cases.
More importantly, the higher precisions meet our expectation to generate a Chest X-ray corpus with accurate semantic labels, to lay a solid foundation for the later processes.


**2.3** **Processing Chest X-ray Images**
Comparing to the popular ImageNet classification problem, significantly smaller spatial extents of many diseases
inside the typical X-ray image dimensions of 3000 _×_ 2000
pixels impose challenges in both the capacity of computing hardware and the design of deep learning paradigm. In
ChestX-ray8, X-rays images are directly extracted from the
DICOM file and resized as 1024 _×_ 1024 bitmap images without significantly losing the detail contents, compared with


image sizes of 512 _×_ 512 in OpenI dataset. Their intensity
ranges are rescaled using the default window settings stored
in the DICOM header files.

**2.4** **Bounding Box for Pathologies**

As part of the ChestX-ray8 database, a small number
of images with pathology are provided with hand labeled
bounding boxes (B-Boxes), which can be used as the ground
truth to evaluate the disease localization performance. Furthermore, it could also be adopted for one/low-shot learning setup [15], in which only one or several samples are
needed to initialize the learning and the system will then
evolve by itself with more unlabeled data. We leave this as
future work.

In our labeling process, we first select 200 instances for
each pathology (1,600 instances total), consisting of 983
images. Given an image and a disease keyword, a boardcertified radiologist identified only the corresponding disease instance in the image and labeled it with a B-Box. The
B-Box is then outputted as an XML file. If one image contains multiple disease instances, each disease instance is labeled separately and stored into individual XML files. As
an application of the proposed ChestX-ray8 database and
benchmarking, we will demonstrate the detection and localization of thoracic diseases in the following.


**3** **Common Thoracic Disease Detection and**

**Localization**

Reading and diagnosing Chest X-ray images may be an
entry-level task for radiologists but, in fact it is a complex
reasoning problem which often requires careful observation
and good knowledge of anatomical principles, physiology
and pathology. Such factors increase the difficulty of developing a consistent and automated technique for reading
chest X-ray images while simultaneously considering all
common thoracic diseases.

As the main application of ChestX-ray8 dataset, we
present a unified weakly-supervised multi-label image classification and pathology localization framework, which can
detect the presence of multiple pathologies and subsequently generate bounding boxes around the corresponding
pathologies. In details, we tailor Deep Convolutional Neural
Network (DCNN) architectures for weakly-supervised object localization, by considering large image capacity, various multi-label CNN losses and different pooling strategies.

**3.1** **Unified DCNN Framework**

Our goal is to first detect if one or multiple pathologies
are presented in each X-ray image and later we can locate them using the activation and weights extracted from
the network. We tackle this problem by training a multilabel DCNN classification model. Fig. 4 illustrates the
DCNN architecture we adapted, with similarity to several previous weakly-supervised object localization methods [31, 54, 12, 19]. As shown in Fig. 4, we perform



the network surgery on the pre-trained models (using ImageNet [10, 39]), e.g., AlexNet [26], GoogLeNet [45],
VGGNet-16 [44] and ResNet-50 [17], by leaving out the
fully-connected layers and the final classification layers. Instead we insert a transition layer, a global pooling layer, a
prediction layer and a loss layer in the end (after the last convolutional layer). In a similar fashion as described in [54],
a combination of deep activations from transition layer (a
set of spatial image features) and the weights of prediction
inner-product layer (trained feature weighting) can enable
us to find the plausible spatial locations of diseases.


**Multi-label Setup:** There are several options of imagelabel representation and the choices of multi-label classification loss functions. Here, we define a 8-dimensional
label vector **y** = [ _y_ 1 _, ..., y_ _c_ _, ..., y_ _C_ ] _, y_ _c_ _∈{_ 0 _,_ 1 _}, C_ = 8
for each image. _y_ _c_ indicates the presence with respect to
according pathology in the image while a all-zero vector

[0 _,_ 0 _,_ 0 _,_ 0 _,_ 0 _,_ 0 _,_ 0 _,_ 0] represents the status of “Normal” (no
pathology is found in the scope of any of 8 disease categories as listed). This definition transits the multi-label classification problem into a regression-like loss setting.


**Transition Layer:** Due to the large variety of pre-trained
DCNN architectures we adopt, a transition layer is usually required to transform the activations from previous layers into a uniform dimension of output, _S × S × D, S ∈_
_{_ 8 _,_ 16 _,_ 32 _}_ . _D_ represents the dimension of features at spatial location ( _i, j_ ) _, i, j ∈{_ 1 _, ..., S}_, which can be varied in
different model settings, e.g., _D_ = 1024 for GoogLeNet and
_D_ = 2048 for ResNet. The transition layer helps pass down
the weights from pre-trained DCNN models in a standard
form, which is critical for using this layers’ activations to
further generate the heatmap in pathology localization step.


**Multi-label Classification Loss Layer:** We first experiment 3 standard loss functions for the regression task instead
of using the softmax loss for traditional multi-class classification model, i.e., Hinge Loss (HL), Euclidean Loss (EL)
and Cross Entropy Loss (CEL). However, we find that the
model has difficulty learning positive instances (images with
pathologies) and the image labels are rather sparse, meaning there are extensively more ‘0’s than ‘1’s. This is due to
our one-hot-like image labeling strategy and the unbalanced
numbers of pathology and “Normal” classes. Therefore, we
introduce the positive/negative balancing factor _β_ _P_ _, β_ _N_ to
enforce the learning of positive examples. For example, the
weighted CEL (W-CEL) is defined as follows,
_L_ _W_ – _CEL_ ( _f_ ( _⃗x_ ) _, ⃗y_ ) =



where _β_ _P_ is set to _|P |_ while _β_ _N_ is set to _|N_ _|_ . _|P_ _|_

and _|N_ _|_ are the total number of ‘1’s and ‘0’s in a batch of
image labels.



_β_ _P_ �



� _−_ ln( _f_ ( _x_ _c_ )) + _β_ _N_ �

_y_ _c_ =1 _y_ _c_ =0



� _−_ ln(1 _−_ _f_ ( _x_ _c_ )) _,_ (1)

_y_ _c_ =0



where _β_ _P_ is set to _[|][P]_ _[|]_ [+] _[|][N]_ _[|]_




_[|]_ _|_ [+] _P |_ _[|][N]_ _[|]_ while _β_ _N_ is set to _[|][P]_ _|_ _[|]_ _N_ [+] _[|]_ _|_ _[N]_ _[|]_


Figure 4. _The overall flow-chart of our unified DCNN framework and disease localization process._



**3.2** **Weakly-Supervised Pathology Localization**


**Global Pooling Layer and Prediction Layer:** In our
multi-label image classification network, the global pooling and the predication layer are designed not only to be
part of the DCNN for classification but also to generate the
likelihood map of pathologies, namely a heatmap. The location with a peak in the heatmap generally corresponds to
the presence of disease pattern with a high probability. The
upper part of Fig. 4 demonstrates the process of producing
this heatmap. By performing a global pooling after the transition layer, the weights learned in the prediction layer can
function as the weights of spatial maps from the transition
layer. Therefore, we can produce weighted spatial activation
maps for each disease class (with a size of _S × S × C_ ) by
multiplying the activation from transition layer (with a size
of _S × S × D_ ) and the weights of prediction layer (with a
size of _D × C_ ).


The pooling layer plays an important role that chooses
what information to be passed down. Besides the conventional max pooling and average pooling, we also utilize the
Log-Sum-Exp (LSE) pooling proposed in [32]. The LSE
pooled value _x_ _p_ is defined as



_r_, the pooled value ranges from the maximum in **S** (when
_r →∞_ ) to average ( _r →_ 0). It serves as an adjustable option between max pooling and average pooling. Since the
LSE function suffers from overflow/underflow problems,
the following equivalent is used while implementing the
LSE pooling layer in our own DCNN architecture,



_S_ _[·]_ �



� _exp_ ( _r ·_ ( _x_ _ij_ _−_ _x_ _[∗]_ )

( _i,j_ ) _∈_ **S**



_x_ _p_ = _x_ _[∗]_ + [1] _r_ _[·]_ [ log]







 _S_ [1]







 _,_ (3)



_S_ _[·]_ �



� _exp_ ( _r · x_ _ij_ )

( _i,j_ ) _∈_ **S**



_x_ _p_ = [1] _r_ _[·]_ [ log]







 _S_ [1]







 _,_ (2)



where _x_ _[∗]_ = _max{|x_ _ij_ _|,_ ( _i, j_ ) _∈_ **S** _}_ .
**Bounding Box Generation:** The heatmap produced
from our multi-label classification framework indicates the
approximate spatial location of one particular thoracic disease class each time. Due to the simplicity of intensity distributions in these resulting heatmaps, applying an ad-hoc
thresholding based B-Box generation method for this task is
found to be sufficient. The intensities in heatmaps are first
normalized to [0 _,_ 255] and then thresholded by _{_ 60 _,_ 180 _}_ individually. Finally, B-Boxes are generated to cover the isolated regions in the resulting binary maps.


**4** **Experiments**
**Data:** We evaluate and validate the unified disease classification and localization framework using the proposed
ChestX-ray8 database. In total, 108,948 frontal-view X-ray
images are in the database, of which 24,636 images contain
one or more pathologies. The remaining 84,312 images are
normal cases. For the pathology classification and localization task, we randomly shuffled the entire dataset into three



where _x_ _ij_ is the activation value at ( _i, j_ ), ( _i, j_ ) is one location in the pooling region **S**, and _S_ = _s × s_ is the total number of locations in **S** . By controlling the hyper-parameter


subgroups for CNN fine-tuning via Stochastic Gradient Descent (SGD): i.e. training (70%), validation (10%) and testing (20%). We only report the 8 thoracic disease recognition
performance on the testing set in our experiments. Furthermore, for the 983 images with 1,600 annotated B-Boxes of
pathologies, these boxes are only used as the ground truth to
evaluate the disease localization accuracy in testing (not for
training purpose).

**CNN Setting:** Our multi-label CNN architecture is implemented using Caffe framework [22]. The ImageNet
pre-trained models, i.e., AlexNet [26], GoogLeNet [45],
VGGNet-16 [44] and ResNet-50 [17] are obtained from the
Caffe model zoo. Our unified DCNN takes the weights from
those models and only the transition layers and prediction
layers are trained from scratch.

Due to the large image size and the limit of GPU memory, it is necessary to reduce the image _batch_ ~~_s_~~ _ize_ to load
the entire model and keep activations in GPU while we increase the _iter_ ~~_s_~~ _ize_ to accumulate the gradients for more iterations. The combination of both may vary in different
CNN models but we set _batch_ _size × iter_ ~~_s_~~ _ize_ = 80 as

a constant. Furthermore, the total training iterations are customized for different CNN models to prevent over-fitting.
More complex models like ResNet-50 actually take less iterations (e.g., 10000 iterations) to reach the convergence.
The DCNN models are trained using a Dev-Box linux server
with 4 Titan X GPUs.

**Multi-label Disease Classification:** Fig. 5 demonstrates
the multi-label classification ROC curves on 8 pathology
classes by initializing the DCNN framework with 4 different pre-trained models of AlexNet, GoogLeNet, VGG
and ResNet-50. The corresponding Area-Under-Curve
(AUC) values are given in Table 4. The quantitative performance varies greatly, in which the model based on
ResNet-50 achieves the best results. The “Cardiomegaly”
(AUC=0.8141) and “Pneumothorax” (AUC=0.7891) classes
are consistently well-recognized compared to other groups
while the detection ratios can be relatively lower for
pathologies which contain small objects, e.g., “Mass”
(AUC=0.5609) and “Nodule” classes. Mass is difficult to
detect due to its huge within-class appearance variation. The
lower performance on “Pneumonia” (AUC=0.6333) is probably because of lack of total instances in our patient population (less than 1% X-rays labeled as Pneumonia). This
finding is consistent with the comparison on object detection performance, degrading from PASCAL VOC [13] to
MS COCO [28] where many small annotated objects appear.

Next, we examine the influence of different pooling
strategies when using ResNet-50 to initialize the DCNN
framework. As discussed above, three types of pooling
schemes are experimented: average looping, LSE pooling
and max pooling. The hyper-parameter _r_ in LSE pooling varies in _{_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _,_ 5 _,_ 8 _,_ 10 _,_ 12 _}_ . As illustrated in Fig.



Figure 5. _A comparison of multi-label classification performance_
_with different model initializations._


6, average pooling and max pooling achieve approximately
equivalent performance in this classification task. The performance of LSE pooling start declining first when _r_ starts
increasing and reach the bottom when _r_ = 5. Then it
reaches the overall best performance around _r_ = 10. LSE
pooling behaves like a weighed pooling method or a transition scheme between average and max pooling under different _r_ values. Overall, LSE pooling ( _r_ = 10) reports the
best performance (consistently higher than mean and max
pooling).


Figure 6. _A comparison of multi-label classification performance_
_with different pooling strategies._


Last, we demonstrate the performance improvement by
using the positive/negative instances balanced loss functions
(Eq. 1). As shown in Table 4, the weighted loss (WCEL) provides better overall performance than CEL, especially for those classes with relative fewer positive instances,
e.g. AUC for “Cardiomegaly” is increased from 0.7262 to
0.8141 and from 0.5164 to 0.6333 for “Pneumonia”.


**Disease Localization:** Leveraging the fine-tuned DCNN


|Setting|Atelectasis|Cardiomegaly|Effusion|Infiltration|Mass|Nodule|Pneumonia|Pneumothorax|
|---|---|---|---|---|---|---|---|---|
|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|Initialization with different pre-trained models|
|**AlexNet**|0.6458|0.6925|0.6642|0.6041|**0.5644**|0.6487|0.5493|0.7425|
|**GoogLeNet**|0.6307|0.7056|0.6876|0.6088|0.5363|0.5579|0.5990|0.7824|
|**VGGNet-16**|0.6281|0.7084|0.6502|0.5896|0.5103|0.6556|0.5100|0.7516|
|**ResNet-50**|**0.7069**|**0.8141**|**0.7362**|**0.6128**|0.5609|**0.7164**|**0.6333**|**0.7891**|
|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|Different multi-label loss functions|
|**CEL**|0.7064|0.7262|0.7351|0.6084|0.5530|0.6545|0.5164|0.7665|
|**W-CEL**|0.7069|0.8141|0.7362|0.6128|0.5609|0.7164|0.6333|0.7891|


Table 3. _AUCs of ROC curves for multi-label classification in different DCNN model setting._

|T(IoBB)|Atelectasis|Cardiomegaly|Effusion|Infiltration|Mass|Nodule|Pneumonia|Pneumothorax|
|---|---|---|---|---|---|---|---|---|
|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|
|**Acc.**|0.7277|0.9931|0.7124|0.7886|0.4352|0.1645|0.7500|0.4591|
|**AFP**|0.8323|0.3506|0.7998|0.5589|0.6423|0.6047|0.9055|0.4776|
|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|
|**Acc.**|0.5500|0.9794|0.5424|0.5772|0.2823|0.0506|0.5583|0.3469|
|**AFP**|0.9167|0.4553|0.8598|0.6077|0.6707|0.6158|0.9614|0.5000|
|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|
|**Acc.**|0.2833|0.8767|0.3333|0.4227|0.1411|0.0126|0.3833|0.1836|
|**AFP**|1.0203|0.5630|0.9268|0.6585|0.6941|0.6189|1.0132|0.5285|
|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|
|**Acc.**|0.1666|0.7260|0.2418|0.3252|0.1176|0.0126|0.2583|0.1020|
|**AFP**|1.0619|0.6616|0.9603|0.6921|0.7043|0.6199|1.0569|0.5396|
|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|
|**Acc.**|0.1333|0.6849|0.2091|0.2520|0.0588|0.0126|0.2416|0.0816|
|**AFP**|1.0752|0.7226|0.9797|0.7124|0.7144|0.6199|1.0732|0.5437|



Table 4. _Pathology localization accuracy and average false positive number for 8 disease classes._



models for multi-label disease classification, we can calculate the disease heatmaps using the activations of the
transition layer and the weights from the prediction layer,
and even generate the B-Boxes for each pathology candidate. The computed bounding boxes are evaluated against
the hand annotated ground truth (GT) boxes (included in
ChestX-ray8). Although the total number of B-Box annotations (1,600 instances) is relatively small compared to the
entire dataset, it may be still sufficient to get a reasonable
estimate on how the proposed framework performs on the
weakly-supervised disease localization task. To examine the
accuracy of computerized B-Boxes versus the GT B-Boxes,
two types of measurement are used, i.e, the standard Intersection over Union ratio (IoU) or the Intersection over the
detected B-Box area ratio (IoBB) (similar to Area of Precision or Purity). Due to the relatively low spatial resolution
of heatmaps (32 _×_ 32) in contrast to the original image dimensions (1024 _×_ 1024), the computed B-Boxes are often
larger than the according GT B-Boxes. Therefore, we define
a correct localization by requiring either _IoU > T_ ( _IoU_ )
or _IoBB > T_ ( _IoBB_ ). Refer to the supplementary material for localization performance under varying _T_ ( _IoU_ ).
Table 4 illustrates the localization accuracy (Acc.) and Average False Positive (AFP) number for each disease type,
with _T_ ( _IoBB_ ) _∈{_ 0 _._ 1 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _,_ 0 _._ 9 _}_ . Please refer
to the supplementary material for qualitative exemplary disease localization results for each of 8 pathology classes.



**5** **Conclusion**


Constructing hospital-scale radiology image databases
with computerized diagnostic performance benchmarks has
not been addressed until this work. We attempt to build
a “machine-human annotated” comprehensive chest X-ray
database that presents the realistic clinical and methodological challenges of handling at least tens of thousands of patients (somewhat similar to “ImageNet” in natural images).
We also conduct extensive quantitative performance benchmarking on eight common thoracic pathology classification and weakly-supervised localization using ChestX-ray8
database. The main goal is to initiate future efforts by promoting public datasets in this important domain. Building
truly large-scale, fully-automated high precision medical diagnosis systems remains a strenuous task. ChestX-ray8 can
enable the data-hungry deep neural network paradigms to
create clinically meaningful applications, including common disease pattern mining, disease correlation analysis, automated radiological report generation, etc. For future work,
ChestX-ray8 will be extended to cover more disease classes
and integrated with other clinical information, e.g., followup studies across time and patient history.


**Acknowledgements** This work was supported by the Intramural Research Programs of the NIH Clinical Center and
National Library of Medicine. We thank NVIDIA Corporation for the GPU donation.


**References**


[[1] Open-i: An open access biomedical search engine. https:](https://openi.nlm.nih.gov)
[//openi.nlm.nih.gov. 2, 3, 4](https://openi.nlm.nih.gov)

[2] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, and L. Zitnick. Vqa: Visual question answering. In _ICCV_, 2015. 1, 2

[3] A. R. Aronson and F.-M. Lang. An overview of MetaMap:
historical perspective and recent advances. _Journal of the_
_American Medical Informatics Association_, 17(3):229–236,
may 2010. 3

[4] J. Ba, K. Swersky, S. Fidler, and R. Salakhutdinov. Predicting
deep zero-shot convolutional neural networks using textual
descriptions. In _ICCV_, 2015. 1

[5] S. Bird, E. Klein, and E. Loper. _Natural language processing_
_with Python_ . ”O’Reilly Media, Inc.”, 2009. 4

[6] W. W. Chapman, W. Bridewell, P. Hanbury, G. F. Cooper, and
B. G. Buchanan. A simple algorithm for identifying negated
findings and diseases in discharge summaries. _Journal of_
_Biomedical Informatics_, 34(5):301–310, oct 2001. 3

[7] E. Charniak and M. Johnson. Coarse-to-fine n-best parsing
and MaxEnt discriminative reranking. In _Proceedings of the_
_43rd Annual Meeting on Association for Computational Lin-_
_guistics (ACL)_, pages 173–180, 2005. 4

[8] M.-C. De Marneffe and C. D. Manning. _Stanford typed de-_
_pendencies manual_ . Stanford University, apr 2015. 4

[9] D. Demner-Fushman, M. D. Kohli, M. B. Rosenman, S. E.
Shooshan, L. Rodriguez, S. Antani, G. R. Thoma, and C. J.
McDonald. Preparing a collection of radiology examinations
for distribution and retrieval. _Journal of the American Medi-_
_cal Informatics Association_, 23(2):304–310, July 2015. 4

[10] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei. Imagenet: A large-scale hierarchical image database. In
_Computer Vision and Pattern Recognition_, pages 248–255.
IEEE, 2009. 5

[11] Q. Dou, H. Chen, L. Yu, L. Zhao, J. Qin, D. Wang, V. Mok,
L. Shi, and P. Heng. Automatic detection of cerebral microbleeds from mr images via 3d convolutional neural networks.
_IEEE Trans. Medical Imaging_, 35(5):1182–1195, 2016. 2

[12] T. Durand, N. Thome, and M. Cord. Weldon: Weakly supervised learning of deep convolutional neural networks. _IEEE_
_CVPR_, 2016. 5

[13] M. Everingham, S. M. A. Eslami, L. J. Van Gool,
C. Williams, J. Winn, and A. Zisserman. The pascal visual
object classes challenge: A retrospective. _International Jour-_
_nal of Computer Vision_, pages 111(1): 98–136, 2015. 1, 2,

7

[14] H. Greenspan, B. van Ginneken, and R. M. Summers. Guest
editorial deep learning in medical imaging: Overview and
future promise of an exciting new technique. _IEEE Trans._
_Medical Imaging_, 35(5):1153–1159, 2016. 2

[15] B. Hariharan and R. Girshick. Low-shot visual object recognition. _arXiv preprint arXiv:1606.02819_, 2016. 5

[16] M. Havaei, N. Guizard, N. Chapados, and Y. Bengio. Hemis:
Hetero-modal image segmentation. In _MICCAI_, pages (2):
469–477. Springer, 2016. 2

[17] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. _arXiv preprint arXiv:1512.03385_,
2015. 5, 7




[18] J. Hosang, R. Benenson, P. Doll´ar, and B. Schiele. What
makes for effective detection proposals? _IEEE transactions_
_on pattern analysis and machine intelligence_, 38(4):814–
830, 2016. 11

[19] S. Hwang and H.-E. Kim. Self-transfer learning for weakly
supervised lesion localization. In _MICCAI_, pages (2): 239–
246, 2015. 5

[20] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wng, P.-X. Lu,
and G. Thoma. Two public chest x-ray datasets for computeraided screening of pulmonary diseases. _Quantitative Imaging_
_in Medicine and Surgery_, 4(6), 2014. 4

[21] A. Jamaludin, T. Kadir, and A. Zisserman. Spinenet: Automatically pinpointing classification evidence in spinal mris.
In _MICCAI_ . Springer, 2016. 2

[22] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. _arXiv preprint_
_arXiv:1408.5093_, 2014. 7

[23] J. Johnson, A. Karpathy, and L. Fei-Fei. Densecap: Fully
convolutional localization networks for dense captioning. In
_CVPR_, 2016. 1, 2, 3

[24] A. Karpathy and L. Fei-Fei. Deep visual-semantic alignments
for generating image descriptions. In _CVPR_, 2015. 1, 2

[25] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz,
S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, M. Bernstein,
and L. Fei-Fei. Visual genome: Connecting language and
vision using crowdsourced dense image annotations. 2016.
2, 3

[26] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet
classification with deep convolutional neural networks. In
_Advances in neural information processing systems_, pages
1097–1105, 2012. 1, 5, 7

[27] R. Leaman, R. Khare, and Z. Lu. Challenges in clinical natural language processing for automated disorder normalization. _Journal of Biomedical Informatics_, 57:28–37, 2015. 3

[28] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollr, and L. Zitnick. Microsoft coco: Common
objects in context. _ECCV_, pages (5): 740–755, 2014. 1, 2, 7

[29] D. McClosky. _Any domain parsing: automatic domain adap-_
_tation for natural language parsing_ . Thesis, Department of
Computer Science, Brown University, 2009. 4

[30] P. Moeskops, J. Wolterink, B. van der Velden, K. Gilhuijs,
T. Leiner, M. Viergever, and I. Isgum. Deep learning for
multi-task medical image segmentation in multiple modalities. In _MICCAI_ . Springer, 2016. 2

[31] M. Oquab, L. Bottou, I. Laptev, and J. Sivic. Is object localization for free?-weakly-supervised learning with convolutional neural networks. In _IEEE CVPR_, pages 685–694,
2015. 5

[32] P. O. Pinheiro and R. Collobert. From image-level to pixellevel labeling with convolutional networks. In _Proceedings of_
_the IEEE Conference on Computer Vision and Pattern Recog-_
_nition_, pages 1713–1721, 2015. 6

[33] B. Plummer, L. Wang, C. Cervantes, J. Caicedo, J. Hockenmaier, and S. Lazebnik. Flickr30k entities: Collecting regionto-phrase correspondences for richer image-to-sentence models. In _ICCV_, 2015. 1, 2, 3


[34] R. Qiao, L. Liu, C. Shen, and A. van den Hengel. Less is
more: zero-shot learning from online textual documents with
noise suppression. In _CVPR_, 2016. 1

[35] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In _MIC-_
_CAI_, pages 234–241. Springer, 2015. 2

[36] H. Roth, L. Lu, A. Farag, H.-C. Shin, J. Liu, E. B. Turkbey,
and R. M. Summers. Deeporgan: Multi-level deep convolutional networks for automated pancreas segmentation. In
_MICCAI_, pages 556–564. Springer, 2015. 2

[37] H. R. Roth, L. Lu, A. Seff, K. M. Cherry, J. Hoffman,
S. Wang, J. Liu, E. Turkbey, and R. M. Summers. A new
2.5D representation for lymph node detection using random
sets of deep convolutional neural network observations. In
_MICCAI_, pages 520–527. Springer, 2014. 2

[38] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
A. Berg, and L. Fei-Fei. Imagenet large scale visual recognition challenge. _International Journal of Computer Vision_,
pages 115(3): 211–252, 2015. 1, 2

[39] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al.
Imagenet large scale visual recognition challenge. _Interna-_
_tional Journal of Computer Vision_, 115(3):211–252, 2015. 5

[40] A. Setio, F. Ciompi, G. Litjens, P. Gerke, C. Jacobs, S. van
Riel, M. Wille, M. Naqibullah, C. Snchez, and B. van Ginneken. Pulmonary nodule detection in ct images: False
positive reduction using multi-view convolutional networks.
_IEEE Trans. Medical Imaging_, 35(5):1160–1169, 2016. 2

[41] H. Shin, L. Lu, L. Kim, A. Seff, J. Yao, and R. Summers.
Interleaved text/image deep mining on a large-scale radiology database for automated image interpretation. _Journal of_
_Machine Learning Research_, 17:1–31, 2016. 2

[42] H. Shin, K. Roberts, L. Lu, D. Demner-Fushman, J. Yao, and
R. Summers. Learning to read chest x-rays: Recurrent neural
cascade model for automated image annotation. In _CVPR_,
2016. 2, 11

[43] H. Shin, H. Roth, M. Gao, L. Lu, Z. Xu, I. Nogues, J. Yao,
D. Mollura, and R. Summers. Deep convolutional neural
networks for computer-aided detection: Cnn architectures,
dataset characteristics and transfer learnings. _IEEE Trans._
_Medical Imaging_, 35(5):1285–1298, 2016. 2

[44] K. Simonyan and A. Zisserman. Very deep convolutional
networks for large-scale image recognition. _arXiv preprint_
_arXiv:1409.1556_, 2014. 5, 7

[45] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. In _Proceedings of the IEEE_
_Conference on Computer Vision and Pattern Recognition_,
pages 1–9, 2015. 5, 7

[46] M. Tapaswi, Y. Zhu, R. Stiefelhagen, A. Torralba, R. Urtasun, and S. Fidler. Movieqa: Understanding stories in movies
through question-answering. In _ICCV_, 2015. 1

[47] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W.
Smeulders. Selective search for object recognition. _Inter-_
_national journal of computer vision_, 104(2):154–171, 2013.

11




[48] I. Vendrov, R. Kiros, S. Fidler, and R. Urtasun. Orderembeddings of images and language. In _ICLR_, 2016. 1

[49] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and
tell: A neural image caption generator. In _CVPR_, pages
3156–3164, 2015. 1, 2

[50] H.-J. Wilke, M. Kmin, and J. Urban. Genodisc dataset: The
benefits of multi-disciplinary research on intervertebral disc
degeneration. In _European Spine Journal_, 2016. 2

[51] Q. Wu, P. Wang, C. Shen, A. Dick, and A. van den Hengel.
Ask me anything: free-form visual question answering based
on knowledge from external sources. In _CVPR_, 2016. 1, 2, 3

[52] J. Yao and et al. A multi-center milestone study of clinical
vertebral ct segmentation. In _Computerized Medical Imaging_
_and Graphics_, pages 49(4): 16–28, 2016. 2

[53] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. In _TACL_,
2014. 2

[54] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba.
Learning deep features for discriminative localization. _arXiv_
_preprint arXiv:1512.04150_, 2015. 5

[55] Y. Zhu, O. Groth, M. Bernstein, and L. Fei-Fei. Visual7w:
Grounded question answering in images. In _CVPR_, 2016. 1,
2, 3


**A** **Supplementary Materials**

**A.1** **SNOMED-CT Concepts**


In this work, we only consider the semantic types of Diseases or Syndromes and Findings (namely ‘dsyn’ and ‘fndg’
respectively). Table 5 shows the corresponding SNOMEDCT concepts that are relevant to the target diseases (these
mappings are developed by searching the disease names in
the UMLS _⃝_ [R] terminology service 3, and verified by a boardcertificated radiologist.


**A.2** **Rules of Negation/Uncertainty**


Although many text processing systems can handle the
negation/uncertainty detection problem, most of them exploit regular expressions on the text directly. One of
the disadvantages to use regular expressions for negation/uncertainty detection is that they cannot capture various
syntactic constructions for multiple subjects. For example,
in the phrase of “clear of A and B”, the regular expression
can capture “A” as a negation but not “B”, particularly when
both “A” and “B” are long and complex noun phrases.


3 [https://uts.nlm.nih.gov/metathesaurus.html](https://uts.nlm.nih.gov/metathesaurus.html)


CUI Concept

Atelectasis

C0004144 atelectasis

C0264494 discoid atelectasis

C0264496 focal atelectasis

Cardiomegaly
C0018800 cardiomegaly
Effusion

C0013687 effusion

C0031039 pericardial effusion
C0032227 pleural effusion disorder
C0747635 bilateral pleural effusion
C0747639 loculated pleural effusion
Pneumonia

C0032285 pneumonia
C0577702 basal pneumonia
C0578576 left upper zone pneumonia
C0578577 right middle zone pneumonia
C0585104 left lower zone pneumonia
C0585105 right lower zone pneumonia
C0585106 right upper zone pneumonia
C0747651 recurrent aspiration pneumonia
C1960024 lingular pneumonia
Pneumothorax

C0032326 pneumothorax
C0264557 chronic pneumothorax
C0546333 right pneumothorax
C0546334 left pneumothorax


Table 5. _Sample Target Diseases and their corresponding concept_
_and identifiers (CUIs) in SNOMED-CT._



To overcome this complication, we hand-craft a number of novel rules of negation/uncertainty defined on the
syntactic level in this work. More specifically, we utilize
the syntactic dependency information because it is close to
the semantic relationship between words and thus has become prevalent in biomedical text processing. We defined
our rules on the dependency graph, by utilizing the dependency label and direction information between words. Table 6 shows the rules we defined for negation/uncertainty
detection on the syntactic level.
**A.3** **More Disease Localization Results**

Table 7 illustrates the localization accuracy (Acc.)
and Average False Positive (AFP) number for each disease type, with _IoU_ _> T_ ( _IoU_ ) only and _T_ ( _IoU_ ) _∈_
_{_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _,_ 0 _._ 4 _,_ 0 _._ 5 _,_ 0 _._ 6 _,_ 0 _._ 7 _}_ .
Table 8 to Table 15 illustrate localization results from

each of 8 disease classes together with associated report and
mined disease keywords. The heatmaps overlay on the original images are shown on the right. Correct bounding boxes
(in green), false positives (in red) and the groundtruth (in
blue) are plotted over the original image on the left.
In order to quantitatively demonstrate how informative
those heatmaps are, a simple two-level thresholding based
bounding box generator is adopted here to catch the peaks
in the heatmap and later generated bounding boxes can be
evaluated against the ground truth. Each heatmap will approximately results in 1-3 bounding boxes. We believe the
localization accuracy and AFP (shown in Table 7) could be
further optimized by adopting a more sophisticated bounding box generation method, e.g. selective search [47] or
Edgebox [18]. Nevertheless, we reserve the effort to do so,
since our main goal is not to compute the exact spatial location of disease patterns but just to obtain some instructive
location information for future applications, e.g. automated
radiological report generation. Take the case shown in Table
8 for an example. The peak at the lower part of the left lung
region indicates the presence of “atelectasis”, which confer the statement of “...stable abnormal study including left
basilar infilrate/atelectasis, ...” presented in the impression
section of the associated radiological report. By combining
with other information, e.g. a lung region mask, the heatmap
itself is already more informative than just the presence indication of certain disease in an image as introduced in the
previous works, e.g. [42].


Rule Example

Negation

no _←∗←_ DISEASE No acute pulmonary disease
_∗→_ _prep_ ~~_w_~~ _ithout →_ DISEASE changes without focal airspace disease
clear/free/disappearance _→_ _prep_ ~~_o_~~ _f →_ DISEASE clear of focal airspace disease, pneumothorax, or pleural effusion
_∗→_ _prep_ ~~_w_~~ _ithout →_ evidence _→_ _prep_ ~~_o_~~ _f →_ DISEASE Changes without evidence of acute infiltrate
no _←_ _neg ←_ evidence _→_ _prep_ ~~_o_~~ _f →_ DISEASE No evidence of active disease
Uncertainty

cannot _←_ _md ←_ exclude The aorta is tortuous, and cannot exclude ascending aortic aneurysm
concern _→_ _prep_ ~~_f_~~ _or →∗_ There is raises concern for pneumonia
could be/may be/... which could be due to nodule/lymph node
difficult _→_ _prep_ ~~_t_~~ _o →_ exclude interstitial infiltrates difficult to exclude
may _←_ _md ←_ represent which may represent pleural reaction or small pulmonary nodules
suggesting/suspect/... _→_ _dobj →_ DISEASE Bilateral pulmonary nodules suggesting pulmonary metastases


Table 6. _Rules and corresponding examples for negation and uncertainty detection._

|T(IoU)|Atelectasis|Cardiomegaly|Effusion|Infiltration|Mass|Nodule|Pneumonia|Pneumothorax|
|---|---|---|---|---|---|---|---|---|
|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|T(IoU) = 0.1|
|**Acc.**|0.6888|0.9383|0.6601|0.7073|0.4000|0.1392|0.6333|0.3775|
|**AFP**|0.8943|0.5996|0.8343|0.6250|0.6666|0.6077|1.0203|0.4949|
|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|T(IoU) = 0.2|
|**Acc.**|0.4722|0.6849|0.4509|0.4796|0.2588|0.0506|0.3500|0.2346|
|**AFP**|0.9827|0.7205|0.9096|0.6849|0.6941|0.6158|1.0793|0.5173|
|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|T(IoU) = 0.3|
|**Acc.**|0.2444|0.4589|0.3006|0.2764|0.1529|0.0379|0.1666|0.1326|
|**AFP**|1.0417|0.7815|0.9472|0.7236|0.7073|0.6168|1.1067|0.5325|
|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|T(IoU) = 0.4|
|**Acc.**|0.0944|0.2808|0.2026|0.1219|0.0705|0.0126|0.0750|0.0714|
|**AFP**|1.0783|0.8140|0.9705|0.7489|0.7164|0.6189|1.1239|0.5427|
|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|T(IoU) = 0.5|
|**Acc.**|0.0500|0.1780|0.1111|0.0650|0.0117|0.0126|0.0333|0.0306|
|**AFP**|1.0884|0.8354|0.9919|0.7571|0.7215|0.6189|1.1291|0.5478|
|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|T(IoU) = 0.6|
|**Acc.**|0.0222|0.0753|0.0457|0.0243|0.0000|0.0126|0.0166|0.0306|
|**AFP**|1.0935|0.8506|1.0051|0.7632|0.7226|0.6189|1.1321|0.5478|
|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|T(IoU) = 0.7|
|**Acc.**|0.0055|0.0273|0.0196|0.0000|0.0000|0.0000|0.0083|0.0204|
|**AFP**|1.0965|0.8577|1.009|0.7663|0.7226|0.6199|1.1331|0.5488|



Table 7. _Pathology localization accuracy and average false positive number for 8 disease classes with T_ ( _IoU_ ) _ranged from_ 0 _._ 1 _to_ 0 _._ 7 _._


|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings include: 1.<br>left basilar at-<br>electasis/consolidation. 2. prominent<br>hilum (mediastinal adenopathy).<br>3.<br>left pic catheter (tip in atriocaval junc-<br>tion). 4. stable, normal appearing car-<br>diomediastinal silhouette.<br>impression:<br>small right pleural ef-<br>fusion<br>otherwise<br>stable<br>abnormal<br>study including left basilar inﬁl-<br>trate/atelectasis,<br>prominent<br>hilum,<br>and position of left pic catheter (tip<br>atriocaval junction).|Effusion;<br>Inﬁltration;<br>Atelectasis||


Table 8. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Atelectasis” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._

|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings include: 1. cardiomegaly (ct<br>ratio of 17/30). 2. otherwise normal<br>lungs and mediastinal contours. 3. no<br>evidence of focal bone lesion. dictat-<br>ing|Cardiomegaly||



Table 9. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Cardiomegaly” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._

|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings: no appreciable change since<br>XX/XX/XX. small right pleural effu-<br>sion. elevation right hemidiaphragm.<br>diffuse small nodules throughout the<br>lungs, most numerous in the left<br>mid and lower lung.<br>impression:<br>no change with bilateral small lung<br>metastases.|Effusion;<br>Nodule||



Table 10. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Effusion” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._


|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings:<br>port-a-cath reservoir re-<br>mains in place on the right. chest tube<br>remains in place, tip in the left apex.<br>no pneumothorax. diffuse patchy in-<br>ﬁltrates bilaterally are decreasing.<br>impression: inﬁltrates and effusions<br>decreasing.|Inﬁltration||


Table 11. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Infiltration” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._

|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings:<br>right<br>internal<br>jugular<br>catheter remains in place.<br>large<br>metastatic lung mass in the lateral<br>left upper lobe is again noted. no in-<br>ﬁltrate or effusion. extensive surgical<br>clips again noted left axilla.<br>impression: no signiﬁcant change.|Mass||



Table 12. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Mass” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._

|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings: pa and lateral views of the<br>chest demonstrate stable 2.2 cm nod-<br>ule in left lower lung ﬁeld posteriorly.<br>the lungs are clear without inﬁltrate<br>or effusion. cardiomediastinal silhou-<br>ette is normal size and contour. pul-<br>monary vascularity is normal in cal-<br>iber and distribution.<br>impression: stable left likely hamar-<br>toma.|Nodule;<br>Inﬁltration||



Table 13. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Nodule” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._


|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings: unchanged left lower lung<br>ﬁeld inﬁltrate/air bronchograms. un-<br>changed right perihilar inﬁltrate with<br>obscuration of the right heart bor-<br>der.<br>no evidence of new inﬁltrate.<br>no evidence of pneumothorax the car-<br>diac and mediastinal contours are sta-<br>ble.<br>impression:<br>1.<br>no evidence<br>pneumothorax.<br>2.<br>unchanged left<br>lower lobe and left lingular consoli-<br>dation/bronchiectasis. 3. unchanged<br>right middle lobe inﬁltrate|Pneumonia;<br>Inﬁltration||


Table 14. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Pneumonia” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._

|Radiology report|Keyword|Localization Result|
|---|---|---|
|ﬁndings: frontal lateral chest x-ray<br>performed in expiration.<br>left apical<br>pneumothorax visible.<br>small pneu-<br>mothorax visible along the left heart<br>border and left hemidiaphragm. pleu-<br>ral thickening, mass right chest. the<br>mediastinum cannot be evaluated in<br>the expiration. bony structures intact.<br>impression:<br>left post biopsy pneu-<br>mothorax.|Mass;<br>Pneumothorax||



Table 15. _A sample of chest x-ray radiology report, mined disease keywords and localization result from the “Pneumothorax” Class. Correct_
_bounding box (in green), false positives (in red) and the ground truth (in blue) are plotted over the original image._


**B** **ChestX-ray14 Dataset**

After the CVPR submission, we expand the disease categories to include 6 more common thorax diseases (i.e. Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening and Hernia) and update the NLP mined labels. The
statistics of ChestX-ray14 dataset are illustrated in Table 16
and Figure 8. The bounding boxes for Pathologies are unchanged at this point.

|Item #|X-ray8|Ov.|X-ray14|Ov.|
|---|---|---|---|---|
|Report|108,948|-|112,120|-|
|Atelectasis<br>Cardiomegaly<br>Effusion<br>Inﬁltration<br>Mass<br>Nodule<br>Pneumonia<br>Pneumothorax<br>Consolidation<br>Edema<br>Emphysema<br>Fibrosis<br>PT<br>Hernia<br>No ﬁndings|5,789<br>1,010<br>6,331<br>10,317<br>6,046<br>1,971<br>1,062<br>2,793<br>-<br>-<br>-<br>-<br>-<br>-<br>84,312|3,286<br>475<br>4,017<br>4,698<br>3,432<br>1,041<br>703<br>1,403<br>-<br>-<br>-<br>-<br>-<br>-<br>0|11,535<br>2,772<br>13,307<br>19,871<br>5,746<br>6,323<br>1,353<br>5,298<br>4,667<br>2,303<br>2,516<br>1,686<br>3,385<br>227<br>60,412|7,323<br>1,678<br>9,348<br>10,319<br>2,138<br>3,617<br>1,046<br>3,099<br>3,353<br>1,669<br>1,621<br>959<br>2,258<br>117<br>0|



Table 16. _Total number (#) and # of Overlap (Ov.) of the corpus in_
_ChestX-ray8 and ChestX-ray14 datasets.PT: Pleural Thickening_


**B.1** **Evaluation of NLP Mined Labels**

To validate our method, we perform the following experiments. First, we resort to some existing annotated corpora
as an alternative, i.e. OpenI dataset. Furthermore, we annotated clinical reports suitable for evaluating finding recognition systems. We randomly selected 900 reports and asked
two annotators to mark the above 14 types of findings. Each
report was annotated by two annotators independently and
then agreements are reached for conflicts.
Table 18 shows the results of our method using OpenI
and our proposed dataset, measured in precision (P), recall
(R), and F1-score. Much higher precision, recall and F1scores are achieved compared to the existing MetaMap approach (with NegEx enabled). This indicates that the usage
of negation and uncertainty detection on syntactic level successfully removes false positive cases.


**B.2** **Benchmark Results**

In a similar fashion to the experiment on ChestX-ray8,
we evaluate and validate the unified disease classification
and localization framework on ChestX-ray14 database. In
total, 112,120 frontal-view X-ray images are used, of which
51,708 images contain one or more pathologies. The remaining 60,412 images do not contain the listed 14 disease
findings. For the pathology classification and localization



|ResNet-50|ChestX-ray8|ChestX-ray14|
|---|---|---|
|**Atelectasis**<br>**Cardiomegaly**<br>**Effusion**<br>**Inﬁltration**<br>**Mass**<br>**Nodule**<br>**Pneumonia**<br>**Pneumothorax**<br>**Consolidation**<br>**Edema**<br>**Emphysema**<br>**Fibrosis**<br>**PT**<br>**Hernia**|0.7069<br>0.8141<br>0.7362<br>0.6128<br>0.5609<br>0.7164<br>0.6333<br>0.7891<br>-<br>-<br>-<br>-<br>-<br>-|0.7003<br>0.8100<br>0.7585<br>0.6614<br>0.6933<br>0.6687<br>0.6580<br>0.7993<br>0.7032<br>0.8052<br>0.8330<br>0.7859<br>0.6835<br>0.8717|


Table 17. _AUCs of ROC curves for multi-label classification for_
_ChestX-ray14 using published data split. PT: Pleural Thickening_


Figure 7. _Multi-label classification performance on ChestX-ray14_
_with ImageNet pre-trained ResNet._


task, we randomly shuffled the entire dataset into three subgroups on the patient level for CNN fine-tuning via Stochastic Gradient Descent (SGD): i.e. training ( _∼_ 70%), validation ( _∼_ 10%) and testing ( _∼_ 20%). All images from the
same patient will only appear in one of the three sets. [4] We
report the 14 thoracic disease recognition performance on
the published testing set in comparison with the counterpart
based on ChestX-ray8, shown in Table 17 and Figure 7.
Since the annotated B-Boxes of pathologies are unchanged, we only test the localization performance on the
original 8 categories. Results measured by the Intersection
over the detected B-Box area ratio (IoBB) (similar to Area
of Precision or Purity) are demonstrated in Table 19.
Overall, both of the classification and localization performance on ChestX-ray14 is equivalent to the counterpart on
ChestX-ray8.


4 [Data split files could be downloaded via https://nihcc.app.](https://nihcc.app.box.com/v/ChestXray-NIHCC)
[box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)


MetaMap Our Method
Disease
Precision / Recall / F1-score Precision / Recall / F1-score

OpenI
Atelectasis 87.3 / 96.5 / 91.7 88.7 / 96.5 / 92.4

Cardiomegaly 100.0 / 85.5 / 92.2 100.0 / 85.5 / 92.2
Effusion 90.3 / 87.5 / 88.9 96.6 / 87.5 / 91.8

Infiltration 68.0 / 100.0 / 81.0 81.0 / 100.0 / 89.5
Mass 100.0 / 66.7 / 80.0 100.0 / 66.7 / 80.0

Nodule 86.7 / 65.0 / 74.3 82.4 / 70.0 / 75.7

Pneumonia 40.0 / 80.0 / 53.3 44.4 / 80.0 / 57.1

Pneumothorax 80.0 / 57.1 / 66.7 80.0 / 57.1 / 66.7

Consolidation 16.3 / 87.5 / 27.5 77.8 / 87.5 / 82.4

Edema 66.7 / 90.9 / 76.9 76.9 / 90.9 / 83.3

Emphysema 94.1 / 64.0 / 76.2 94.1 / 64.0 / 76.2
Fibrosis 100.0 / 100.0 / 100.0 100.0 / 100.0 / 100.0

PT 100.0 / 75.0 / 85.7 100.0 / 75.0 / 85.7

Hernia 100.0 / 100.0 / 100.0 100.0 / 100.0 / 100.0

_Total_ 77.2 / 84.6 / 80.7 89.8 / 85.0 / 87.3

ChestX-ray14
Atelectasis 88.6 / 98.1 / 93.1 96.6 / 97.3 / 96.9

Cardiomegaly 94.1 / 95.7 / 94.9 96.7 / 95.7 / 96.2
Effusion 87.7 / 99.6 / 93.3 94.8 / 99.2 / 97.0

Infiltration 69.7 / 90.0 / 78.6 95.9 / 85.6 / 90.4
Mass 85.1 / 92.5 / 88.7 92.5 / 92.5 / 92.5

Nodule 78.4 / 92.3 / 84.8 84.5 / 92.3 / 88.2

Pneumonia 73.8 / 87.3 / 80.0 88.9 / 87.3 / 88.1

Pneumothorax 87.4 / 100.0 / 93.3 94.3 / 98.8 / 96.5

Consolidation 72.8 / 98.3 / 83.7 95.2 / 98.3 / 96.7

Edema 72.1 / 93.9 / 81.6 96.9 / 93.9 / 95.43

Emphysema 97.6 / 93.2 / 95.3 100.0 / 90.9 / 95.2
Fibrosis 84.6 / 100.0 / 91.7 91.7 / 100.0 / 95.7

PT 85.1 / 97.6 / 90.9 97.6 / 97.6 / 97.6

Hernia 66.7 / 100.0 / 80.0 100.0 / 100.0 / 100.0

_Total_ 82.8 / 95.5 / 88.7 94.4 / 94.4 / 94.4


Table 18. _Evaluation of image labeling results on OpenI and ChestX-ray14 dataset. Performance is reported using P, R, F1-score. PT:_
_Pleural Thickening_

|T(IoBB)|Atelectasis|Cardiomegaly|Effusion|Infiltration|Mass|Nodule|Pneumonia|Pneumothorax|
|---|---|---|---|---|---|---|---|---|
|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|T(IoBB) = 0.1|
|**Acc.**|0.6222|1|0.7974|0.9106|0.5882|0.1519|0.8583|0.5204|
|**AFP**|0.8293|0.1768|0.6148|0.4919|0.3933|0.4685|0.4360|0.4543|
|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|**T(IoBB) = 0.25** (Two times larger on both x and y axis than ground truth B-Boxes)|
|**Acc.**|0.3944|0.9863|0.6339|0.7967|0.4588|0.0506|0.7083|0.3367|
|**AFP**|0.9319|0.2042|0.6880|0.5447|0.4288|0.4786|0.4959|0.4857|
|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|T(IoBB) = 0.5|
|**Acc.**|0.1944|0.9452|0.4183|0.6504|0.3058|0|0.4833|0.2653|
|**AFP**|0.9979|0.2785|0.7652|0.6006|0.4604|0.4827|0.5630|0.5030|
|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|T(IoBB) = 0.75|
|**Acc.**|0.0889|0.8151|0.2287|0.4390|0.1647|0|0.2917|0.1735|
|**AFP**|1.0285|0.4045|0.8222|0.6697|0.4827|0.4827|0.6169|0.5243|
|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|T(IoBB) = 0.9|
|**Acc.**|0.0722|0.6507|0.1373|0.3577|0.0941|0|0.2333|0.1224|
|**AFP**|1.0356|0.4837|0.8445|0.7043|0.4939|0.4827|0.6331|0.5346|



Table 19. _Pathology localization accuracy and average false positive number for ChestX-ray14._


Figure 8. _The circular diagram shows the proportions of images with multi-labels in each of 14 pathology classes and the labels’ co-_

_occurrence statistics._




