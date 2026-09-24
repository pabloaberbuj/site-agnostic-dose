

**HHS Public Access** Author manuscript 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Published in final edited form as: Med Phys. 2024 June ; 51(6): 3932–3949. doi:10.1002/mp.17115. 

# **Multimodal Radiotherapy Dose Prediction Using a Multi-Task Deep Learning Model** 

**Austen Maniscalco** , **Ezek Mathew** , **David Parsons** , **Justin Visak** , **Mona Arbab** , **Prasanna Alluri** , **Xingzhe Li** , **Narine Wandrey** , **Mu-Han Lin** , **Asal Rahimi** , **Steve Jiang** , **Dan Nguyen** 

Medical Artificial Intelligence and Automation Laboratory, Department of Radiation Oncology, University of Texas Southwestern Medical Center, Dallas, TX, 75390, USA 

## **Abstract** 

**Background:** In radiation therapy (RT), accelerated partial breast irradiation (APBI) has emerged as an increasingly preferred treatment modality over conventional whole breast irradiation due to its targeted dose delivery and shorter course of treatment. APBI can be delivered through various modalities including Cobalt-60-based systems and linear accelerators with C-arm, O-ring, or robotic arm design. Each modality possesses distinct features, such as beam energy or the degrees of freedom in treatment planning, which influence their respective dose distributions. These modality-specific considerations emphasize the need for a quantitative approach in determining the optimal dose delivery modality on a patient-specific basis. However, manually generating treatment plans for each modality across every patient is time-consuming and clinically impractical. 

**Purpose:** We aim to develop an efficient and personalized approach for determining the optimal RT modality for APBI by training predictive models using two different deep learning-based convolutional neural networks. The baseline network performs a single-task, predicting dose for a single modality. Our proposed multi-task network, which is capable of leveraging shared information among different tasks, can concurrently predict dose distributions for various RT modalities. Utilizing patient-specific input data, such as a patient’s computed tomography (CT) 

Dan.Nguyen@UTSouthwestern.edu . VII.CONFLICTS OF INTEREST There are no conflicts of interest to disclose. 

Maniscalco et al. 

Page 2 

scan and treatment protocol dosimetric goals, the multi-task model predicts patient-specific dose distributions across all trained modalities. These dose distributions provide patients and clinicians quantitative insights, facilitating informed and personalized modality comparison prior to treatment planning. 

**Methods:** The dataset, comprising 28 APBI patients and their 92 treatment plans, was partitioned into training, validation, and test subsets. 8 patients were dedicated to the test subset, leaving 68 treatment plans across 20 patients to divide between the training and validation subsets. Single-task models were trained for each modality, and one multi-task model was trained to predict doses for all modalities simultaneously. Model performance was evaluated across the test dataset in terms of Mean Absolute Percent Error (MAPE). We conducted statistical analysis of model performance using the two-tailed Wilcoxon signed-rank test. 

**Results:** Training times for five single-task models ranged from 255 to 430 minutes per modality, totaling 1925 minutes, while the multi-task model required 2384 minutes. Multi-task model prediction required an average of 1.82 seconds per patient, compared to single-task model predictions at 0.93 seconds per modality. The multi-task model yielded MAPE of 1.1033 ± 0.3627% as opposed to the collective MAPE of 1.2386 ± 0.3872% from single-task models, and the differences were statistically significant (p = 0.0003, 95% confidence interval = [−0.0865, −0.0712]). 

**Conclusion:** Our study highlights the potential benefits of a multi-task learning framework in predicting RT dose distributions across various modalities without notable compromises. This multi-task architecture approach offers several advantages, such as flexibility, scalability, and streamlined model management, making it an appealing solution for clinical deployment. With such a multi-task model, patients can make more informed treatment decisions, physicians gain more quantitative insight for pre-treatment decision-making, and clinics can better optimize resource allocation. With our proposed goal array and multi-task framework, we aim to expand this work to a site-agnostic dose prediction model, enhancing its generalizability and applicability. 

### **Keywords** 

radiation therapy; deep learning; artificial intelligence; multi-task; modality comparison; dose prediction; breast cancer 

## **I. INTRODUCTION** 

### **I.1. Background** 

In the management of patients with early stage breast carcinoma, the prevailing standard has been a combination of surgery, such as lumpectomy, with subsequent whole breast irradiation. However, modern radiation therapy (RT) is shifting focus towards a reduction in both the duration and scope of treatment using an approach called accelerated partial breast irradiation (APBI)<sup>1–5</sup> . The variability inherent in APBI is exemplified by Goldberg et al.’s review of seven modern APBI trials, each proposing distinct RT prescription doses, fractionations, and/or target volumes<sup>6</sup> . For instance, the RAPID trial described by Whelan et al. had prescribed 38.5 Gy over 10 fractions twice daily with specific clinical target volume (CTV) delineation guidance, whereas the University of Florence’s regimen 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 3 

described by Meattini et al. had prescribed 30 Gy over 5 fractions across 2 weeks with different guidance for CTV delineation<sup>7,8</sup> . The lack of agreement on the optimal APBI treatment approach also extends to RT modality selection and treatment technique, given the diverse range of models and treatment techniques that have recently emerged which each have distinct capabilities and limitations. 

Recent advancements in external beam radiation therapy (EBRT) dose delivery include techniques such as intensity-modulated radiation therapy (IMRT), volumetric modulated arc therapy (VMAT), and helical tomotherapy (HT), facilitated by modern linear accelerators (LINACs)<sup>9</sup> . Each LINAC may have distinct capabilities, such as available photon beam energies or available degrees of freedom (DOF) in treatment planning, which influence dose delivery. This study features several commercial RT delivery modalities, such as Varian’s TrueBeam (Varian Medical Systems, Inc., Palo Alto, CA), Varian’s Ethos (Varian Medical Systems, Inc., Palo Alto, CA), Elekta’s Unity (Elekta AB, Crawley, United Kingdom), Accuray’s CyberKnife (Accuray, Inc., Sunnyvale, CA) and Xcision’s GammaPod (Xcision Medical Systems LLC, Columbia, MD). 

Varian’s TrueBeam (TB) model, for example, is a conventional C-arm LINAC that uses photon beam energies generally ranging from 6 to 18 MV, including flattening filter free (FFF) modes, and allows rotation of the collimator and treatment couch. Varian’s Ethos model is an O-ring LINAC that features deep-learning based adaptive workflows and is designed to operate rapidly and increase patient throughput<sup>10</sup> . However, Ethos faces some limitations: it cannot rotate its couch in treatment planning, its photon energy is limited to 6 MV-FFF, and its VMAT optimization may be too lengthy to fit into the rapid adaptive workflow as compared to that of IMRT<sup>11</sup> . Elekta’s Unity model integrates magnetic resonance imaging (MRI) capability in a LINAC, but this comes at an inherent dosimetric cost due to the magnetic field interactions with electrons in tissue<sup>12</sup> . Additionally, the Unity model delivers dose at a greater distance from the isocenter, its couch movements are limited to the superior-inferior direction, its Multi-Leaf Collimator (MLC) cannot be rotated, its MLC leaves are limited to traveling in the cranio-caudal direction, its photon beam energy is limited to 7 MV-FFF, and it currently only delivers dose via the step-andshoot IMRT technique. While Ding et al. determined that the Unity model could achieve clinically acceptable plans, they observed dosimetric differences as compared to plans from a conventional LINAC<sup>13</sup> . 

Accuray’s CyberKnife (CK) model differs from the aforementioned LINACs: It delivers dose with a robotic arm that traverses a predefined path in a non-isocentric manner, its photon beam energy is limited to 6 MV-FFF, and it has active image guidance capability that can correct for patient movement during treatment. CK’s primary limitation is its prolonged setup and treatment time, as Kurup et al. found that a single session for a patient may last up to 60 minutes<sup>14</sup> . Lastly, Xcision’s GammaPod (GP) employs a novel Cobalt-60-based treatment system that immobilizes a patient’s breast using a vacuum suction within a breast cup to ensure a reproducible setup in the prone position. The GP system allows for isocentric dose delivery using a non-coplanar technique, and Mutaf et al. describe that the treatment couch motor allows for dynamic rotation relative to the isocenter during beam delivery 15. In lieu of MLC, beam shaping is achieved via a rotating hemispherical collimator 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 4 

with pre-drilled holes with diameters of either 15mm, or 25mm. The GP system’s most notable limitations include the management of radioactive materials and a limited scope of application. 

Dose distributions in EBRT can also vary based on a vendor’s treatment planning system (TPS) and the optimization algorithm used. Gallio et al. investigated intra- and inter-TPS dose discrepancies, identifying intra-TPS differences for Pinnacle between manual and auto planning<sup>16</sup> . Their publication’s figures displayed observable dosimetric differences interTPS, but their group did not identify statistical significance within organs at risk (OARs). Additionally, Tsuruta et al. examined dose differences across 26 plans using three distinct dose calculation algorithms: Acuros XB (AXB), anisotropic analytical algorithm (AAA), and x-ray voxel Monte Carlo (XVMC)<sup>17</sup> . They identified statistically significant dosimetric differences between all three algorithms at isocenter, as well as statistically significant differences between 2 out of 3 algorithms in the planning target volume (PTV) max dose and homogeneity index (HI). 

Previous works by Moran et al. and Qiu et al. scrutinized differences among EBRT dose delivery techniques for patients undergoing APBI, demonstrating the dosimetric benefit of utilizing IMRT and/or VMAT over 3D-conformal techniques<sup>18,19</sup> . Similar analyses were published for other disease sites, such as works by Chao et al., Holt et al., and Pigorsch et al., collectively demonstrating the advantage of IMRT/VMAT outside of APBI<sup>20–22</sup> . Ding et al. and Nachbar et al. compared doses from treatment plans generated for an MRILINAC and conventional LINAC, determining that MRI-LINAC dose distributions were “clinically acceptable”<sup>13,23</sup> . Works by Lin et al. and Zhang et al. contrasted CK plans with conventional LINAC VMAT plans, respectively concluding that VMAT was dosimetrically preferential for the prostate and CK for the brain<sup>24,25</sup> . Stroubinis et al. and Li et al. analyzed tradeoffs between Varian’s Halcyon and C-arm LINACs for spinal irradiation, finding that the modalities yielded dosimetrically similar plans but cautioning that Halcyon’s couch has limited degrees of freedom which can complicate patient setup<sup>26,27</sup> . Though these studies offer generalized conclusions for a sample of the population, we posit that selection of the optimal EBRT dose delivery technique can benefit from patient-specific assessment rather than sole reliance on broad recommendations. Factors like a patient’s specific RT target volume, organ geometry, or individual risk factors could uniquely influence their optimal dose distribution. For instance, a physician may prioritize a technique that minimizes heart mean dose, even if the chosen technique isn’t the absolute best in terms of the other organ metrics. 

### **I.2. Multi-Task Network Proposal** 

Considering this complex dosimetric landscape and individualized patient needs, there is a pressing need for a comprehensive, patient-centric approach that compares dosimetric differences across various techniques based on an individual patient’s PTV and OARs. However, treatment planning is a time-consuming process, and it may not be clinically feasible to require treatment planners to generate multiple plans for each patient with different machine model(s) and TPS(s). Cilla et al. reported that auto-planning reduced average treatment planning time by nearly one third, yet a single plan still required 60 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Page 5 

Maniscalco et al. 

to 80 minutes<sup>28</sup> . Moreover, in the situation that certain facilities have a limited range of machine models available, they may opt to refer patients to other institutions with alternative machines that offer the potential for superior dosimetric outcomes. In such a circumstance, the referring facility would desire to minimize the amount of time spent by their staff towards creating treatment plans for their own machines that ultimately won’t be utilized. 

We propose a deep learning-based multi-task convolutional neural network (CNN) designed to predict dose distributions for various radiation therapy modalities simultaneously. With an individual patient’s CT scan, PTV, and OARs, the model can predict this patient’s corresponding dose distributions. This multi-modality dose prediction model aims to provide patient-centric guidance in selecting the most suitable radiation therapy modality. After the dose distributions are simultaneously predicted, they can be assessed using dose-volume histograms (DVHs), dosimetric goals, or dose distribution visuals to identify the most optimal modality for that patient. 

We could not identify any prior works regarding multi-modality dose prediction within RT. However, there is a robust body of work regarding dose prediction for an individual modality. Early works in this topic include knowledge-based dose prediction by application of an artificial neural network (ANN), as demonstrated by Shiraishi et al on prostate patients and Campbell et al. on pancreatic patients<sup>29,30</sup> . Shortly thereafter, Nguyen et al. introduced a novel deep learning architecture for dose prediction, termed a hierarchically densely connected U-Net (HD U-Net), and evaluated its performance on head and neck (H&N) patients<sup>31</sup> . In more recent works, Zhan et al. highlighted the use of a generative adversarial network and numerous loss functions for rectal and cervical dose prediction, whereas Li et al. implemented a difficulty-aware mechanism within a modified multi-stage U-Net for H&N dose prediction<sup>32,33</sup> . In our group’s recent related work, we demonstrated the performance of our fine-tuned, patient-specific ART dose prediction models for H&N patients<sup>34</sup> . We seek to build on these prior single-task dose prediction works using our multi-task framework. 

There are a few prior works related to broader multi-task learning in dose prediction. For example, Wen et al. crafted a network that predicts a single modality’s dose distribution with the assistance of two additional auxiliary dose prediction tasks: an isodose line map task and a dose gradient map task<sup>35</sup> . Additionally, a mask-free dose predictor was proposed by Jiao et al. that only requires the CT as model input to predicts the tumor segmentation mask and corresponding dose distribution, though their work was limited to the prostate and they did not compare performance with a model that adds OARs or targets as input<sup>36</sup> . Unlike these works, we aim to utilize related but distinct tasks to improve the predictive performance of our multi-task model without task interdependence. 

Our CNN is based on the U-Net architecture, leveraging a shared encoder and decoder that transmit feature maps to multiple independent output convolutional layers. In the shared decoder, we produce two types of feature maps: one that fuses decoder-level features with corresponding encoder-level features via skip connections prior to each upsampling operation, and another that performs upsampling without skip connections. The feature maps generated with skip connections are independently transmitted to each output layer 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Page 6 

Maniscalco et al. 

tasked with predicting a single dose distribution, whereas the feature maps that omitted skip connections are transmitted to an output layer tasked with predicting all dose prediction types simultaneously. This task, which we simply refer to as a regularization task, omits skip connections to encourage a robust representation of the input data throughout the encoder and decoder and potentially regularizes the network to prevent modality-specific overfitting. As each dose prediction task is isolated to its own layer, our architecture maintains distinct weights tailored to each task and offers flexibility for future modifications. As new radiation therapy modalities are introduced and pre-existing modalities are revised, our design facilitates the addition, removal, and/or fine-tuning of specific output layers. We hypothesize that our deep learning-based multi-task CNN can leverage shared information to train a robust multi-modality dose prediction model that demonstrates statistically significant differences in dose prediction as compared to multiple individually trained dose prediction networks. 

## **II. METHODS** 

### **II.1. Breast Patient Data** 

28 APBI patients were selected for this study. 16 patients had a supine non-contrast planning CT scan with arms up while immobilized by a vacuum bag and wooden frame, whereas the remaining 12 patients had a prone non-contrast planning CT scan in the GP image loader. Gross tumor volumes (GTVs) were contoured by radiation oncologists. Conventional CTVs and PTVs were generated based on the Florence trial’s original method as published by Meattini et al., and adaptive CTVs and PTVs were generated using our institutional guidelines<sup>8</sup> . 

The conventional CTV was generated with a uniform 10mm 3D expansion from GTV and then a 3mm crop from skin surface. The conventional PTV was generated with an additional uniform 10mm 3D expansion from the conventional CTV, then a 3mm crop from skin surface, and finally limiting its extent to no greater than 4mm deep in the ipsilateral lung. Next, an adaptive CTV was generated with a uniform 10mm 3D expansion from GTV, a 5mm crop from skin surface, and subtracting the chest wall. The adaptive PTV was generated with a uniform 3mm expansion from adaptive CTV. 

The following OARs were contoured for all patients and used in this study: body, left breast, right breast, heart, left lung, right lung, ribs, skin and spinal canal. All patients were prescribed 30 Gy over 5 fractions to the PTV. All 16 supine patients contributed 5 unique treatment plans to the dataset for a total of 80 treatment plans: Ethos, Unity, CK and TB treatment plans were created based on the adaptive PTV, and an additional TB plan was created based on the conventional PTV. The remaining 12 prone patients each contributed a GP treatment plan based on the adaptive PTV. The grand total number of treatment plans across all patients was 92. Any treatment plans that were not already created in clinical practice were retrospectively created for this study by experienced treatment planners. 

DICOM data was exported from each TPS and converted to NumPy arrays. The CT array contained Hounsfield units (HU) for each voxel position. To preprocess the CT array, we clipped HU values by replacing any values less than −1000 with −1000 and any values 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 7 

greater than 5000 with 5000. Following this, we rescaled HU values to an absolute range of 0 to 1. Each dose array was divided by prescription dose, which was 30 Gy, so that its values were normalized from 0 to slightly above 1 (exact value subject to each individual plan’s maximum dose). Individual binary arrays were generated for both PTVs and each OAR, in which an element had value of 1 if the specific structure was present for the respective voxel or 0 otherwise. A Euclidean distance array was created for each patient (Figure 1). An element value of 1.0 represents PTV location, and an element value that is below 1.0 and decreasing towards 0.0 represents increasing Euclidean distance from the PTV. 

Finally, we introduce a novel concept that we term a “goal array”, which consolidates spatial information from multiple OAR arrays into a single array and augments it with dosimetric context through treatment planning objectives (Figure 2). Traditional methods for training dose prediction models rely on specific OARs as model input, which limit the practicality and generalizability of the model because one or more of the requisite input OARs may be unnecessary or omitted in clinical practice due to patient-specific considerations. To generate the goal array, we assign dose values to voxels based on treatment planning OAR sparing goals. For example, an objective for PTV maximum dose may be less than 110% of the prescription dose, so voxels in the goal array where PTV is present are assigned value slightly below 110% of prescription dose. When an objective is volumetric-based, such as the volume of ipsilateral lung receiving 9 Gy or more should be less than 10% (V9 Gy < 10%), we solve for a sub-volume of requisite spared lung most distal to the PTV – in this case, 90% of the ipsilateral lung – and assign these voxels to values of 9 Gy. All OARs were also assigned an arbitrary maximum dose objective, such as 105% of prescription dose, to guarantee that all of their voxels had values in the final goal array. In the circumstance of overlapping sub-volume objectives, we consolidate them all by taking the minimum value for each voxel across all sub-volumes. For any voxel(s) contained within the body, but outside any PTV or OAR, an arbitrary value equal to 95% of the prescription dose was assigned. Goal array voxels outside the body were equal to zero. As a final preprocessing step, we divided the goal array by prescription dose to match the dose array’s normalization method. 

In total, the input data shall include the following arrays: 1) CT scan, 2) PTV, 3) PTV distance array, and 4) Goal array. As a form of data augmentation for TB, the input data’s PTV and corresponding dose label had a 50% chance to be based on either the conventional plan or adaptive plan. All data was resized to 5mm<sup>3</sup> voxel spacing because we were resource-limited by the graphics processing unit’s (GPU) video random access memory (VRAM) maximum capacity. The precise shape of each patient’s resized arrays varied based on the extent encompassed by the CT scan and its original voxel spacing. 5mm<sup>3</sup> voxel spacing allowed us to capture a significant percentage of the data in randomly selected 3D patches, which were NumPy arrays with uniform spatial dimension sizes of 112, 112, and 64 for x, y, and z respectively. Therefore, considering we used 4 input channels and a batch size of 4, the final array shape for the input data was [4, 4, 112, 112, 64]. 

The input data array is created through random patch selection. We choose a patch center randomly by sampling from a Gaussian distribution that uses the GTV’s center of mass as its mean. The volume of interest (VOI) expanded from the patch center by increasing 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 8 

its extent in a random direction for each spatial dimension until it reached the requisite patch shape. After this process, we performed a final random translation of the VOI in each spatial dimension by up to 5 elements per dimension. There were two final and independent forms of data augmentation: 50% chance of randomly rotating the VOI and 50% chance of randomly flipping the VOI. If the VOI was to be rotated, there was an additional independent 25% chance of executing the rotation by one of 0, 90, 180 or 270 degrees. The input data array is then converted to a tensor. 

### **II.2. Model Design** 

Our CNN is based on the U-Net architecture and implemented using the PyTorch framework (Figure 3). Input data with 4 channels is sent to a shared encoder, which is composed of four encoding layers. Each encoding layer contains two convolutional layers and a downsampling layer. Our convolutional layers sequentially utilize a 3D convolution with kernel size of 3 and stride of 1, a custom activation function, and group normalization across 6 groups of channels. The custom activation function is a convolved concatenated ReLU (CCReLU) – this function concatenates a ReLU and a negative ReLU, then performs a convolution to maintain consistent data shape. Downsampling layers first use a kernel size of 2 to retrieve the 3D max pool, 3D average pool, and 3D strided convolution (stride of 2) of the input tensor, followed by a concatenation of these three resulting tensors. The concatenated tensor is finally passed to a modified convolutional layer that uses kernel size of 1, reducing the number of feature channels to maintain consistent data shape. 

The first encoding layer shall output a tensor of shape [4, 36, 112, 112, 64] with 36 initial feature maps. The second, third, and fourth encoding layers output tensors of shapes [4, 72, 56, 56, 32], [4, 144, 28, 28, 16], and [4, 288, 14, 14, 8] respectively. The final encoding layer slightly differs from the other encoding layers, as it contains an additional two convolutional layers after its downsampling layer. This point is the bottleneck of our network and outputs a tensor of shape [4, 576, 7, 7, 4] with 576 feature maps, which is then sent to the shared decoder. 

The shared decoder contains four decoding layers. Each decoding layer contains an upsampling layer and two convolutional layers. An upsampling layer contains two separate upsampling operations, a transposed convolutional layer, and a convolutional layer. One upsampling operation uses nearest-neighbor interpolation on a copy of the input data, and the other upsampling operation uses trilinear interpolation on a copy of the input data. The transposed convolutional layer also receives a copy of the input data, and only differs from our standard convolutional layer by its utilization of a 3D transposed convolution in place of the usual 3D convolution. These three output tensors are concatenated and the concatenated tensor is sent to the upsampling layer’s convolutional layer, which performs convolution using kernel size of 1 and stride of 1. Exiting the upsampling layer, the decoder has optional skip connections, and then two additional convolutional layers. A copy of the feature maps is created at this point. One tensor will continue through the shared decoder while utilizing skip connections, so its feature maps will be concatenated from the encoder and decoder at the same level of the U-Net before performing the two convolutions. The other tensor will bypass skip connections, so the size of the output tensor from the upsampling layers 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 9 

is modified to maintain consistent tensor shape across both types of tensors. After the first decoding layer, both tensors yield the shape of [4, 288, 14, 14, 8]. The second, third, and fourth decoding layers yield both types of tensors with the same shapes of [4, 144, 28, 28, 16], [4, 72, 56, 56, 32], and [4, 36, 112, 112, 64] respectively. 

The tensor containing features maps from the skip connection path shall be distributed to each of five independent output convolutional layers. We utilize skip connections to allow the network to carry forward high resolution information from the input data’s lowlevel features and concatenate this to the compressed latent representations containing the input data’s high-level features. Each of these five output convolutional layers is tasked with dose prediction for a distinct radiation therapy modality. On the other hand, the other tensor that bypassed skip connections shall be distributed to a sixth independent convolutional layer. This sixth output layer predicts the dose distribution for all 5 modalities simultaneously, omitting skip connections to regularize the network and encourage strong, high-level representation of the input data in the network’s feature maps. All six of the output convolutional layers consist of one convolutional operation and a linear activation function. Each predicted dose tensor has only one channel and shape of [4, 1, 112, 112, 64], so the regularization task’s output tensor has shape of [4, 5, 112, 112, 64]. 

### **II.3. Model Training** 

The dataset was partitioned into train, validation, and test subsets. We withheld data from 8 out of the 28 patients to establish the test dataset. 4 test patients have CK, Unity, Ethos, and TB plans, whereas the other 4 test patients have GP plans. From the remaining 20 patients, 10 of these patients contributed one of their modalities to the validation dataset. In other words, a patient may have contributed its CK plan to the validation dataset, but any remaining plans were used for model training. The test and validation datasets were equally comprised of left-sided and right-sided plans for each modality. In total, 56 RT plans remained available for model training: 10 CK plans, Unity plans, 10 Ethos plans, 10 adaptive PTV TB plans, 10 conventional PTV TB plans, and 6 GP plans. 

To establish a baseline for performance evaluation of our multi-task architecture, we trained independent single-task dose prediction models for each of the five RT modalities as illustrated in Figure 4. In single-task model training, we maintained an identical dataset split for each modality. For example, in training a single-task CK dose prediction model, the same patients were assigned to either validation or training based on those that dedicated their CK dose to either of validation or training for the multi-task dose prediction model. 

All model training in this study commenced from scratch. We utilized the AdamW optimizer with randomly initialized model weights. The learning rate (LR) cycled between a minimum of 1e-6 and a maximum of 1e-4 using a LR scheduler. The LR scheduler, termed OneCycleLR, was theorized by Smith et al. to boost performance when faced with a small amount of labeled training data<sup>37</sup> . OneCycleLR adjusts the LR cyclically during model training in steps, increasing the LR upwards to the maximum and then back downwards. We used 3D DropBlock with a maximum block size of 5×5×5, and a dropout rate of 0.05. Weight decay was set to zero, as we did not observe a performance improvement with it. A fixed random seed minimized variations in randomization between different model trainings. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 10 

The batch size was set to 4 for all training. Validation data patch selection was consistent such that each patient’s patch extent did not differ between epochs and no augmentation was performed. Each model was trained for 5,000 epochs, and model weights were saved at the epoch with the lowest validation loss. 

Two loss criteria were employed during model training. MSE loss (LMSE) was calculated voxel-wise for each modality’s dose distribution (Equation 1). LMSE emphasizes minimization of large discrepancies between predictions and labels. 





MSLE loss (LMSLE) was also calculated voxel-wise for each modality’s dose distribution, but only evaluated for voxels within the body contour and those with predicted values greater than −1 (Equation 2). LMSLE was employed to greater emphasize penalization of small dose differences, honing model weights towards finer dose prediction accuracy within the body contour. 



The total loss for each task (LTASK) was a summation of LMSE and LMSLE, which were weighted by constants α and β, respectively (Equation 3). These constants balanced the magnitudes of their respective losses. For this study, these constants were approximated as α = 4 and β = 1. In single-task model training, only LTASK was employed for backpropagation. 





The comprehensive multi-task loss (LMTL) used for backpropagation in multi-task model training was a sum of losses from each task (Equation 4). A scaling factor, δ, was used to balance the magnitude of the regularization task’s loss (LTASK_6) as compared to the sum of the modality-specific dose prediction tasks 1–5. In this study, δ was approximated as 4. 



Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 11 

We employed a weight freezing scheme during model training as follows: The model weights in the regularization layers, which were randomly initialized, were immediately frozen and remained frozen throughout the entire MT model training process. For taskspecific layers, their weights were frozen if their respective task-specific validation loss did not improve for 500 epochs. Additionally, whenever the first task-specific layer was frozen, the weights of both the encoder and decoder layers were also frozen. 

### **II.4. Evaluation Metrics** 

Upon the completion of training for all models, dose distributions were predicted for all patients within the test subset. These dose predictions were subsequently normalized to align with the percentage of PTV that received prescription dose (DRX) in the corresponding ground truth plan for each respective patient and modality. For instance, if the percentage of PTV that received 30 Gy (V30 Gy) equaled 91% in the ground truth TB plan for a patient, then predicted dose distribution(s) for that combination of patient and modality were normalized to reflect the same PTV criteria of V30 Gy=91%. 

To differentiate between dose predictions from the multi-task and single-task models, the Mean Absolute Percent Error (MAPE) was calculated for voxels within the patient’s body contour. MAPE was computed for each of the predicted dose distributions (Dpred) and their corresponding ground truth dose distributions (DGT), all relative to the patient’s DRX (Equation 5). In this study, all patients had DRX = 30 Gy. 



In addition, we utilized the open-source PyMedPhys python library to calculate gamma passing rates (set to 3%/3mm with 10% low-dose threshold and global normalization) for each dose prediction in reference to the ground truth dose<sup>38</sup> . Overall, the evaluation process included quantitative comparison of cumulative time to train each model and generate predictions, statistical significance testing for differences in MAPE between models, quantitative comparison of gamma passing rate and various dose metrics, and qualitative comparison of dose distributions. For statistical analysis, we selected the twotailed Wilcoxon signed-rank test. The assumptions of the Wilcoxon signed-rank test require that observations are paired, independent, randomly sampled, and measured on an ordinal or continuous scale. We favored the Wilcoxon signed-rank test over the paired t-test because our test dataset violates the normality assumption, whereas former is a non-parametric test that does not rely on it. Observations were paired, as we compared multi-task model predictions with single-task model predictions. Observations were also independent, as dose predictions did not directly influence each other. Furthermore, observations were randomly sampled and evaluated with MAPE, which is a continuous metric. 

The following dose metrics were utilized for model performance comparison: Homogeneity index (HI), conformity index, gradient index, max dose, mean dose, minimum dose, the 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 12 

minimum dose delivered to a specified percent of a structure (D_ %), and the percent volume of a structure receiving at least a specified dose (V_ Gy). The homogeneity index (HI) was evaluated within the PTV by subtracting D98% from D2% and then dividing by D50% (Equation 6). A lower HI value indicates improved dose homogeneity within the PTV. 





Conformity index (CI) was also evaluated for the PTV, and was obtained using the equation proposed by Paddick et al<sup>39</sup> . This CI equation squares the PTV volume that received at least DRX and divides it by the product of PTV volume and the volume of body that received at least DRX (Equation 7). A conformity index value of 1.0 is optimal as it indicates ideal dose conformity to the PTV, whereas a value below 1.0 indicates less ideal PTV coverage. 



Finally, gradient index (GI) was evaluated with respect to the body. It is a measure of the volume of body that received at least 50% of DRX divided by the volume of body that received at least 100% of DRX (Equation 8). GI characterizes the gradient of dose drop-off as a ratio of volume that received intermediate dose as compared to volume that received prescription dose. Broadly, lower GI values are desirable as they indicate a steeper dose drop-off, leading to an overall reduction in dose delivered to the patient. 



## **III. RESULTS** 

Five single-task (ST) models were trained, one for each modality of CK, Ethos, TB, Unity, and GP. Single-task models completed training in 420 minutes, 430 minutes, 421 minutes, 399 minutes, and 255 minutes respectively, for a total of 1925 minutes. The multi-task (MT) model completed training in 2384 minutes. Both ST and MT models were used to predict all relevant test dataset dose distributions on the GPU. The average time required for multi-task model prediction across the test dataset was 1.82 seconds, and the average time required for single-task model prediction across the test dataset was 0.93 seconds. 

MAPE was first computed for all of the supine patients’ ST and MT models dose predictions (Table 1). The average MAPE for supine patients’ predicted dose distributions was 1.1784 ± 0.3635% for the MT model and 1.2725 ± 0.3415% for the ST models. Furthermore, each 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 13 

modality’s average MAPE was lower for the MT model than the ST models. Then, MAPE was computed for the prone patients’ ST and MT models’ GP dose predictions, yielding average MAPE of 0.8029 ± 0.1490% for the MT model and 1.1029 ± 0.3552% for the GP ST model (Table 2). Therefore, the overall MAPE across the dataset was 1.1033 ± 0.3627% and 1.2386 ± 0.3872% for the MT and ST models respectively. The two-tailed Wilcoxon signed-rank test was performed with these 24 paired MAPE values, resulting in p-value = 0.0003 and 95% confidence interval = [−0.0865, −0.0712]. The average gamma passing rates for the MT and ST models’ dose predictions were 76.59 ± 13.74% and 75.18 ± 13.44%, respectively. 

Detailed PTV and OAR metrics were calculated for each ground truth dose distribution and predicted dose distribution. These metrics were averaged across the entire test dataset in absolute values, as all treatment plans were created with the same PTV prescription dose and OAR sparing goals (Table 3). Overall, metrics extracted from the MT and ST models dose predictions aligned well with the ground truth dose distribution metrics. 

Figure 5 offers an example of dose distributions for a supine patient’s CK treatment plan overlaid on their CT. The PTV is also displayed, segmented in red. The ground truth dose, or the original planned dose, is displayed in the top row. The ST model dose prediction follows in the middle row, which is then followed by MT model dose prediction in the bottom row. Qualitatively, the ST model’s dose prediction demonstrates excessive low dose prediction as compared to the MT model’s dose prediction. This is further evidenced in a comparison of the corresponding DVHs (Figure 6). 

An additional visual example of dose distributions, this time for a prone patient’s GP treatment plan, is shown in Figure 7. The top, middle and bottom rows display the ground truth dose, ST model predicted dose, and MT model predicted dose. In this case, the ST model’s predicted dose distribution shows excessive high dose on the lateral side of the ipsilateral breast. The dose differences between ground truth dose and predicted dose are displayed for this case in Figure 8. 

Finally, Figure 9 demonstrates a multi-modality comparison for one of the supine patients from the test dataset. From left to right, dose distributions are shown for the label/ground truth treatment plan, the ST model prediction, and the MT model prediction. From top to bottom, the CK, Ethos, TB, and Unity dose distributions are displayed. The MAPE for this patient’s dose distributions was lower in the MT predictions than that of the ST predictions for CK, Ethos, and Unity. 

## **IV. DISCUSSION** 

In this study, we compared the performance of modality-specific single-task (ST) dose prediction models against a unified multi-modality multi-task (MT) model. Across the test dataset, the ST models demonstrated a combined mean absolute percent error (MAPE) of 1.2386 ± 0.3872%, while the MT model achieved a lower MAPE of 1.1033 ± 0.3627%. The difference in performance is statistically significant, as indicated by the two-tailed Wilcoxon signed-rank test (p-value = 0.0003). 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 14 

In further assessment of performance variations among different treatment modalities, we observed that GP exhibited the largest difference in performance, whereas TB showed the smallest performance gap. Given that each model only trained with 6 GP samples versus 20 TB samples (and 10 samples for all other modalities), this led us to infer that the performance of ST models for their respective trained modalities could potentially approach that of the MT model as the number of samples increases. Notably, the MT model’s advantage may be attributable to its shared encoder and decoder layers because their weights are tuned with a larger effective sample size with inter-modality training data. 

While it is intriguing that the MT model demonstrated lower MAPE across the test dataset compared to the ST models, practical application of this work may only require that the MT model is non-inferior to the ST model method. The strength of this multi-task framework lies in its shared features, flexibility, and practicality, so it may be the most desirable approach if it does not come at a cost. Sharing encoder and decoder layers may aid prediction generalizability and mitigation of task-specific overfitting. The flexibility of the MT model allows for straightforward fine-tuning for other dose-related tasks, and the ease of adding or removing task-specific layers aligns with the evolving needs of a clinic. The practicality of the MT model as a “one-size-fits-all” solution can greatly simplify model management and deployment within clinical settings. 

We encountered a few challenges inherent in multi-task learning frameworks. At the onset of the study, we initiated our hyperparameter optimization process using a single-task model selected at random. This hyperparameter search utilized its own distinct dataset split and a unique random seed to ensure independence from the final model training. The optimal hyperparameters were then uniformly applied across all models, including the multi-task framework, with adjustments made only when necessitated by the multi-task model’s unique requirements. While our intention was to perform a fair comparison, this approach has limitations, such as the potential risk that the multi-task model’s capacity might not be fully optimized to address the increased complexity and diversity of tasks it is intended to handle. This limitation could also extend to the other single-task modalities that were not specifically optimized. 

This study did not make architecture or hyperparameter modifications on a per-modality basis, nor did we implement custom changes in model parameters during training on a per-modality basis. There is potential for improvement in model performance through customization of each individual task’s layer(s), such as adjusting the number of convolutions for a specific task. Furthermore, optimizing hyperparameters on a per-task basis, such as employing task-specific learning rates, could lead to further improvements. Lastly, modification of our freezing scheme for model weights during training may also improve the multi-task model performance. Our current freezing scheme was devised to mitigate the potential for task-specific overfitting, but it has some degree of arbitrariness and may be subject to further refinement in future works. 

Outside of the challenges specific to multi-task learning, we addressed challenges due to computational limitations by uniformly resizing input data to 5mm<sup>3</sup> spatial resolution. This resolution may contribute to both models’ under-prediction of PTV minimum dose and 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Page 15 

Maniscalco et al. 

over-prediction of PTV maximum dose, as this constraints both models’ capacity to discern fine details between the body surface and maximum skin depth of approximately 5mm. 

In part of the model performance evaluation, we calculated MAPE using the prescription dose as the denominator rather than voxel-wise dose values. This approach reduces the influence of minor absolute dose discrepancies in low dose regions on the error metric, offering a more clinically relevant assessment of model accuracy. For instance, consider a scenario in which the original treatment plan specifies 1 Gy for a specific voxel, but the model predicts a stylistically different plan with 0.5 Gy in that voxel. If such discrepancies occur across multiple voxels, they could significantly distort a voxel-wise MAPE and obscure the clinical significance of the metric. However, we recognize that while this metric provides a clinically relevant overview, it should not be used as an exclusive measure for performance evaluation and should be considered in conjunction with the other results presented in this study. 

As an additional part of model performance evaluation, we calculated gamma passing rates. However, we believe it is challenging to interpret the significance of this metric in application to dose prediction. In patient-specific quality assurance, it is common to observe high gamma passing rates because it is evaluating the deliverability of an already calculated treatment plan. On the other hand, our model does not receive input regarding beam arrangement, physician-specific dosimetric goals, or planner-specific variation, and therefore lacks stylistic knowledge that would be required to achieve higher gamma passing rates. In addition, deep-learning model with stylistic inputs may still yield dose predictions with inconsistent gamma passing rates due to noise, as demonstrated in the simulations performed by Chen et al. that evaluated the impact of Gaussian noise on gamma passing rates<sup>40</sup> . 

We acknowledge an additional limitation related to this unique and relatively small dataset. As our dataset was limited in size, it may be possible to enhance model performance across a broader population with larger training and validation subsets. For those seeking to apply this work, it may be challenging to acquire such a dataset if they only have access to a limited number of distinct radiation therapy machines and/or limited staffing that constrains their ability to generate numerous treatment plans for model training. Future strategies to address this might include the establishment of an open-source database and/or the utilization of commercial dose prediction software. Alternatively, clinics with relatively few unique machines could explore multi-task dose prediction for a variety of treatment planning techniques available on those machines. 

While the multi-modality dose predictor offers advantages in terms of quantifying dosimetric outcomes, it is important to recognize that certain modalities offer non-dosimetric benefits that are not captured by this model. For example, the Ethos and Unity models both offer adaptive treatment capabilities that allow for daily changes based on patient anatomy, but the Unity model is augmented with MRI capability for enhanced soft tissue visualization. It is crucial to consider each modality’s unique benefits alongside their dosimetric outcomes, as they may affect a treatment’s overall effectiveness. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Page 16 

Maniscalco et al. 

Our study has effectively demonstrated that multi-task learning can achieve strong performance, even when working with a relatively small dataset. The ability of a multi-task network to serve as a base model for future related tasks has significant implications. It allows for efficient and practical approaches to clinical needs, such as the addition of a dose prediction task for a new modality. Fine-tuning for new tasks with respect to the pre-trained modalities, such as treatment beam angle optimization on a per-modality basis, may outperform training from scratch. This architecture could also be extended to predict dose distributions for different intra-modality techniques, such as 3D conformal versus intensity-modulated radiation therapy dose distributions. An important feature of this model is its modularity, as task-specific layers have independent weights and can be removed without downstream consequences when necessary. Furthermore, we posit that a MT model may be pre-trained on a vast collection of patient data, and modality-specific/task-specific layers can be fine-tuned with smaller datasets. 

Beyond technical merits, the multi-modality dose prediction model has valuable applications for many individuals in radiation therapy. Patients, for instance, would be able to make an informed decision in the selection of their preferred dose delivery modality. Their decisionmaking process may weigh various factors such as the predicted dosimetric results, earliest appointment availability, lowest financial burden, and/or ideal geographical proximity. For example, one patient may opt to receive treatment using a modality that provides the best dosimetric outcome regardless of the clinic’s location or cost-effectiveness. On the other hand, another patient may prefer a modality that is predicted to yield an acceptable albeit slightly inferior dosimetric outcome because that modality is conveniently near their home and offers earlier appointment availability. 

Physicians may also derive benefit from the multi-modality dose predictor as a quantitative tool. A physician may have numerous unique protocols in mind for defining target volume extent, target volume margins, and/or organ sparing preferences, so this tool can enable quantitative analysis of the downstream dosimetric differences. Physicians can also leverage the quantitative data provided by the model in cost-benefit analyses to determine the most suitable modality on a patient-specific basis, taking into consideration the dosimetric differences and the patient’s financial situation. This quantitative data can also aid in obtaining treatment authorization from patients’ health insurance companies. 

The benefit to a clinic stems from optimization of resource allocation, increased patient referrals, and/or justification of machine procurement. In terms of resource allocation, clinics would be capable of allocating patients towards underutilized modalities if the treatment is dosimetrically non-inferior as compared to other heavily utilized modalities. Additionally, if this multi-modality dose predictor is shared, physicians from clinics with limited resources can refer patients to advanced radiation centers if they stand to benefit from treatment via more cutting-edge modalities. Lastly, for clinics interested in acquiring new machine(s), the model can provide quantitative justification for the purchase(s). By utilizing public training datasets or a public pre-trained multi-modality model, clinics can predict dosimetric differences among their own patients based on their current modalities and any other modalities of interest to aid in strategic decision-making. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 17 

With our multi-task dose prediction model framework, future works will concentrate on optimizing and broadening the network architecture. We aim to incorporate additional dose prediction tasks, such as prone position dose prediction for a breast patient that may only have data for the supine position, as this could offer significant clinical benefit. We are also interested in introducing new related tasks, such as beam angle selection/prediction, beam fluence prediction, and/or multi-leaf collimator leaf position prediction for volumetric modulated arc therapy. As an example of a task unrelated to dose prediction, we have also considered deformable image registration (DIR)-based organ segmentation for translation of segmented volumes between different scans of the same patient. DIR-based organ segmentation may be clinically beneficial as a tool to translate segmented volumes on an intra-patient basis, such as translation of contoured volumes from a free-breathing scan to a breath-hold scan. On a broader scale, we intend to explore the extension of this multi-task framework to various disease sites, enhancing the model’s generalizability and its overall impact within radiation therapy. 

## **V. CONCLUSION** 

The findings from our study demonstrate the efficacy of a multi-task learning framework in predicting radiation therapy dose distributions across different dose delivery modalities for patients undergoing accelerated partial breast irradiation. We trained a multi-task deep learning model that predicts dose distributions for these modalities simultaneously, and compared the performance of this multi-task model to that of multiple single-task models which each predict dose for a single modality’s dose distribution. The overall mean absolute percent error of the multi-task model across the test dataset was lower than that of the singletask models, and the difference was found to be statistically significant. This performance boost likely stems from the shared encoder and decoder layers within the multi-task model, which trained across a broader variety of patient data, potentially enhancing the model’s generalizability and mitigating the risk of task-specific overfitting during training. 

This multi-task framework can serve as a quantitative tool to aid patients, physicians, and clinics in making more informed decisions throughout the radiation therapy workflow. Patients, for example, may consider the dosimetric benefit of an advanced modality versus its financial cost, treatment position and treatment time. On the other hand, physicians gain the ability to see dose distributions for varying target margins and modalities, allowing for greater customization of treatment planning on a patient-specific basis. Additionally, clinics may use this tool to inform resource allocation and attract new patients with the ability to customize their treatment planning process. 

Lastly, the multi-task framework can be beneficial in clinical deployment as it can perform numerous related tasks simultaneously without performance detriment. The combination of numerous single-task models into a multi-task model can reduce overhead in the future development and management of deep learning models. Furthermore, the model’s architecture may be beneficial for modular adjustments, as feature maps in the encoder and decoder may be less task-specific than that of a single-task model. The removal or addition of task-specific layers in the future would be relatively straightforward as clinical needs evolve over time. Additionally, our novel approach to handling model input data 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 18 

may empower modality-agnostic dose prediction, as it does not rely on disease-site specific organs at risk as input data like most other approaches require. In future works, we anticipate expanding this framework to a site-agnostic dose predictor and expanding its utility to other closely related tasks. 

## **ACKNOWLEDGEMENTS** 

This study is supported by NIH grants R01CA237269, R01CA254377, and R01CA258987. 

## **Note regarding data availability:** 

Authors will share data upon request to the corresponding author. 

## **VII. REFERENCES** 

1. Rose CM, Recht A. Accelerated partial-breast irradiation (APBI): let’s give it a good test. International journal of radiation oncology, biology, physics. 2003;57(5):1217–1218. [PubMed: 14630254] 

2. Kuerer HM, Julian TB, Strom EA, et al. Accelerated partial breast irradiation after conservative surgery for breast cancer. Annals of surgery. 2004;239(3):338. [PubMed: 15075650] 

3. Shah C, Badiyan S, Ben Wilkinson J, et al. Treatment efficacy with accelerated partial breast irradiation (APBI): final analysis of the American Society of Breast Surgeons MammoSite<sup>®</sup> breast brachytherapy registry trial. Annals of surgical oncology. 2013;20:3279–3285. [PubMed: 23975302] 

4. Kirby AM. Updated ASTRO guidelines on accelerated partial breast irradiation (APBI): to whom can we offer APBI outside a clinical trial? The British Journal of Radiology. 2018;91(1085):20170565. [PubMed: 29513031] 

5. Hickey BE, Lehman M. Partial breast irradiation versus whole breast radiotherapy for early breast cancer. Cochrane Database of Systematic Reviews. 2021. (8). 

6. Goldberg M, Whelan TJ. Accelerated partial breast irradiation (APBI): where are we now? Current Breast Cancer Reports. 2020;12:275–284. [PubMed: 33101597] 

7. Whelan TJ, Julian JA, Berrang TS, et al. External beam accelerated partial breast irradiation versus whole breast irradiation after breast conserving surgery in women with ductal carcinoma in situ and node-negative breast cancer (RAPID): a randomised controlled trial. The Lancet. 2019;394(10215):2165–2172. 

8. Meattini I, Marrazzo L, Saieva C, et al. Accelerated partial-breast irradiation compared with wholebreast irradiation for early breast cancer: Long-term results of the randomized phase III APBIIMRT-Florence trial. Journal of Clinical Oncology. 2020;38(35):4175–4183. [PubMed: 32840419] 

9. Cho B Intensity-modulated radiation therapy: a review with a physics perspective. Radiation oncology journal. 2018;36(1):1. [PubMed: 29621869] 

10. Wang G-Y, Zhu Q-Z, Zhu H-L, et al. Clinical performance evaluation of O-Ring Halcyon Linac: A real-world study. World Journal of Clinical Cases. 2022;10(22):7728. [PubMed: 36158510] 

11. Moazzezi M, Rose B, Kisling K, Moore KL, Ray X. Prospects for daily online adaptive radiotherapy via ethos for prostate cancer patients without nodal involvement using unedited CBCT auto ‐ segmentation. Journal of applied clinical medical physics. 2021;22(10):82–93. 

12. Raaijmakers AJ, Raaymakers BW, Lagendijk JJ. Integrating a MRI scanner with a 6 MV radiotherapy accelerator: dose increase at tissue–air interfaces in a lateral magnetic field due to returning electrons. Physics in Medicine & Biology. 2005;50(7):1363. [PubMed: 15798329] 

13. Ding S, Li Y, Liu H, et al. Comparison of intensity modulated radiotherapy treatment plans between 1.5 T MR-linac and conventional linac. Technology in Cancer Research & Treatment. 2021;20:1533033820985871. [PubMed: 33472549] 

14. Kurup G CyberKnife: A new paradigm in radiotherapy. Journal of medical physics/Association of Medical Physicists of India. 2010;35(2):63. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 19 

15. Mutaf YD, Zhang J, Yu CX, et al. Dosimetric and geometric evaluation of a novel stereotactic radiotherapy device for breast cancer: the GammaPod<sup>™</sup> . Medical Physics. 2013;40(4):041722. [PubMed: 23556892] 

16. Gallio E, Giglioli FR, Girardi A, et al. Evaluation of a commercial automatic treatment planning system for liver stereotactic body radiation therapy treatments. Physica Medica. 2018;46:153–159. [PubMed: 29519402] 

17. Tsuruta Y, Nakata M, Nakamura M, et al. Dosimetric comparison of Acuros XB, AAA, and XVMC in stereotactic body radiotherapy for lung cancer. Medical physics. 2014;41(8Part1):081715. [PubMed: 25086525] 

18. Moran JM, Ben-David MA, Marsh RB, et al. Accelerated partial breast irradiation: what is dosimetric effect of advanced technology approaches? International Journal of Radiation Oncology* Biology* Physics. 2009;75(1):294–301. [PubMed: 19540076] 

19. Qiu J-J, Chang Z, Horton JK, Wu Q-RJ, Yoo S, Yin F-F. Dosimetric comparison of 3D conformal, IMRT, and V-MAT techniques for accelerated partial-breast irradiation (APBI). Medical Dosimetry. 2014;39(2):152–158. [PubMed: 24480375] 

20. Chao KC, Majhail N, Huang C-j, et al. Intensity-modulated radiation therapy reduces late salivary toxicity without compromising tumor control in patients with oropharyngeal carcinoma: a comparison with conventional techniques. Radiotherapy and oncology. 2001;61(3):275–280. [PubMed: 11730997] 

21. Holt A, Van Gestel D, Arends MP, et al. Multi-institutional comparison of volumetric modulated arc therapy vs. intensity-modulated radiation therapy for head-and-neck cancer: a planning study. Radiation oncology. 2013;8:1–11. [PubMed: 23280007] 

22. Pigorsch SU, Kampfer S, Oechsner M, et al. Report on planning comparison of VMAT, IMRT and helical tomotherapy for the ESCALOX-trial pre-study. Radiation Oncology. 2020;15:1–10. 

23. Nachbar M, Mönnich D, Kalwa P, Zips D, Thorwarth D, Gani C. Comparison of treatment plans for a high-field MRI-linac and a conventional linac for esophageal cancer. Strahlentherapie und Onkologie. 2019;195(4):327–334. [PubMed: 30361744] 

24. Lin Y-W, Lin K-H, Ho H-W, et al. Treatment plan comparison between stereotactic body radiation therapy techniques for prostate cancer: non-isocentric CyberKnife versus isocentric RapidArc. Physica Medica. 2014;30(6):654–661. [PubMed: 24726212] 

25. Zhang S, Yang R, Wang X. Dosimetric quality and delivery efficiency of robotic radiosurgery for brain metastases: Comparison with C ‐ arm linear accelerator based plans. Journal of Applied Clinical Medical Physics. 2019;20(11):104–110. 

26. Stroubinis T, Psarras M, Zygogianni A, Protopapa M, Kouloulias V, Platoni K. Craniospinal Irradiation: A Dosimetric Comparison Between O-Ring Linac and Conventional C-arm Linac. Advances in Radiation Oncology. 2023;8(2):101139. [PubMed: 36636383] 

27. Li F, Park J, Lalonde R, et al. Is Halcyon feasible for single thoracic or lumbar vertebral segment SBRT? Journal of Applied Clinical Medical Physics. 2022;23(1):e13458. [PubMed: 34845817] 

28. Cilla S, Ianiro A, Romano C, et al. Template-based automation of treatment planning in advanced radiotherapy: a comprehensive dosimetric and clinical evaluation. Scientific reports. 2020;10(1):423. [PubMed: 31949178] 

29. Shiraishi S, Moore KL. Knowledge ‐ based prediction of three ‐ dimensional dose distributions for external beam radiotherapy. Medical physics. 2016;43(1):378–387. [PubMed: 26745931] 

30. Campbell WG, Miften M, Olsen L, et al. Neural network dose models for knowledge ‐ based planning in pancreatic SBRT. Medical physics. 2017;44(12):6148–6158. [PubMed: 28994459] 

31. Nguyen D, Jia X, Sher D, et al. 3D radiotherapy dose prediction on head and neck cancer patients with a hierarchically densely connected U-net deep learning architecture. Physics in medicine & Biology. 2019;64(6):065020. [PubMed: 30703760] 

32. Bo Z, Jianghong X, Chongyang C, et al. Multi-constraint generative adversarial network for dose prediction in radiotherapy. FREE Full text Med Image Anal. 2022;7(10.1016). 

33. Li F, Niu S, Han Y, Zhang Y, Dong Z, Zhu J. Multi-stage framework with difficulty-aware learning for progressive dose prediction. Biomedical Signal Processing and Control. 2023;82:104541. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 20 

34. Maniscalco A, Liang X, Lin MH, Jiang S, Nguyen D. Intentional deep overfit learning for patient ‐ specific dose predictions in adaptive radiotherapy. Medical Physics. 2023;50(9):5354– 5363. [PubMed: 37459122] 

35. Wen L, Xiao J, Tan S, et al. A Transformer-Embedded Multi-Task Model for Dose Distribution Prediction. International Journal of Neural Systems. 2023.2350043–2350043. [PubMed: 37420338] 

36. Jiao Z, Peng X, Xiao J, Wu X, Zhou J, Wang Y. Mask-Free Radiotherapy Dose Prediction via Multi-Task Learning. Paper presented at: 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) 2022. 

37. Smith LN, Topin N. Super-convergence: Very fast training of neural networks using large learning rates. Paper presented at: Artificial intelligence and machine learning for multi-domain operations applications 2019. 

38. Biggs S, Jennings M, Swerdloff S, et al. PyMedPhys: A community effort to develop an open, Python-based standard library for medical physics applications. Journal of Open Source Software. 2022;7(78):4555. 

39. Paddick I A simple scoring ratio to index the conformity of radiosurgical treatment plans. Journal of neurosurgery. 2000;93(supplement_3):219–222. [PubMed: 11143252] 

40. Chen M, Mo X, Parnell D, Olivera G, Galmarini D, Lu W. Impact of image noise on gamma index calculation. Paper presented at: Journal of Physics: Conference Series 2014. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 21 



#### **Figure 1:** 

A planning target volume (PTV)-based Euclidean distance array is overlaid on a patient’s planning computed tomography (CT) scan. A value of 1.0 represents the PTV’s location, which is segmented in blue. As the distance array voxel values decrease below 1.0 and approach 0.0, Euclidean distance from the PTV is increasing. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 



<!-- Start of picture text -->
: Bs<br><!-- End of picture text -->

**Figure 2:** The goal array prior to normalization. The normalized goal array is used as input data rather than organ at risk (OAR)-specific arrays, as it compresses OAR spatial information into a single array and augments this array with dosimetric planning objectives. For example, all voxels where the planning target volume (PTV) is present are assigned to a maximum value of 32.5 Gy. On the other hand, the most distal 90% of ipsilateral lung is assigned to a value of 9 Gy, conveying the ipsilateral lung dosimetric objective which desires less than 10% of its volume to receive 9 Gy or greater. The remaining unspecified portion of ipsilateral lung is assigned to an arbitrarily selected dosimetric goal of max dose less than 31 Gy. 

Maniscalco et al. Page 22 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 23 



<!-- Start of picture text -->
=, SD)<br>Shared Encoder | Shared Decoder ‘Convolutional Layers Multi-task Outputs<br>) Ly | {CK Dose<br>> | ¢ (Wis,  (ee Lesa \__ UnityPredictionDose }<br>Input Data [4,72,56,56,32] [4,72,56,56,32]<br>[Input Data) Ethos Dose<br>Ch P ; Bassam Lea \__ Prediction<br>PTV — —> | | 14,144,28,28,16] [4,144,28,28,16]<br>Distanceistance AArray - | TB Dose |<br>-<br>Goal Array TB Conv Layer aoe<br>[4,4,112,112,64] ( [4,288,14,14,8) [4,288,14,14,8] a ca<br>wsrarza | |{RararTa| | u Prediction<br>aaa J U a —— J ereTask<br>a, (ee<br><!-- End of picture text -->

#### **Figure 3:** 

Our convolutional neural network architecture. The encoder and decoder paths are similar to the U-Net architecture. The input data is made up of 4 channels: computed tomography (CT) scan, planning target volume (PTV), a PTV distance array, and a goal array. An input data patch of size 112×112×64 is randomly selected and sent to the network’s shared encoder, and then shared decoder. There are skip connections between the encoder and decoder at each level with an option to bypass the skip connection. The feature maps are distributed to each of six independent convolutional layers. Five convolutional layers are tasked with dose prediction for one radiation therapy (RT) dose delivery modality such as CyberKnife (CK), Unity, Ethos, TrueBeam (TB), and GammaPod (GP). These five layers all receive a copy of the feature maps generated with skip connections. On the other hand, the final convolutional layer is referred to as a regularization task (R.T.) convolutional layer. This layer receives feature maps that bypassed skip connections and predicts doses for all 5 modalities. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 24 



<!-- Start of picture text -->
a,<br>|4,36,112,112,64) 14,36,112,112,64]<br>EE<br>‘Input Data) Input Data ( [4,72,56,56,32] } [4,72,56,56,32]<br>DistancePTVor > F r —_ (14.144,26y 28.18 [4,144,28,28,16] —_ Convolutionalase —_ Dose{ Single PredictionModality +<br>Goal Array Array ; ¥| .<br>[4,4,112,112,64] | (4,288,14,14,8] (4,288,14,14,8]<br>a<br>((4,576,7,7.4] j + [4,576,7,7,4]<br>a,<br><!-- End of picture text -->

**Figure 4:** 

A simplified, single-task version of our convolutional neural network architecture. Each model trained with this architecture is modality-specific, and therefore can only predict dose for the trained modality. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Page 25 

Maniscalco et al. 



#### **Figure 5:** 

A supine test patient’s CyberKnife (CK) dose distributions are overlaid on their computed tomography (CT) scan. The dose display range is set from 10% of prescription value to 110% of prescription value. The red segmented structure is the planning target volume (PTV). The top row shows the original plan’s dose distribution, otherwise referred to as the ground truth. The middle row and bottom row display the single-task (ST) and multitask (MT) models’ CK dose predictions, respectively. Relative to the ground truth, the ST model’s CK dose prediction shows excessive low dose on the medial side of the ipsilateral breast. The mean absolute percent error (MAPE) for the MT model’s dose prediction in this case was 1.0336%, versus 1.2299% for the ST model’s dose prediction. However, the gamma passing rates for these MT and ST dose distributions were 62.45% and 82.21%, respectively. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Page 26 

Maniscalco et al. 



<!-- Start of picture text -->
K: Ground Trut olid) vs MT Prediction (Dotted) vs ST Prediction (Dashed)<br>1<br>100% | 1i<br>i<br>1<br>80% aI z<br>°i<br>o q<br>= 60% it— Hh :<br>eco i<br>oq<br>iSLHt i<br>vy3 40% yt = rt |i E<br>iS i' t Hl1<br>] \ 1<br>Ht R yt<br>' H 4!1<br>20%<br># 7] \<br>el! \<br>ie! \<br>mY \<br>nyt \<br>th ENS ee | 1<br>OGy 10Gy 20Gy 30Gy OGy 10Gy 20Gy 30Gy<br>Dose Dose<br>PTV_3000_5 ADAPT — Heart GTV_30005 — Ribs<br>— Breast left — Lung right Body —— Spinal canal<br>Breast right —— Skin Lung left<br><!-- End of picture text -->

#### **Figure 6:** 

Dose volume histograms for a single patient’s CyberKnife (CK) dose distributions. The planning target volume (PTV), gross tumor volume (GTV), and organs at risk are displayed, split into two DVHs to better distinguish individual organ doses. Solid lines are the ground truth as calculated from the original plan’s dose. Dotted lines correspond to the multi-task (MT) model’s dose prediction, and dashed lines correspond to the single-task (ST) model’s dose prediction. In this figure, we show that the MT model’s dose prediction demonstrates greater accuracy in the low dose region for the right breast than the ST model’s dose prediction. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 27 



<!-- Start of picture text -->
iGammaPod Planned Dose o<br>. i<br>“~ Sa ‘ > ;<br>Single-Task Model Prediction L 7 al<br>, & i ,<br>ey Py .<br>Multi-Task Model Prediction L 7 nd<br>Ne<br><!-- End of picture text -->

#### **Figure 7:** 

A prone test patient’s GammaPod (GP) dose distributions are overlaid on their computed tomography (CT) scan, with a dose display range from 10% - 110% of the planning target volume (PTV) prescription dose. The PTV is segmented in red. The top, middle, and bottom rows display the original plan’s dose, single-task (ST) model’s predicted dose, and the multi-task (MT) model’s predicted dose, respectively. The ST model predicted higher dose on the lateral side of the ipsilateral breast, relative to the original plan’s dose. The mean absolute percent error (MAPE) for the MT model dose prediction is 0.7190%, compared to 0.7179% for the ST model dose prediction. Additionally, the gamma passing rate is 78.16% and 73.54% for the MT and ST predictions, respectively. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 28 



<!-- Start of picture text -->
.<br>iGammaPod Planned Dose 5<br>~~ | .<br>: y<br>f ; =~ es y<br>eg > .<br>OT | cael<br>iSingle-Task Model Difference L ~ ad<br>i<br>“wT D<br>Multi-Task Model Difference L 7 al<br>!<br><!-- End of picture text -->

**Figure 8:** 

The ground truth, or original plan’s dose, is overlaid on the patient’s computed tomography scan (CT) in the top row. Dose difference maps are overlaid on CT in the middle and bottom rows. Dose difference maps are equal to the voxel-wise percent difference between the ground truth dose distribution and a model’s predicted dose distribution. The dose difference map display range is relative, from −50% to +50% difference for a single voxel. The middle row and bottom row have dose difference maps for the single-task (ST) model and multi-task (MT) models’ dose predictions, respectively. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 29 



<!-- Start of picture text -->
K Planned Dose K ST Prediction K MT Prediction<br>= ~ aN<br>a 4 , ~ ¥ & sa<br>™ = * Re 4<br>Ethos Planned Dose Ethos ST Prediction Ethos MT Prediction<br>oo 4 y S<br>, | |<br>. ys = Jj Fie)<br>5 n ae if<br>_————— =” Ss =<br>TB Planned Dose TB ST Prediction TB MT Prediction<br>- ay PS A A.<br>— ae A pt ye / Le ia<br>Unity Planned Dose Unity ST Prediction Unity MT Prediction<br>.<br>= a SN = a<br>; eal a aS % iat<br><!-- End of picture text -->

#### **Figure 9:** 

Multi-modality comparison for one supine test patient. Dose distributions are overlaid on the computed tomography (CT) image and range from 10% - 110% of the planning target volume (PTV) prescription dose, with the PTV segmented in red. From top to bottom, CyberKnife (CK), Ethos, TrueBeam(TB), and Unity dose distributions are displayed. From left to right, we show the ground truth label, the single-task (ST) model’s dose prediction, and the multi-task (MT) model’s dose prediction. The average mean absolute percent error 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 30 

(MAPE) for this patient’s ST dose predictions was 1.3719 ± 0.3238%, whereas the average MAPE for MT dose predictions was 1.2940 ± 0.2651%. 

Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 31 

#### **Table 1:** 

Results from the supine patient portion of the test dataset. This table displays the mean absolute percent error (MAPE) between the multi-task (MT) model’s predicted dose distributions and the ground truth dose distributions, and the MAPE between the single-task (ST) models’ predicted dose distributions and the ground truth dose distributions. The MAPE for TrueBeam is an average of error across its two dose predictions for the conventional PTV and adaptive PTV. 13 out of the 16 MAPE values above were lower for the MT model than the ST models. The MT model’s average MAPE across the supine patients’ plans within the test dataset is 1.1784 ± 0.3635%, whereas the ST models’ average MAPE is 1.2725 ± 0.3415%. 

|**Mean Absolute P**<br>**(MAP**<br>**across dose p**|**ercent Error**<br>**E)**<br>**redictions**|**CyberKnife**|**Unity**|**Ethos**|**TrueBeam**|
|---|---|---|---|---|---|
|**Ptit #1**|**Multi-Task**|1.0336%|1.4233%|0.8064%|0.4412%|
|**aen**|**Single-Task**|1.2299%|1.3546%|0.8344%|0.4562%|
|**Pi #2**|**Multi-Task**|1.4155%|1.7968%|1.4242%|1.2060%|
|**atent**|**Single-Task**|1.8499%|1.8418%|1.4476%|1.2772%|
|**i #**|**Multi-Task**|1.4773%|1.1188%|0.8770%|0.6580%|
|**Patent 3**|**Single-Task**|1.5192%|1.2154%|1.2084%|0.6374%|
||**Multi-Task**|1.3782%|1.5353%|1.3465%|0.9159%|
|**Patient #4**|**Single-Task**|1.5201%|1.6304%|1.4330%|0.9040%|
|**Modality**|**Multi-Task**|1.3262 ± 0.1993%|1.4685 ± 0.2808%|1.1135 ± 0.3168%|0.8053 ± 0.3302%|
|**Patient Average**|**Single-Task**|1.5298 ± 0.2534%|1.5106 ± 0.2802%|1.2308 ± 0.2861%|0.8187 ± 0.3567%|



Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 32 

#### **Table 2:** 

Results from the prone patient portion of the test dataset, which only have GammaPod treatment plans. Mean absolute percent error (MAPE) is calculated between each predicted dose distribution and the ground truth dose distribution. The multi-task (MT) and single-task (ST) models average MAPE for across all GammaPod plans in test dataset is 0.8029 ± 0.1490% and 1.1029 ± 0.3552%, respectively. 

|**Mean Absolute Per**<br>**across dose**|**cent Error (MAPE)**<br>**predictions**|**Patient #5**|**Patient #6**|**Patient #7**|**Patient #8**|**GammaPod Patient Average**|
|---|---|---|---|---|---|---|
|**GammaPod**|**Multi-Task**<br>**Single-Task**|1.0199%<br>1.2512%|0.6938%<br>0.9211%|0.7788%<br>1.5215%|0.7190%<br>0.7179%|0.8029 ± 0.1490%<br>1.1029 ± 0.3552%|



Med Phys. Author manuscript; available in PMC 2025 June 01. 

Maniscalco et al. 

Page 33 

#### **Table 3:** 

Comparison between the average of all ground truth dose distributions, multi-task model predicted dose distributions, and single-task model predicted dose distributions. Structures, such as the planning target volume (PTV), are compared using dose metrics. The volume of PTV receiving 30 Gy (V30 Gy<sup>) is equivalent in all three dose distributions because the predicted dose distributions were scaled to match</sup> that ground truth value. As a consequence of utilizing coarse 5mm<sup>3</sup> voxel spacing for model input and dose prediction, some voxel-wise metrics from the final upsampled prediction - such as PTV minimum dose - may be less precise than volume-based metrics like the aforementioned PTV V30 Gy. 

|**Structure, Metric**|**Ground Truth**|**Multi-Task Prediction**|**Single-Task Prediction**|
|---|---|---|---|
|PTV, Homogeneity Index|0.13 ± 0.03|0.20 ± 0.05|0.23 ± 0.06|
|PTV, Conformity Index|0.87 ± 0.04|0.81 ± 0.06|0.81 ± 0.08|
|PTV, Max Dose|33.69 ± 1.30 Gy|34.90 ± 1.22 Gy|35.49 ± 1.83 Gy|
|PTV, V30 Gy|91.99 ± 1.07 %|91.99 ± 1.07 %|91.99 ± 1.07 %|
|PTV, D99%|27.85 ± 0.82 Gy|26.03 ± 1.89 Gy|25.36 ± 1.87 Gy|
|PTV, Minimum Dose|20.48 ± 6.80 Gy|17.93 ± 6.45 Gy|16.97 ± 6.16 Gy|
|Body, Gradient Index|3.11 ± 0.45|3.14 ± 0.40|3.08 ± 0.46|
|Ipsilateral Lung, D5%|8.06 ± 6.00 Gy|8.45 ± 6.53 Gy|8.78 ± 6.14 Gy|
|Ipsilateral Lung, D10%|5.96 ± 4.59 Gy|6.09 ± 4.74 Gy|6.33 ± 4.50 Gy|
|Ipsilateral Lung, D20%|3.79 ± 2.93 Gy|3.89 ± 3.06 Gy|4.16 ± 3.11 Gy|
|Ipsilateral Lung, V9 Gy|5.07 ± 6.23 %|5.95 ± 6.95 %|5.93 ± 7.64 %|
|Ipsilateral Lung, V20 Gy|1.00 ± 2.32 %|1.17 ± 2.22 %|1.20 ± 2.36 %|
|Ipsilateral Lung, Mean Dose|2.24 ± 1.64 Gy|2.38 ± 1.86 Gy|2.49 ± 1.82 Gy|
|Contralateral Lung, V1.5 Gy|0.91 ± 1.94 %|0.18 ± 0.44 %|0.25 ± 0.87 %|
|Contralateral Lung, D5%|0.59 ± 0.52 Gy|0.41 ± 0.29 Gy|0.46 ± 0.36 Gy|
|Contralateral Lung, D10%|0.43 ± 0.37 Gy|0.31 ± 0.21 Gy|0.35 ± 0.27 Gy|
|Contralateral Lung, Mean Dose|0.20 ± 0.15 Gy|0.16 ± 0.10 Gy|0.16 ± 0.12 Gy|
|Ipsilateral Breast, D40%|8.48 ± 5.62 Gy|9.06 ± 5.50 Gy|8.97 ± 5.38 Gy|
|Ipsilateral Breast, D20%|19.85 ± 8.43 Gy|20.53 ± 7.83 Gy|20.51 ± 7.76 Gy|
|Ipsilateral Breast, V15 Gy|27.06 ± 11.83 %|28.28 ± 11.30 %|27.93 ± 10.92 %|
|Ipsilateral Breast, V30 Gy|11.41 ± 5.65 %|12.13 ± 5.92 %|12.11 ± 5.67 %|
|Contralateral Breast, Max Dose|0.80 ± 0.65 Gy|1.04 ± 0.95 Gy|0.85 ± 0.91 Gy|
|Heart, D5%|0.94 ± 0.75 Gy|0.96 ± 0.83 Gy|1.10 ± 1.01 Gy|
|Heart, D30%|0.44 ± 0.32 Gy|0.46 ± 0.38 Gy|0.49 ± 0.49 Gy|
|Heart, V1.5 Gy|2.82 ± 5.76 %|4.17 ± 7.78 %|6.04 ± 11.20 %|
|Heart, V7 Gy|0.00 ± 0.01 %|0.00 ± 0.00 %|0.01 ± 0.02 %|
|Heart, Mean Dose|0.38 ± 0.27 Gy|0.40 ± 0.31 Gy|0.43 ± 0.40 Gy|
|Skin, Max Dose|32.79 ± 0.98 Gy|34.16 ± 1.45 Gy|34.30 ± 1.40 Gy|



Med Phys. Author manuscript; available in PMC 2025 June 01. 

