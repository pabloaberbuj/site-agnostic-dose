medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

# **Generalizable Deep Learning Framework for Radiotherapy Dose Prediction Across Cancer Sites, Prescriptions and Treatment Modalities** 

Ho-hsin Chang<sup>1,2</sup> , Rex Alexander Cardan<sup>2</sup> , Ritish Nedunoori<sup>2</sup> , John B. Fiveash<sup>2</sup> , Richard A. Popple<sup>2</sup> , Sandeep Bodduluri<sup>1</sup> , Dennis Stanley<sup>2</sup> , Joseph Harms<sup>3</sup> , Carlos E. Cardenas<sup>2</sup> 

1Department of Biomedical Engineering, University of Alabama at Birmingham, Room 361, 1075 13th Street South, Birmingham, AL, USA 35294 

2Department of Radiation Oncology, University of Alabama at Birmingham, 1700 6th Avenue South Birmingham, AL, USA 35233 

3Department of Radiation Oncology, Washington University, MO, USA 

**Corresponding Author:** Carlos E. Cardenas, <u>cecardenas@uabmc.edu</u> 

**Ethics statement:** This retrospective study was reviewed and approved by our institution’s Institutional Review Board (IRB�No.�300012175). All patient data were de�identified prior to analysis, and the requirement for informed consent was waived in accordance with IRB guidelines. 

**NOTE: This preprint reports new research that has not been certified by peer review and should not be used to guide clinical practice.** 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

# **Abstract** 

Optimizing radiotherapy dose distributions remain a resource-intensive bottleneck. Existing AI-based dose prediction methods often have limited generalizability because they rely on small, heterogeneous datasets. We present nnDoseNetv2, an autoconfigured, end-to-end framework for dose prediction across diverse disease sites (head and neck, prostate, breast, and lung), prescription levels (1.5–84 Gy), and treatment modalities (IMRT, VMAT, and 3D-CRT). By integrating machine-specific beam geometry with 3D structural information, the framework is designed to generalize across varied clinical scenarios. 

A single multi-site model was trained on 1,000 clinical plans. On sites seen during training, performance was comparable to specialized site-specific models. On unseen sites (liver and whole brain), the model outperformed site-specific models, with mean absolute errors of 2.46% and 6.97% of prescription, respectively. 

These results suggest that geometric awareness can bridge disparate anatomical domains while eliminating the need for site-specific model maintenance, providing a scalable and high-fidelity approach for personalized radiotherapy planning. 

# **1 Introduction** 

Radiotherapy (RT) treatment planning is a complex, iterative process that can take days to weeks. Planners begin by configuring the treatment planning system (TPS) with beam parameters (e.g. delivery machine, number of beams, beam angles, etc.) which strongly influence the final dose distribution. To reach a desirable plan, planners iteratively run TPS optimizations and adjust the planning goal to balance the trade-off between tumor coverage and sparing healthy tissue. Due to time and resource constraints, planners routinely face a compromise between plan quality and turnaround time. Therefore, a fast, automated deep learning (DL) dose prediction tool could reduce planning time from weeks to minutes by providing dose estimation and potential clinical goal for the RT planning. 

The use of DL tools to assist RT workflows is becoming increasingly accepted<sup>1–3</sup> . However, DL research for dose prediction has been dominated by site-specific models<sup>4–</sup> 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

29. These models are typically trained on homogeneous datasets with minimal variation in delivery machines, disease sites, or prescriptions. Such a fragmented data selection strategy increases the technical burden of model management; maintaining a distinct model for every clinical scenario is unsustainable for most centers, particularly for resource-limited clinics with constrained hardware and data access. Furthermore, this fragmentation hinders the generalizability and robustness of trained models by limiting both data diversity and the total scale of training data. Currently, the largest public dataset remains the 2020 OpenKBP challenge dataset<sup>30,31</sup> , which comprises 340 processed head and neck 9-field intensity modulated radiation therapy (IMRT) plans containing a primary 70 Gy target with optional targets of 63 Gy and 56 Gy. Therefore, a beam aware, multi-site training approach utilizing heterogeneous data is crucial to improve generalizability, robustness and scale of the model and further improve the advancing global oncology care. 

Our previous work, nnDoseNet, addressed the technical barriers of DL implementation by providing an off-the-shelf, auto-configuring framework. By automatically tailoring preprocessing and training pipelines to specific data distributions and available GPU memory, nnDoseNet enables clinics without high-performance computing (HPC) infrastructure or specialized programming expertise to develop robust models. This approach recognizes that clinical goals and planning strategies vary significantly across institutions; rather than providing a static, pre-trained model, this flexible framework allows clinics to train tailored models using their own data. By providing an end-to-end training pipeline that processes data directly from the DICOM format, the barrier to entry for AI adoption is significantly lowered. However, while nnDoseNet streamlined model management through this systemic pipeline, it was primarily developed using homogeneous datasets. It lacked the capacity to handle multi-prescription, cross-site, or multi-modality scenarios because the original framework could not effectively extract and incorporate beam geometry from the plan DICOM. 

While recent studies have incorporated beam geometry to validate cross-site performance, these approaches often lack publicly available code or a standardized, 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

end-to-end framework. This lack of open-resource tools prevents other institutions from adopting these tools. 

With the nnDoseNet<sup>5</sup> pipeline and a sufficiently large dataset, we hypothesize that a single model that handles multiple delivery machines, disease sites, and prescriptions, is feasible.  In this research, a multi-site model is trained with a large-scale, multi-site dataset (HN, prostate, breast, and lung) consisting of 1,000 clinically delivered RT plans at our institution, making this the first deep learning dose predictor trained on a multisite dataset of this scale. The multi-site model, trained with nnDoseNet, and tested on an independent set of 300 cases from seen and unseen disease site. Our results show that the multi-site model achieves comparable (and better in some cases) accuracy than site-specific models, demonstrating the feasibility of a general model without loss of performance. To our knowledge, this is the first demonstration of a deep learning model trained across such diverse RT planning scenarios. 

# **2 Materials and Methods 2.1 Large Scale Multi-Site Dataset** 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

An institutional dataset comprising 1,300 cases across six anatomical sites was utilized for this study, with all records anonymized via DICOMAnon™ Ultimate (Red Ion LLC). Each case includes a comprehensive suite of RT records in DICOM format, including planning CT images, structure sets (organs-at-risk and targets), clinically delivered treatment plans, and corresponding 3D dose distributions. The dataset reflects realworld clinical heterogeneity, consisting of a mixture of 3D conformal radiotherapy (3DCRT), IMRT, and volumetric modulated arc therapy (VMAT). 

To evaluate the model’s performance across diverse clinical scenarios, we incorporated 







<!-- Start of picture text -->
aid a gai ate. aH i<br><!-- End of picture text -->



<!-- Start of picture text -->
wei FE ee I :<br><!-- End of picture text -->

**_Figure 1 Characterization of dataset diversity across disease sites._** _Distribution of key radiotherapy parameters for the six disease sites included in this study. The head and neck (HN), prostate, breast, and lung cohorts served as seen sites for both model training and testing. The whole brain (WB) and liver cohorts were reserved as unseen sites for independent validation of model generalizability. The plots illustrate the wide variance in prescription doses, target counts, and treatment modalities, highlighting the heterogeneity required for a universal dose prediction framework. Notably, the training and testing distributions for the four seen sites are closely matched, demonstrating an even data split that maintains representative complexity across both cohorts._ 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

four seen disease sites (head and neck [HN], prostate, breast, and lung) for model training and initial testing, alongside two unseen sites (liver and whole brain) reserved exclusively for external validation. The training cohort consisted of 1,000 cases (250 per seen site), while the testing cohort comprised 300 cases (50 per seen site and 50 per unseen site). 

As illustrated in Figure 1, the training and testing subsets for these sites exhibit nearly identical distributions across all six clinical parameters (dose per fraction, total dose, etc). This distributional consistency ensures that the testing set is representative of the training data diversity, providing a reliable measure of model performance. The data complexity is further evidenced by the wide range of targets and prescriptions across sites: the prostate dataset includes 1–4 targets (18–84 Gy), breast includes 1–5 targets (2.5–50 Gy), lung includes 1–4 targets (1.5–66 Gy), and HN includes 1–4 targets (2–70 Gy). Variations in prescription and fractionation reflect diverse clinical intents, spanning boost phases, stereotactic treatments, conventional radiotherapy, and palliative regimens. 

# **2.2 Data Preparation** 

# **2.2.1 CT and RT DOSE DICOM Extraction** 

The data conversion pipeline is built upon the established nnDoseNet framework, utilizing an automated workflow to translate DICOM radiotherapy files into NIfTI volumes to standardize the spatial metadata for the DL pipeline. CT images are generated from CT DICOM files, and ground-truth dose maps are extracted from RTDOSE files. Anatomical structures are processed via the RTSTRUCT conversion module, which extracts, encodes, and separates targets and organs-at-risk (OARs) into corresponding input channels. For this project, the pipeline was configured to recognize 18 key anatomical structures across different disease sites: the body, spine, lungs, esophagus, heart, liver, bladder, rectum, femoral heads, bowel, thyroid, trachea, carina, main bronchi, mandible, brainstem, larynx, and parotids. 

For target and dose encoding, the framework employs a two labeling strategy in target designed to evaluate sensitivity to absolute magnitude versus relative distribution: 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

(1) Prescription-labeled (cGy): Target structures are encoded by assigning specific prescription values directly to each constituent voxel (e.g., 70 Gy is encoded as 7,000 cGy), with ground-truth dose maps maintained in absolute Gy. 

(2) Percentage-labeled (%): Both target masks and ground-truth dose maps are normalized to the maximum target prescription (0 to 100%). This approach is 



<!-- Start of picture text -->
Hl nnDose_dicom nnDose_raw inDose_preprocess nnDose_results nnDose_prediction |<br><!-- End of picture text -->



<!-- Start of picture text -->
: ‘son atjson plan ison Evaluation Report {aH<br>Hi fmage MSE H<br>: DICOMs Pingu | [models | ® !<br>H ee oe ‘ : # 105% H<br>: ) i ¢ + prescription, H<br>H p q ‘ Taggets | Danae Of image :<br>i 5 ‘ ¢ Target and OARs: H<br>Ht { REstruct +t ‘ t VaseDasr VagDag, DyBu ::<br>H ; H { f Dareae :<br>i 1 g i y Homogeneity, :<br>t , ~, _ ‘<br>i » REPlan « {Beam | Conformity :<br>H i yg '<br>H A ‘ g yo:<br>i ’ t t G H<br>H ) G ‘ u Difference Map H<br>' i H<br>:5) Rtdose |tpetabel i<br>Htossskd = H<br>i DICOMs = T Models | '<br>lentaorecrsonn----5, :<br><!-- End of picture text -->





**_Figure 2 Integrated nnDoseNet framework for automated, beam-aware dose prediction._** _The upgraded pipeline incorporates a module for extracting machinespecific beam geometry from RTPLAN files and integrating it directly into the data conversion process. nnDoseNet provides an end-to-end architecture that autoconfigure data preprocessing, systematic model training, and robust dose evaluation pipeline. The framework simplifies complex DL workflows by reducing model training to a single-command interface and enabling rapid architectural tuning via standardized JSON configuration files._ 

specifically designed to eliminate training bias toward high-dose prescriptions. 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

# **2.2.2 RTPLAN and Beam Geometry** 

The major difference between nnDoseNetv1 and nnDoseNetv2, a beam geometry extraction pipeline leveraging RTPLAN DICOM files is being utilized. By analyzing the isocenter coordinates, source-axis distance (SAD), and beam angles, each beam’s geometry was projected onto the CT volume and created a 3D beam geometry map or a distance-to-beam axis representation. For cross-section area shape, the field shape for beam projection is utilized in 3D-CRT plans and beams-eye-view (BEV) is utilized in IMRT and VMAT. This beam geometry map encodes the machine-specific beam arrangement and is used as an additional input compared to the previous version of nnDoseNet. This beam geometry extraction will be integrated into nnDoseNet as part of future work. 

# **2.2.3 Efficiency Improvement** 

To enhance computational throughput and reduce memory overhead, several architectural optimizations were implemented. First, the body contour—previously handled as an isolated channel—was consolidated directly into the OAR map, reducing the overall input dimensionality to a streamlined four-channel architecture. Second, all images were resampled to match the native resolution of the dose maps, avoiding the overhead of high-resolution CT grids. Finally, volumes were cropped to a region of interest (ROI) extending ±5 slices cranio-caudally beyond the target boundaries, focusing the model’s learning capacity on the primary irradiated regions while excluding distant anatomical structures that do not contribute to the dose distribution. 

# **2.2.4 Auto-configured Image Preprocess Pipeline** 

The same intensity preprocessing as nnU-Net<sup>32</sup> were applied, that the CT were clipped to the [5th, 95th] percentile range of foreground image intensities. The beam geometry map was normalized values to [0,1]. The target and OAR channels were used as label channels (i.e. integers, no intensity normalization applied). For dose maps, the data can be pre-processed in two formats: 1) dose values in cGy (prescription-labeled), and 2) normalized dose values by highest target prescription (percentage-labeled). 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

# **2.3 Training** 

# **2.3.1 Experiments of Training Single-site and Multi-site Models** 

Two experiments were conducted to train our model: a single-site experiment and a multi-site experiment (Figure 3). In the single-site experiment, four individual models were trained/validated on 250 plans from one site and tested on 50 held-out cases from that same site (total of 200 test cases across four training sites). In the multi-site experiment, models were trained/validated on 1000 plans (250 from each site combine **d** ) and tested it on the same 200 held-out cases (50 per site). 

Consistent with the nnU-Net<sup>32</sup> training framework, all nnDoseNet<sup>5</sup> models were trained using 5-fold cross-validation, meaning that training occurred using 200 plans and crossvalidated occurred on 50 plans for each fold. To ensure a fair and direct comparison between the single-site and multi-site training approaches, identical plans were in same fold during cross-validation. This guarantees that for any given fold, both the specialized 



<!-- Start of picture text -->
4Single-Site Models 1 Multi-Site Model<br>'i<br>{<br>HN (250train } SOtest yi HN 7000 train 50 test '<br>ets &£ ag<br>H'. Model ttH5 = H<br><!-- End of picture text -->



<!-- Start of picture text -->
'Prostate. (250train } 50 test H ’ Prostate 50 test H<br>' . 1 a > 1<br>' P= Prostate im c> H ‘ foe } H<br>' Model og | Mutti- ——_<br>Breast Sotest [MN | Breast ter sotest Jam<br>q ¢ \) ae Breast |= im if # Gay A !<br><!-- End of picture text -->



<!-- Start of picture text -->
q wei Model 5 +t a jj<br>' it '<br>HHt “6dLung ) (250Lungtrain } aap 50=test otBt} ' Lung=61 50o test- '1<br>i : Model it : '<br>H ht '<br>Ga|<br><!-- End of picture text -->

**_Figure 3 Schematic of the single-site models and multi-site models training_** _. (Left) Single-site models, where four individual models were independently trained on 250 cases from each anatomical site (head and neck, prostate, breast, and lung); and (Right) Universal multi-site model, where a single model was trained on the consolidated dataset of 1,000 cases (250 cases per site). Both paradigms were evaluated on the same 200 seen site cases (50 per site) to ensure a direct comparison of performance. This experimental design serves to evaluate whether a single, beam-aware model can achieve clinical parity with specialized, site-specific models while significantly reducing the technical overhead of model management._ 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

and generalized models were trained on the exact same set of patient cases. 

# **2.3.2 Model Input** 

All models used the same 4-channel input which includes the CT, targets, OARs, and beam maps. All of the training configurations use the default nnDoseNet autoconfiguration settings, which were optimized prior to this work. All models trained for this work use a 3D U-Net with depths of 6, batch size of 2 and stochastic gradient descent with momentum of 0.9. Patch sizes were determined for each model using the selfconfiguring process outlined in the nnU-net workflow. 

# **2.3.3 Loss Function** 

Default loss function of nnDoseNet<sup>5</sup> , the combination loss of MSE and DVH loss is utilized in this project. DVH loss, proposed by Wang et al.<sup>26</sup> , integrates the clinically relevant metrics, value-DVH (vDVH) and criteria-DVH (cDVH). The vDVH (equation 1) represents the dose gradient in contours while the cDVH (Equation 2) represents a specific dose value within a relevant volume of the contours. � Deontour” ����������� ��� Dose M. represents the map of the ground truth (GT) or prediction (Pred) in certain contour (target or oar). N represents the number of contours and n represents the number of voxels within the contour. �� Pr, �(D) represents the highest x percentile of dose value in the dose map, D. 



medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

# **2.4 Post-processing and Evaluation** 

We compared the prediction of all 4×2 single-site models to the 1×2 multi-site models in both the Gy-value trained (Gy out) model and the percentage trained (percentage out) model. In addition to trained sites, our models were evaluated on the two unseen disease sites of liver and whole-brain (WB) cases, with each site represented by 50 cases. 

Voxel-wise mean absolute error (MAE) and clinically relevant dose-volume histogram metrics (vDVH and cDVH) are utilized for evaluation. To facilitate comparison across multiple prescriptions, error metrics were normalized to the percentage of the prescribed dose; lower values denote higher prediction accuracy relative to the clinically delivered plans (ground truth). All tests of significance used paired Wilcoxon Signed-Rank Test when comparing the performance between two models. 

# **2.5 The difference between nnDoseNet1 and nnDoseNet2** 

The core philosophy of the framework remains unchanged, to establish an automated pipeline for dose prediction using DICOM or NIfTI datasets. However, nnDoseNetv2 introduces significant upgrades to the data extraction module, most notably the ability to extract beam geometry directly from DICOM plan files. Computational efficiency was enhanced in both preprocessing and training by allowing the use of dose grid resolution rather than the typically higher CT resolution, and by cropping the input volume to exclude regions distant from the target. Finally, the input channel configuration was optimized; by leveraging the defined ROI, the beam map was integrated without increasing the input dimensionality, maintaining a four-channel architecture. 

# **3 Results 3.1 Single-site vs multi-site model on seen site.** 

The top and middle rows of Figure 5 present models trained using absolute dose (Gy) and percentage of prescription, respectively. In both configurations, the multi-site model (green) exhibits lower overall deviation. Furthermore, it shows no statistically significant difference compared to single-site models (p-value of MAE, vDVH and cDVH of prostate: 0.82, 0.89, 0.63; breast: 0.02, 0.87, 0.97; lung: 0.06, 0.30, 0.47), with the exception of HN cases (p < 0.01 for all metrics). For instance, the multi-site model achieved an MAE of 1.24 ± 0.49% for breast cases, which is comparable to the single-site breast model 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

(1.29 ± 0.64%). Similarly, for HN cases, the multi-site model yielded an MAE of 1.09 ± 

0.36%, closely matching the single-site result of 1.05 ± 0.36%. This indicates that the 





<!-- Start of picture text -->
a ee [: >| él<br><!-- End of picture text -->



<!-- Start of picture text -->
(4 | @l @ I @|<br><!-- End of picture text -->



**Figure 4 Representative dose prediction results across seen anatomical sites.** Qualitative comparison of the input features and predicted dose distributions for the head and neck (HN), prostate, breast, and lung cohorts. _For each site, the input CT, beam geometry map, and anatomical contours are displayed alongside the groundtruth dose and corresponding predictions from the site-specific and universal multisite models. All results are shown for the prescription-labeled (Gy) models, with axial views representing typical cases from the 200-case independent testing set. The color bar indicates absolute dose values in Gy._ The visual agreement between the predictions and ground-truth distributions demonstrates the framework's ability to _predict dose distribution_ across diverse anatomical regions and delivery geometries. 

multi-site model achieves performance comparable to site-specific models while demonstrating greater stability. 

When comparing the two multi-site approaches (Figure 5, bottom row), the percentagetrained model significantly outperforms the Gy-trained model across all trained sites (pvalue of MAE, vDVH and cDVH of Breast: <0.001, 0.01, 0.01; Lung: <0.01, 0.01, <0.01; HN: <0.01, <0.01, <0.01), with the exception of prostate plans. For example, in Breast cases, the percentage-trained model achieved a VDVH error of 1.64 ± 0.92%, whereas the Gy-trained model resulted in a higher error of 2.71 ± 1.87%. However, for Prostate 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

cases, the performance was nearly identical, with the percentage-trained model yielding an MAE of 2.07 ± 0.84% compared to 2.04 ± 0.83% for the Gy-trained model. 

# **3.2 Multi-site model prediction on unseen sites** 

Two multi-site models were evaluated on two unseen anatomical sites, WB and Liver, with each test set consisting of 50 clinically delivered plans. For Liver cases (Figure 6), the multi-site model (MAE: 2.46 ± 1.49%) significantly outperformed most single-site models (p < 0.001), such as the Breast model (MAE: 3.47 ± 1.98%) and the Prostate model (MAE: 4.67 ± 2.33%). The only exceptions were the Gy-trained prostate model and the percentage-trained lung model, which showed no statistical significance (p > 0.05) difference compared to the multi-site model on DVH-based metrics. 

Regarding WB patients, the multi-site models significantly (p < 0.001) outperformed the Prostate and Lung models. Specifically, the multi-site model achieved an MAE of 6.97 ± 3.04%, which was markedly superior to the Prostate model (15.41 ± 3.41%) and the Lung model (8.01 ± 3.14%). However, the single-site HN model achieved slightly better results (MAE: 6.69 ± 2.71%) than the multi-site model across most metrics. 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 









<!-- Start of picture text -->
theksbex ss aa laakal<br><!-- End of picture text -->











**Figure 5** **_Evaluation of model scalability and beam-aware generalizability._** _(Rows 1–2) Site-specific vs. Multi-site performance: Comparison between specialized single-site models (Blue) and the universal multi-site model (Green) utilizing absolute dose (Gy; Row 1) and normalized dose (%; Row 2) training objectives. (Row 3) Labeling strategy comparison: Performance evaluation between Gy-labeled and percentage-labeled multi-site models across the four primary training sites. (Row 4) Impact of beam geometry on generalizability: Comparison between the baseline multi-site framework (nnDoseNet v1; Grey) and the upgraded beamaware framework (nnDoseNet v2; Green). The integration of machine-specific beam geometry in nnDoseNet v2 results in significantly enhanced accuracy on anatomically unseen sites (Liver and Whole Brain) and the Breast cohort. Performance is quantified via three distinct error-based metrics, where lower values indicate higher prediction accuracy: (Left) MAE, representing voxel-to-voxel dose differences. (Mid) cDVH representing differece in clinically relevant DVH metrics. (Right) vDVH representing differece in DVH curve. Statistical significance was determined using Wilcoxon signed-rank test (ns: not significant, * p < 0.05, ** p < 0.01, *** p < 0.001, and **** p < 0.0001.)_ 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 





















**_Figure 4 Generalizability of site-specific and universal multi-site models to unseen anatomical sites._** _Comparative performance of site-specific models (Blue) and the universal multi-site framework (Green) when validated on independent, unseen anatomical cohorts (Liver and Whole Brain). (Top two rows) Comparison of models utilizing the prescription-labeled (Gy) training format. (Bottom two rows) Comparison of models utilizing the percentage-labeled (%) training format. Performance is quantified via three distinct error-based metrics, where lower values indicate higher prediction accuracy: (Left) MAE, representing voxel-to-voxel dose differences. (Mid) cDVH representing difference in clinically relevant DVH metrics. (Right) vDVH representing difference in DVH curve. Statistical significance was determined using Wilcoxon signed-rank test (ns: not significant, * p < 0.05, ** p < 0.01, *** p < 0.001, and **** p < 0.0001.)_ 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

# **3.3 nnDoseNetv1 vs nDoseNet2 on seen and unseen site** 

The primary distinction between nnDoseNetv1 and nnDoseNetv2 lies in the 

incorporation of beam geometry. This feature provides critical context regarding the treatment approach and spatial dose distribution. As shown in bottom row of Figure 5, nnDoseNetv2 significantly (p < 0.05) outperforms v1, particularly on unseen anatomical sites. For example, on the unseen WB dataset using Gy-trained models, nnDoseNetv2 reduced the MAE to 13.61 ± 4.98%, compared to 16.13 ± 7.33% for nnDoseNetv1. Similarly, for the unseen Liver dataset, v2 achieved an MAE of 2.92 ± 1.95%, improving upon the v1 result of 3.29 ± 2.71%. 

# **4 Discussion** 

In this study, we introduced a scalable training approach for radiotherapy dose prediction that utilizes beam geometry to bridge the gap between disparate anatomical sites and treatment approach.  By utilizing an extensive cohort of 1,300 cases across six primary disease sites, we demonstrated that a single, universal model can maintain parity with specialized, site-specific models. To our knowledge, this is the first study to train a single deep learning model on a heterogeneous dataset encompassing a wide prescription range of 1.5 to 84 Gy, four distinct anatomical sites, and three different treatment modalities. Furthermore, we are the first to demonstrate the framework's zeroshot generalizability by validating the model on out-of-distribution disease sites that were entirely excluded from the training phase. 

A total of 1,000 clinically delivered RT plans across four distinct treatment sites were used to train our multi-site model, making it a single model trained with largest clinically delivered dataset for deep learning-based dose prediction reported to date. Our results show that the multi-site model performs comparably to the single-site models on the site seen, and even on liver cases, an unseen site that have similar anatomical structure to training sites. Even though single-site models trained with breast and HN dataset had a better performance than the multi-site model on WB site. This outcome is likely due to domain similarities: Breast and WB cases share comparable beam geometries, while HN cases share common OARs with WB cases. Although the multi-site model did not uniformly outperform the single-site models on their respective domains, it offers significant practical and clinical advantages. From an operational standpoint, a single, 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

unified model is far easier to deploy and maintain than multiple specialized models. This approach streamlines the clinical prediction workflow and reduces the potential for user error, as clinicians can utilize the same tool regardless of the treatment site. 

The heterogenous dataset training method effectively enlarges the training dataset by pooing the heterogenous data and further, it improves the generalizability and robustness of trained model. Furthermore, the multi-site training approach effectively increases the training data by pooling cases from different sites. This diversity acts as a natural regularizer, preventing the model from overfitting to site-specific nuances and enhancing its robustness. By training on a wide variety of anatomies, the model learns more generalizable features of dose-anatomy relationships and can implicitly share knowledge—such as strategies for sparing critical organs—across different anatomical contexts. Consequently, it is better equipped to handle atypical patient anatomy or rare clinical presentations. Perhaps the most significant advantage is the model's potential for transfer learning; it can serve as a powerful foundation model that can be fine-tuned with a much smaller, site-specific dataset, drastically lowering the barrier to entry for adopting automated planning solutions in new clinics or for novel treatment types. 

The incorporation of beam geometry improved the sharpness of the dose fall-off at the field edge, particularly for IMRT and 3D-CRT, which utilize more complex, conformal dose distributions. Initially, BEV was utilized in generating beam maps among all types of plans (IMRT, VMAT and 3D-CRT). But the predictions utilizing BEV fail on the breast cancer cases with low prescription doses and few beams (3D-CRT plans) where the dose should have cross-sectional shape more of field than conformed closely to the tumor volume. We also found that the WB cases performed better using the breast model and this is likely due to similarities in beam geometry between the two disease sites. These findings indicate that the model is highly reliant on beam geometry to produce a more robust and accurate dose distribution and the possibility to alter a predicted plan by inputting a different treatment geometry. For example, one could predict a VMAT dose distribution using the geometry from an IMRT plan, and vice versa. By incorporating machine-specific information into the input space, a single model can 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

be effectively trained on heterogeneous datasets, marking a significant step toward global dose prediction. 

Potential bias was found when training models only using Gy-labeled datasets and we observed that both single-site and multi-site perform better on prostate cases compared to other sites (Figure 5, top row). We suspect the reason might be that the model tends to prioritize the higher prescribed plan in training, which can efficiently lower the loss in the beginning. Therefore, we trained another set of models using percentage-labeled dataset, where we normalize the target mask and dose maps to percentile of prescription to eliminate the bias on higher prescribed plan. (Figure 5, bottom row) All other single-site models beside prostate, have significant (p < 0.05) better performance on cDVH and vDVH using percentage-trained model, that support our previous conjecture. 

In terms of computational efficiency, we also significantly shortened the prediction time. By resampling all images to the dose resolution and cropping input channels around the target, the models can predict a full dose distribution within approximately 10 seconds on a 24 GB Titan RTX GPU. This is a substantial improvement over the 5-minute prediction time reported in our previous work and applies to both the single-site and multi-site models. 

In addition to being trained on four primary disease sites, this study has several limitations. First, all data were derived from a single institutional planning environment, and the models may learn institution-specific planning preferences and contouring conventions rather than universally generalizable dose–anatomy relationships. Second, the clinically delivered dose was used as ground truth, which reflects historical practice and variability and does not necessarily represent an optimal reference plan. Third, while beam geometry was incorporated, the RTPLAN-derived beam map is a simplified surrogate that may not capture modulation and deliverability details (e.g., control-point fluence/MLC sequencing), which is important given the model’s reliance on the geometry channel. Finally, evaluation was based on MAE and DVH-derived metrics 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

which quantify similarity to delivered dose but do not fully establish clinical acceptability, constraint satisfaction, or deliverability. 

# **5 Conclusion** 

In summary, this study demonstrates the feasibility of training a single multi-site dose prediction model within the nnDoseNet framework using heterogeneous prescriptions and beam geometries, including an explicit beam-geometry input derived from DICOM plan files. Across the evaluated disease sites, the multi-site model achieved performance comparable to site-specific models while providing improved robustness in out-of-distribution testing, and the beam-geometry–enabled approach (nnDoseNetv2) improved prediction accuracy on unseen sites relative to the prior framework. Future work will focus on validation using multi-institutional datasets and expansion to additional disease sites to further assess generalizability. 

# **6 Reference** 

1. Kumar, Y. _et al._ Demonstrating an academic core facility for automated medical image processing and analysis: Workflow design and practical applications. _Diagnostics_ **15** , 803 (2025). 

2. Kumar, Y. _et al._ AI-based framework to fuse pre-RT brain metastases contours with follow-up MRI to improve post-RT assessment. _Neuro-Oncol. Pract._ npaf126 (2025). 

3. Cardenas, C. E., Cardan, R. A., Harms, J., Simiele, E. & Popple, R. A. Knowledge-based planning, multicriteria optimization, and plan scorecards: A winning combination. _Radiother. Oncol._ **202** , 110598 (2025). 

4. Cao, W. _et al._ Dose prediction via deep learning to enhance treatment planning of lung radiotherapy including simultaneous integrated boost techniques. _Med. Phys._ **52** , 3336–3347 (2025). 

5. Chang, H. _et al._ nnDoseNet: Intuitive and flexible deep learning framework to train and evaluate radiotherapy dose prediction models. _Comput. Biol. Med._ **198** , 111237 (2025). 

6. Cortes, K. G. _et al._ Knowledge-based three-dimensional dose prediction for tandem-and-ovoid brachytherapy. _Brachytherapy_ **21** , 532–542 (2022). 

7. Gronberg, M. P. _et al._ Technical Note: Dose prediction for head and neck radiotherapy using a threedimensional dense dilated U-net architecture. _Med. Phys._ **48** , 5567–5573 (2021). 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

8. Hou, Z. _et al._ Deep learning-powered radiotherapy dose prediction: clinical insights from 622 patients across multiple sites tumor at a single institution. _Radiat. Oncol._ **20** , 80 (2025). 

9. Hu, C. _et al._ TrDosePred: A deep learning dose prediction algorithm based on transformers for head and neck cancer radiotherapy. _J. Appl. Clin. Med. Phys._ **24** , e13942 (2023). 

10. Kallis, K. _et al._ Knowledge-based dose prediction models to inform gynecologic brachytherapy needle supplementation for locally advanced cervical cancer. _Brachytherapy_ **20** , 1187–1199 (2021). 

11. Kazemzadeh, A., Rasti, R. & Tavakoli, M. B. Artificial intelligence for radiotherapy dose prediction: A comprehensive review. _Cancer/Radiothérapie_ **29** , 104630 (2025). 

12. Kearney, V., Chan, J. W., Haaf, S., Descovich, M. & Solberg, T. D. DoseNet: a volumetric dose prediction algorithm using 3D fully-convolutional neural networks. _Phys. Med. Biol._ **63** , 235022 (2018). 

13. Koike, Y. _et al._ Patient-specific three-dimensional dose distribution prediction via deep learning for prostate cancer therapy: Improvement with the structure loss. _Phys. Med._ **107** , 102544 (2023). 

14. Leino, A. _et al._ Deep learning-based prediction of the dose–volume histograms for volumetric modulated arc therapy of left-sided breast cancer. _Med. Phys._ **51** , 7986–7997 (2024). 

15. Liu, S., Zhang, J., Li, T., Yan, H. & Liu, J. Technical Note: A cascade 3D U-Net for dose prediction in radiotherapy. _Med. Phys._ **48** , 5574–5582 (2021). 

16. Ma, J. _et al._ A feasibility study on deep learning-based individualized 3D dose distribution prediction. _Med. Phys._ **48** , 4438–4447 (2021). 

17. Maniscalco, A. _et al._ Multimodal radiotherapy dose prediction using a multi-task deep learning model. _Med. Phys._ **51** , 3932–3949 (2024). 

18. McIntosh, C. & Purdie, T. G. Voxel-based dose prediction with multi-patient atlas selection for automated radiotherapy treatment planning. _Phys. Med. Biol._ **62** , 415–431 (2017). 

19. Nguyen, D. _et al._ 3D radiotherapy dose prediction on head and neck cancer patients with a hierarchically densely connected U-net deep learning architecture. _Phys. Med. Biol._ **64** , 065020 (2019). 

20. Osman, A. F. I. & Tamam, N. M. Attention-aware 3D U-Net convolutional neural network for knowledge- 

   - based planning 3D dose distribution prediction of head-and-neck cancer. _J. Appl. Clin. Med. Phys._ **23** , e13630 (2022). 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

21. Osman, A. F. I., Tamam, N. M. & Yousif, Y. A. M. A comparative study of deep learning-based knowledgebased planning methods for 3D dose distribution prediction of head and neck. _J. Appl. Clin. Med. Phys._ **24** , e14015 (2023). 

22. Osman, A. F. I., Tamam, N. M. & Yousif, Y. A. M. A comparative study of deep learning-based knowledgebased planning methods for 3D dose distribution prediction of head and neck. _J. Appl. Clin. Med. Phys._ **24** , e14015 (2023). 

23. Shiraishi, S. & Moore, K. L. Knowledge-based prediction of three-dimensional dose distributions for external beam radiotherapy. _Med. Phys._ **43** , 378–387 (2016). 

24. Sun, Z. _et al._ A hybrid optimization strategy for deliverable intensity-modulated radiotherapy plan generation using deep learning-based dose prediction. _Med. Phys._ **49** , 1344–1356 (2022). 

25. Teng, L. _et al._ Beam-wise dose composition learning for head and neck cancer dose prediction in radiotherapy. _Med. Image Anal._ **92** , 103045 (2024). 

26. Wang, B. _et al._ Deep Learning-Based Head and Neck Radiotherapy Planning Dose Prediction via Beam-Wise Dose Decomposition. in _Medical Image Computing and Computer Assisted Intervention – MICCAI 2022_ (eds Wang, L., Dou, Q., Fletcher, P. T., Speidel, S. & Li, S.) 575–584 (Springer Nature Switzerland, Cham, 2022). doi:10.1007/978-3-031-16449-1_55. 

27. Wang, H.-J. _et al._ Adaptive radiotherapy dose prediction on head and neck cancer patients with a 3D multiheaded U-Net deep learning architecture. _Mach. Learn. Health_ **1** , 015008 (2025). 

28. Yusufaly, T. I. _et al._ A knowledge-based organ dose prediction tool for brachytherapy treatment planning of patients with cervical cancer. _Brachytherapy_ **19** , 624–634 (2020). 

29. Zimmermann, L., Faustmann, E., Ramsl, C., Georg, D. & Heilemann, G. Technical Note: Dose prediction for radiation therapy using feature-based losses and One Cycle Learning. _Med. Phys._ **48** , 5562–5566 (2021). 

30. Babier, A. _et al._ OpenKBP-Opt: an international and reproducible evaluation of 76 knowledge-based planning pipelines. _Phys. Med. Biol._ **67** , 185012 (2022). 

31. Babier, A. _et al._ OpenKBP: The open-access knowledge-based planning grand challenge and dataset. _Med. Phys._ **48** , 5549–5561 (2021). 

32. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J. & Maier-Hein, K. H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. _Nat. Methods_ **18** , 203–211 (2021). 

medRxiv preprint doi: https://doi.org/10.64898/2026.04.17.26350770; this version posted April 22, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted medRxiv a license to display the preprint in perpetuity. All rights reserved. No reuse allowed without permission. 

