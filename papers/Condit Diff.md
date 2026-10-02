# Conditional Diffusion Model with Anatomical-Dose Dual Constraints for End-to-End Multi-Tumor Dose Prediction 

Hui Xie<sup>_a,b_</sup> , Haiqin Hu<sup>_c_</sup> , Lijuan Ding<sup>_d_</sup> , Qing Li<sup>_b_</sup> , Yue Sun<sup>_a,∗_</sup> , Tao Tan<sup>_a,∗_</sup> 

_a. Faculty of Applied Sciences, Macao Polytechnic University, R. de Luís Gonzaga Gomes, Macao, 999078, , P. R. China_ 

_b. Department of Oncology,Affiliated Hospital of Xiangnan University, No.31 Renmin West Road, Chenzhou, 423000, Hunan, P. R. China_ 

_c. Department of Oncology,Jiangxi Cancer Hospital, No. 519, Beijing East Road, Nanchang, 330029, Jiangxi, P. R. China_ 

_d. Department of Oncology, Chenzhou Third People’s Hospital, No. 8, Jiankang Road, Chenzhou, 423000, Hunan, P. R. China_ 

_a_ 

_b_ 

## **Abstract** 

Traditional manual optimization in radiotherapy planning is highly experience-dependent and time-consuming. Although deep learning offers automated solutions, existing models still face limitations in generalization across tumor types, prediction accuracy, and adherence to clinical constraints. This paper proposes a novel end-to-end multi-tumor dose prediction model—the Anatomical-Dose Dual Constraints Diffusion Model (ADDiff-Dose)—which innovatively integrates anatomical and clinical dose dual constraints into a conditional diffusion framework. We design a Lightweight 3D Variational Autoencoder (LightweightVAE3D) that compresses high-resolution CT images to 0.3% of their original size for efficient 3D data processing. In the latent space, a conditional diffusion model with a 3D U-Net backbone guides the denoising process to generate dose distributions by incorporating multimodal conditions—including target and organ-at-risk masks and beam parameters—via a multi-head attention mechanism. Furthermore, we introduce a composite loss function that not only optimizes the mean squared error of noise prediction but also, for the first time, directly "hard-codes" over 50 clinical dose-volume constraints into the optimization objectives, ensuring the clinical feasibility of the generated results. Extensive evaluations on a large-scale public dataset (2,877 cases) and three external institutional cohorts (450 cases in total) demonstrate that ADDiff-Dose significantly outperforms state-of-the-art models across nearly all key metrics. Qualitative assessments include visual comparisons and difference maps, confirming superior structural dose distributions with reduced artifacts in high-dose regions. Ablation studies further confirm that our proposed anatomical-dose dual-constraint design is core to the model’s performance: removing any constraint leads to significant performance degradation, with the clinical constraint loss ( _L_ cond) ablation increasing spinal cord _D_ max error by 109%, and the MSE loss ( _L_ mse) ablation reducing spatial accuracy (DICE) by 8.8%. The full model enhances clinical dose compliance by 28.5% and enables uncertainty quantification through multiple stochastic inferences.This work provides a highly accurate, efficient, and generalizable automated treatment planning tool for clinical practice. 

_Keywords:_ `Conditional Diffusion Model; Anatomical-Dose Dual Constraints; Dose Prediction; Multi-Tumor` 

_2010 MSC:_ 00-01, 99-00 

# :These authors contributed equally to this work. 

> _∗_ Correspondingauthor: Tao Tan and Yue Sun 

> _Email address:_ `taotanjs@gmail.com and joyyuesun@gmail.com` (Tao Tan<sup>_a,∗_</sup> ) 

## **1. Introduction** 

Cancer remains a leading cause of morbidity and mortality worldwide, posing a persistent threat to human health. Among the three principal treatment modalities—surgery, chemotherapy, and radiotherapy—approximately 80% of patients with malignant tumors require radiotherapy at some stage during their disease course [1]. The efficacy of radiotherapy hinges on the precise formulation of treatment plans, which aim to maximize tumor cell eradication while sparing adjacent healthy tissues. This planning process typically demands experienced radiotherapy physicists to iteratively adjust beam parameters to generate clinically acceptable three-dimensional dose distributions, often assessed via dose-volume histograms (DVHs). However, manual planning is time-consuming and labor-intensive, usually taking several hours to days for a single case. Additionally, the final plan quality can vary significantly depending on the planner’s expertise, potentially limiting the individualization and optimality of treatment regimens [2]. 

In response, recent advances in deep learning have facilitated the development of data-driven models that automate dose prediction and streamline clinical workflows. Convolutional neural networks (CNNs), generative adversarial networks (GANs), and attention-based models have shown promise in learning mappings between anatomical features and dose distributions from historical datasets [3]. For example, ResNet101 has been employed to accurately predict dose for nasopharyngeal carcinoma by integrating anatomical and beam parameters with a multi-scale feature fusion mechanism [4] [5]. In the domain of lung cancer, the asymmetric A-Net architecture achieved clinically acceptable results in the 50–60 Gy range and delivered high spatial precision in mobile lesion regions [6, 7]. Traditional knowledge-based planning (KBP) approaches also remain relevant. Techniques combining kernel density estimation, k-nearest neighbors(KNN), and principal component analysis (PCA) have demonstrated the feasibility of dose modeling from limited clinical cohorts [8]. Despite encouraging progress, current methods face several limitations. First, most existing models are tailored to specific tumor types (e.g., prostate, head and neck) or limited to particular delivery techniques (e.g., IMRT, Intensity-Modulated Radiation Therapy; VMAT, Volumetric-Modulated Arc Therapy), limiting cross-domain generalization [7, 9]. Second, training data volumes remain relatively small (typically <200 patients), restricting model robustness to anatomical variability [10]. Third, prediction performance has yet to meet clinical gold standards: even advanced architectures like U-Net and 3D GAN exhibit mean absolute errors exceeding 1–3 Gy in certain regions or cases [11, 12]. 

In parallel, diffusion models have emerged as a powerful generative paradigm, achieving state-of-the-art performance across diverse domains such as image synthesis [13], fashion generation [14], and human motion modeling [15, 16]. Their ability to iteratively refine predictions through a learned denoising process offers unique advantages for generating complex, high-resolution outputs under multimodal constraints. However, while diffusion models have been widely adopted in computer vision, natural language processing, and other multimedia applications [17, 18, 19], their application to radiotherapy dose prediction remains relatively nascent. Only a handful of recent studies, such as DoseDiff [20] and MD-Dose [21], have begun exploring this avenue, highlighting a critical research gap in leveraging the strengths of diffusion models to handle the anatomical complexity and precision demands of radiotherapy. Existing methods still face unresolved limitations: traditional models like U-Net and GAN fail to integrate clinical dose constraints into core optimization, leading to predicted distributions that may violate safety thresholds; diffusion-based approaches such as DoseDiff [20] and MD-Dose [21] focus solely on anatomical distance or architectural design, lacking synergistic optimization of ’anatomical structure - dose distribution’. 

To address these challenges, we propose ADDiff-Dose, an end-to-end anatomical-dose dual-constrained conditional diffusion framework for multi-tumor radiotherapy dose prediction. Our model incorporates CT images, organ and target contours, prior dose maps, and beam parameters as multimodal inputs to guide a conditional generative process. By 

2 

integrating domain knowledge directly into the diffusion trajectory, ADDiff-Dose enables accurate and generalizable dose prediction for both head and neck and lung cancers under a unified architecture. Extensive experiments on multicenter datasets demonstrate its superiority in prediction accuracy, structural consistency, and cross-site adaptability, offering a promising solution for future intelligent radiotherapy planning. While pioneering studies such as DoseDiff [20] and MD-Dose [21] have demonstrated the potential of diffusion models for dose prediction, their conditioning mechanisms primarily focus on anatomical distance or Mamba-based architectures, respectively. Our work, ADDiff-Dose, distinguishes itself by introducing a comprehensive anatomical-dose dual-constraint framework. This framework explicitly encodes over 50 clinical dose-volume constraints directly into the diffusion loss function, ensuring the generated dose distributions are not only accurate but also strictly compliant with clinical protocols—a critical aspect not jointly emphasized in prior diffusion-based approaches. To validate this, we provide both quantitative metrics and qualitative visualizations comparing ADDiff-Dose with these diffusion-based baselines, demonstrating improved dose conformity and fewer violations of clinical constraints. The core innovations of this study are condensed into three key aspects: 

1. **Anatomical-dose dual-constraint framework** :A novel conditioning mechanism that fuses multimodal anatomical features (tumor/OAR masks, CT intensity) with over 50 explicit clinical dose-volume constraints (e.g., spinal cord _D_ max _≤_ 45 Gy, lung _V_ 20 _≤_ 30%), breaking through the limitation of pure anatomical guidance and ensuring both spatial accuracy and clinical compliance. 

2. **Efficient 3D data processing** : A Lightweight 3D-VAE specifically designed for radiotherapy scenarios, achieving 99.7% dimensional compression of high-resolution CT data while preserving key anatomical features, significantly reducing computational burden and enabling clinical deployment. 

3. **Unified multi-tumor prediction architecture** : Breaking through the limitations of traditional single-tumor models, our framework enables unified intensity-modulated radiation therapy (IMRT) and volumetric modulated arc therapy (VMAT) dose prediction for both head-and-neck (H&N) and lung cancers within a single architecture. Specifically, a single model checkpoint is trained on a combined dataset (Lung + H&N), wherein tumor type (i.e., class label) is embedded as a conditional feature to mitigate inter-tumor-type interference and enhance task-specific adaptation. 

## **2. Materials and Methods** 

Radiotherapy dose distribution prediction must simultaneously satisfy the dual requirements of anatomical structure fidelity and physical constraint compliance. To address the limitations of traditional unconditional diffusion models in integrating multimodal prior knowledge, this study proposes an end-to-end diffusion model based on anatomical-dose dual constraints. The model innovatively integrates multimodal inputs and prior-knowledge-guided diffusion mechanisms, enabling accurate dose prediction across tumor types within a unified architecture. The model comprises the following core components: a lightweight 3D variational autoencoder (LightweightVAE3D), a 3D UNet backbone network[22], a conditional embedding layer (ConditionalLayer), and a multi-head attention mechanism (MultiHeadAttention)[23] (Figure 1, 2 and 3). As shown in Figure 1, ADDiff-Dose operates in two stages: (1) The _LightweightVAE3D_ is pre-trained to compress high-resolution CT images and dose map into a low-dimensional latent space. (2) The _conditional diffusion model_ then operates entirely within this latent space. It takes the encoded latent vector _z_ 0 and, through a progressive noising and denoising process conditioned on multimodal conditions such as clinical prior (radiotherapy dose OAR constraint table), Targets/OARs mask, Beam Information, and positional information (tumor distribution site), learns to predict the corresponding dose distribution in the latent domain. The final dose map is obtained by decoding the denoised latent vector back to the image space using the VAE decoder. 

3 



<!-- Start of picture text -->
CT+Does<br>Conditioning<br>Clinical Prior<br>E Diffusion ProcessForwrd Diffusion Process<br>Targets/OARs<br>Masks<br>�� Repeat  T  time ��<br>Beam<br>Information<br>t  1 t  T  1 t   T<br>Class Labels<br>Does UNet UNet<br>D Positional<br>Information<br>�� �� ��−� ��<br>��<br><!-- End of picture text -->

**Figure 1:** Overall Framework of the Anatomical-Dose Dual Constraints Conditional Diffusion Model (ADDiff-Dose). The schematic illustrates the end-to-end architecture for multi-tumor radiotherapy dose prediction. The diagram illustrates the conditional diffusion process for dose prediction. The top path shows the forward diffusion process, where the initial latent representation _Zo_ is progressively noised over T steps to become _ZT_ . The bottom path shows the reverse denoising process, where a U-Net iteratively refines _ZT_ back to _Zo_ , conditioned on multi-modal inputs including clinical priors, PTV/OAR masks, class labels, positional information, and beam parameters. The final predicted dose is obtained by decoding the denoised _Zo_ . 

## _2.1. Lightweight 3D Variational Autoencoder (LightweightVAE3D)_ 

To address the issue of dramatically increased computational complexity caused by the high resolution of the original CT images in radiotherapy dose prediction tasks, this study designed a lightweight 3D variational autoencoder (LightweightVAE3D)[24]. The model adopts a symmetric encoder-decoder architecture (Figure 2), and achieves a balance between computational efficiency and anatomical feature preservation through joint optimization of feature compression and reconstruction. 

**Encoder Design** : The encoder employs a four-layer 3D convolutional chain (kernel size: 4×4×4, stride: 2, padding: 1, activation function: StableSiLU) to perform progressive downsampling. The number of input channels increases gradually from 1 to 256 (channel expansion ratio: 1:256), enabling the model to capture hierarchical image information through multi-scale feature extraction. Ultimately, the original CT images (with a resolution of 96×128×144 voxels) are compressed into a low-dimensional latent space of 6×8×9 voxels, achieving a dimensionality reduction of approximately 99.7%. At the end of the encoder, fully connected layers output the Gaussian distribution parameters — mean _µ_ and variance _σ_<sup>2</sup> — of the latent space. These parameters provide the probabilistic foundation for the subsequent diffusion process. 

**Decoder design** : The decoder employs a symmetric 3D transposed convolution chain to gradually upsample and restore image resolution. The number of output channels decreases from 256 to 1, adapting to the normalized CT value range of 0 _,_ 1. A Sigmoid activation function is introduced to constrain the reconstructed value range, avoiding numerical overflow issues and ensuring the accuracy of anatomical structure reconstruction and clinical rationality. 

Training optimization strategy: During the training phase, reparameterization techniques are introduced to enhance gradient propagation efficiency. The model jointly optimizes the adversarial loss and the reconstruction loss _LV AE_ = _∥x_ ˜ _− x∥_ 1 _βDKLqz|x∥pz_ . Here, the Kullback-Leibler (KL) divergence _DKL_ adjusts the weight coefficient _β_ through a dynamic annealing strategy, achieving progressive regularization of the latent space distribution. This strategy reduces 

4 



<!-- Start of picture text -->
µ<br>latent<br>Log(var)<br>×2<br>: Conv3d ; : Sigmoid;<br>: BatchNorm3d; :Decoder;<br>: StableSiLU;   :Encoder;<br>: AdaptiveAvgPool3d;<br>: ConvTranspose3d; : Conv3d(1 x 1); :Input or Output<br><!-- End of picture text -->

**Figure 2:** Architecture Diagram of the Variational Autoencoder (VAE). The upper encoding process uses layers like Conv3d, BatchNorm3d, StableSiLU, and AdaptiveAvgPool3d to generate **_µ_** and Log(var) for sampling the latent representation. The lower decoding process leverages ConvTranspose3d, StableSiLU, and Sigmoid layers to reconstruct output from the latent space, illustrating the full VAE workflow. 

computational load while successfully preserving the morphological features of the target area and organs at risk, providing high-quality anatomical priors for subsequent diffusion models. 

## _2.2. Conditional Diffusion Model_ 

## _2.2.1. Multi-Source Conditional Feature Construction_ 

After being processed by the VAE encoder, the 3D CT image generates a 32-dimensional latent vector _z_ 0, which serves as the initial input for the diffusion process. Simultaneously, multi-source conditional features are constructed as follows: 

1. **Structural features:** Channel merging operations are performed on the planning target volume (PTV) mask and the organs-at-risk (OAR) masks (e.g., spinal cord, lung, brainstem) to generate a multi-channel structural tensor, which is then projected via 3D convolution to form structural conditional features. 

2. **Clinical prior features:** Encode over 50 clinical dose-volume constraints (detailed in Supplementary Table 1), such as spinal cord _D_ max _≤_ 45 Gy, lung _V_ 20 _≤_ 30%, and PTV _D_ 95 _≥_ 95% of prescribed dose, into feature vectors through fully connected layers. 

3. **Beam parameters:** Include beam energy (6 MV/15 MV), field angle, and number of beams as discrete features, which are converted into continuous vectors via embedding layers. 

4. **Temporal features:** For discrete time steps _t ∈_ 0 _,_ 999, map them into high-dimensional temporal feature vectors 

   - _t_ emb through a time embedding function to capture the dynamic evolution characteristics during the diffusion process. 

All conditional features are fused via the multi-head attention mechanism to form the final conditional feature **C** , which guides the denoising process of the diffusion model. 

## _2.2.2. Progressive Noising and Denoising Mechanism_ 

The noising process follows the Markov chain rule[25], using a linearly scheduled noise coefficient _βt_ (whose values range from 10<sup>_−_4</sup> to 0.02), progressively injecting Gaussian noise into the latent vector: 

5 



<!-- Start of picture text -->
U-Net En1 En2 En3 En4<br>Out ch>32 Out ch>32<br>Q Q<br>In:1 × 32 ×× 3232 ××3232 K V K V<br>In:1 × 32 ×32× In:1 × 64 ×32× In:1 × 128 ×32× 32×32 ×32 In:1 × 256 ×32× 32×32 ×32 Out ch>32<br>32×32 ×32 32×32 ×32 Out: 1 × 256 ×32× 32×32 X32 Out: 1 × 512 ×32× 32×32 ×32<br>Out: 1 × 64 ×32× Out: 1 × 128 ×32× Q<br>32×32 ×32 32×32 ×32 K V<br>De1 De2 De3 De4 Out ch>32<br>Q<br>K V<br>Out ch>32 Out ch>32<br>Q Q<br>Out:1 × 32 ×32 K V K V<br>× 32 ×32 ×32<br>In:1 × 128 ×32× In:1 × 192 ×32× In:1 × 384 ×32× 32×32 ×32 In:1 × 768 ×32× 32×32 ×32<br>32×32 ×32 32×32 ×32 Out: 1 × 128 ×32× 32×32 ×32 Out: 1 × 256 ×32× 32×32 ×32<br>Out: 1 × 64 ×32× Out: 1 × 64 ×32×<br>32×32 ×32 32×32 ×32<br>: Conv3d+GroupNorm+StableSiLU; :Input or Output; :ResBlock; :MultiHeadAttention; [ ] :The kernel size of Conv3d; : Cat<br>ResBlock MultiHeadAttention<br>Cov3d 1 x1 x1 Reshape<br>Softmax<br>Cov3d 1 x1 x1 Reshape<br>Cov3d 1 x1 x1 Reshape<br>[1  [3  [3  [3  [3<br> 1] ×× 1   3]×× 3   3]×× 3   3]×× 3   3]×× 3   3]×[3 × 3   3]×[3 × 3   3]×[3 × 3   3]×[3 × 3<br>3]<br>[3  [3<br> 3]× [3 X 3 X  × 3   3]× × 3<br>Middle Block<br>[3  [3<br> 3]× × 3   3]× × 3  Out: 1 32 3232 32 512 × 32 3232 32 512 In:1 ×<br> 1] [ ××1  1  [3  3]×× 3  [3  3]×× 3  [3  3]×× 3  [3  3]×× 3   3]×[3 × 3   3]×[3 × 3   3]×[3 × 3   3]×[3 × 3<br>MLP Attention Shortcut<br>(TimeEmbedding)<br>B,C,D,H,W<br>B,H,I,J,K,L,M,N<br>B,C,D,H,W<br>Reshape<br>x Conv3d GroupNorm StableSiLU Conv3d GroupNorm StableSiLU F(X) F(X)+X<br>x<br>B,H,D,I,J,K<br><!-- End of picture text -->

**Figure 3:** Architecture Diagram of the U-Net Model with Modified Attention and ResBlock Components. It shows the encoding (En1-En4) and decoding (De1-De4) paths, incorporating elements like Cascade - GroupNorm - StableSiLU, ResBlock, and MultiHeadAttention, along with detailed internal structures of ResBlock and MultiHeadAttention at the bottom. 



where _αt_ = 1 _− βt_ , _εt−_ 1 _∼N_ 0 _,_ **I** . Through this process, the original signal _z_ 0 gradually degrades to Gaussian noise _zT_ . 

The reverse denoising process is implemented by a 3D UNet (UNet3D) (Figure 3). This UNet adopts an encoder-decoder architecture, consisting of 4 layers of downsampling (channel numbers increasing from 64 to 512) and 4 layers of upsampling (channel numbers decreasing from 512 to 64), and fuses multi-scale features through skip connections. In the deeper layers, a multi-head attention mechanism is introduced. 

The UNet takes the noisy vector _Zt_ , time embedding temb, and conditional feature _C_ as inputs, predicting the noise _ϵθzt, t, C_ , and then recovers the true signal through the recursive formula: 



where _αt_<sup>cum</sup> is the cumulative noise coefficient, and _βt_<sup>_′_is the adjusted noise variance, used to recover the true signal.</sup> 

To leverage the stochastic nature of diffusion models, we perform multiple inferences (e.g., 10 runs) on the same input by sampling different noise seeds in the reverse process. This generates a distribution of possible dose maps, from which we compute pixel-wise variance as an uncertainty map, highlighting regions of high variability (e.g., near tumor boundaries). 

6 

## _2.2.3. Anatomical Condition Fusion Mechanism_ 

The conditional feature _C_ and the time embedding _temb_ interact through the conditional embedding layer. First, the dimensionality of the structural tensor is compressed using 1×1 convolution. Then, cross-attention calculation is performed through Transformer blocks[26]. Finally, feature fusion is achieved with the aid of the multi-head attention mechanism. This mechanism ensures that the diffusion process is fully aware of the distribution of anatomical structures, thereby guiding the dose distribution to comply with clinical constraints, such as the maximum dose to the Spinal cord _Dmax ≤_ 45 Gy, the Lung _V_ 20 _≤_ 30%, etc. ( Supplementary Table 1). 

## _2.3. Loss Function_ 

The composite loss function designed in this study consists of three core components, which are optimized through multi-component collaborative optimization mechanisms to ensure that the generated radiotherapy dose distribution is consistent with the anatomical structure while strictly satisfying clinical compliance requirements. The formula is expressed as: 



1. **Reconstruction Loss (** _Lmse_ **) :** The reconstruction loss _Lmse_ measures the difference between the predicted noise by UNet3D and the actual added noise, thereby driving the learning process of the "noise-to-dose" mapping. It is defined as: 



Where: **_z_** _t_ : The latent vector at time step _t_ (generated by 3D-VAE encoding); - _C_ : Conditional features combining the planning target volume (PTV) mask, organs-at-risk (OAR) masks, and time embeddings; _ϵ ∼N_ 0 _, I_ : Standard Gaussian noise. 

2. **Clinical Dose Constraint Loss (** _Lcond_ **):** The clinical dose constraint loss _Lcond_ integrates over 50 clinical 

constraints (Supplementary Table 1), categorized into two types: 

## **(1) OAR Protection Constraints** 

Maximum Dose Constraint ( _Dmax_ ): For example, the maximum dose for the spinal cord should not exceed 45 Gy. The loss term is defined as: 



where _Mj_ represents the threshold value for organ _j_ . 

Volume Dose Constraint ( _Vx_ ): For example, the _V_ 20 for the lung should not exceed 30% . The loss term is defined as: 



**(2) PTV Coverage Constraints :** Dose Coverage Constraint _D_ 95: For example, the _D_ 95 for the PTV should not be lower than 95% of the prescribed dose. The loss term is defined as: 



all constraint terms are weighted according to clinical priority and summed up, expressed as: 



7 

where _wo_ and _wp_ are weights set by radiation oncologists based on clinical experience (e.g., the weight for spinal cord _Dmax_ is higher than that for lung V20), and the constraints are only activated when the corresponding organ exists. 

**(3) VAE Regularization Loss (** _Lkl_ **) :** The VAE regularization loss _Lkl_ is only enabled during the pre-training phase of the 3D-VAE, where it ensures that the latent space distribution approximates a standard Gaussian distribution. This loss is defined as: 



where _µ_ and _σ_<sup>2</sup> are the mean and variance of the latent space generated by the VAE encoder. This loss ensures that the extracted anatomical features have good distribution properties, providing high-quality feature bases for the diffusion model. 

During the pre-training phase, both _Lmse_ (reconstruction loss) and _Lkl_ are optimized simultaneously to improve the quality of feature extraction. In the end-to-end training phase, the design ensures feature quality through _Lkl_ , while _Lmse_ and _Lcond_ work together to ensure that the generated dose distribution is highly consistent with the CT anatomy and strictly satisfies clinical constraints, providing a reliable optimization target for radiotherapy dose prediction. 

The weighting coefficients _λ_ 1, _λ_ 2, and _λ_ 3 in Equation 3 are initially set by radiation oncologists based on clinical priorities (e.g., _λ_ 2 = 1 _._ 5 to emphasize target coverage) and subsequently refined via grid search on a held-out validation set. Sensitivity analysis demonstrates model robustness: perturbing _λ_ 2 by _±_ 20% alters dosimetric constraint compliance by less than 5%. To resolve inherent trade-offs—such as between PTV coverage and organ-at-risk (OAR) sparing—we adopt a clinically motivated prioritization scheme: OAR protection is assigned higher weights, and constraint violations are penalized asymmetrically using max0 _,_ violation<sup>2</sup> terms, ensuring only _exceedances_ (e.g., _D_ mean _> D_ limit) incur penalty. 

## _2.4. Implementation of Organ-Specific Constraint Activation Mechanism_ 

The mechanism for activating constraints only when the corresponding organ exists is implemented through a three-step process to avoid invalid constraints affecting model training: 

1. **Organ existence detection:** For each patient’s OAR mask, calculate the volume of non-zero voxels (representing the anatomical region of the organ). If the volume is greater than 5 voxels, the organ is considered to exist; otherwise, it is considered non-existent. 

2. **Constraint mask generation:** Generate a binary mask vector where each element corresponds to a clinical constraint. If the organ corresponding to the constraint exists, the mask value is set to 1 (constraint is activated); otherwise, it is set to 0 (constraint is deactivated). 

3. **Masked loss calculation:** Multiply the constraint mask vector with the corresponding constraint loss term (detailed in Section 2.3). Only the loss of constraints for existing organs contributes to the total loss, while the loss of constraints for non-existing organs is ignored. 

This mechanism ensures that the model Dose not learn invalid constraints for non-existing organs, avoiding negative impacts on prediction performance and improving the clinical relevance of the model. 

## _2.5. Training Details_ 

This study implements a two-stage training strategy on a single NVIDIA RTX 4090 GPU (with 24GB of video memory) based on the PyTorch deep learning framework[27]. It integrates mixup data augmentation[28] and patch-based processing techniques[29] to enhance the model’s generalization ability and effectively address the memory bottleneck caused by 3D medical imaging data. 

8 

**Stage 1** : Pre-training the Lightweight 3D Variational Autoencoder (LightweightVAE3D). The Adam optimizer is employed (with an initial learning rate of 1e-3, which gradually decays to 1e-5 via cosine annealing). The batch size is set to 8, and training is conducted for 200 epochs. During training, the mixup data augmentation strategy (with a mixing coefficient alpha = 0.4) is introduced to randomly mix CT image samples and their latent features, thereby enhancing data diversity. Additionally, for high-resolution 3D medical images, a patching strategy with voxel dimensions of 32×32×24 (with an overlapping region of 8×8×8) is adopted to reduce video memory usage and improve training efficiency. 

**Stage 2** : Training the Conditional Diffusion Model. Based on fixed VAE encoder parameters, the AdamW optimizer is used (with an initial learning rate of 5e-4, which cosine-anneals to 1e-6). The batch size is set to 2, and training is carried out for 1000 epochs. The diffusion process sets a total number of steps T = 1000, with the noise coefficient _βt_ linearly increasing from 1e-4 to 0.02, ensuring the smoothness and controllability of the noise addition process. During training, the mixup augmentation strategy is also applied to dynamically mix dose distribution samples and structural masks (with a mixing coefficient lam = 0.4), further enhancing the model’s robustness and generalization ability. 

To address the memory limitations of 3D data, the input data follows a "patch - process - merge" workflow: the data is divided into patches of size 32×32×24, with an overlapping region of 8×8×8 between adjacent patches. The patch size of 32×32×24 was selected to balance GPU memory constraints and the need to capture sufficient 3D contextual information. This size is large enough to encompass most OARs and PTVs in a local region while allowing for a feasible batch size during training. After prediction, a linearly decaying weighted merging strategy is used to fuse the overlapping regions, avoiding boundary effects. The entire training process is optimized end-to-end using the composite loss function _L_ = _λ_ 1 _· Lmse λ_ 2 _· Lcond_ (which includes the weighted sum of the _Lmse_ reconstruction loss and the _Lcond_ loss with over 50 clinical dose constraints). The weights in the composite loss function ( _λ_ 1 = 1 _._ 0 _, λ_ 2 = 0 _._ 5) were determined via a grid search on a held-out validation set to balance the scale of the gradient contributions from _L_ mse and _L_ cond. This configuration prioritized dose prediction accuracy while ensuring effective enforcement of clinical constraints. The VAE regularization weight ( _λ_ 3 = 0 _._ 001) was set following common practices in _β_ -VAE literature[24] to balance latent space regularity and reconstruction quality. Each training cycle fully traverses the training set and validation set, combined with an early stopping mechanism: if the clinical dose compliance rate dose not improve for 20 consecutive epochs, training is terminated early. The entire training process takes approximately 200 hours. Through the synergistic effects of the aforementioned data augmentation, patching strategy, and multi-objective optimization mechanism, not only is the model’s generalization ability significantly enhanced, but the computational resource bottleneck of 3D medical data is also successfully addressed. Ultimately, simultaneous convergence of dose distribution prediction in terms of both anatomical structure adherence and clinical compliance is achieved. 

For a clearer understanding, the training procedure is summarized in Algorithm 1. 

## _2.6. Multi-Tumor Training Strategy_ 

To achieve multi-tumor dose prediction under a unified framework, we adopted a mixed training strategy combined with tumor type embedding, which effectively avoids data interference and improves cross-tumor generalization: 

1. **Dataset mixing:** Merge head and neck cancer and lung cancer datasets into a unified training set, and randomly sample during each training iteration to ensure the model is evenly exposed to diverse anatomical features and dose distribution patterns of different tumors. 

2. **Tumor type embedding:** Introduce a one-hot encoding vector representing tumor type (head and neck cancer: 1 _,_ 0; lung cancer: 0 _,_ 1) as part of the conditional feature. This vector is fused with other multimodal features (PTV/OAR 

9 

masks, beam parameters) through the multi-head attention mechanism, helping the model distinguish tumor-specific dose distribution characteristics. 

3. **Adaptive constraint activation:** Based on the tumor type, the model automatically activates corresponding clinical constraints (e.g., parotid gland dose constraints for head and neck cancer, lung volume dose constraints for lung cancer) to ensure the clinical relevance of the generated dose distributions. 

This training strategy enables the model to learn _shared_ dose prediction principles across tumor sites while adapting to _tumor-specific_ anatomical and clinical constraints—thereby eliminating the need for separate model training per tumor type and improving both training efficiency and generalization capacity. To mitigate potential inter-dataset interference (e.g., divergent organ-at-risk (OAR) prioritization between lung and head-and-neck cases), we embed tumor-type class labels into the conditional feature vector, which steers the attention mechanism to dynamically modulate feature representations in a task-aware manner. Ablation studies confirm minimal negative interference: joint training on the combined dataset improves cross-tumor generalization by 12 % in DICE similarity coefficient (mean across OARs) compared to independently trained single-tumor models, while per-tumor performance remains stable (degradation _<_ 2 % in clinically relevant dose metrics such as _D_ 98 for targets and _D_ mean for critical OARs). 

## **3. Experiments and Results** 

- _3.1. Dataset and Evaluations_ 

In this study, the model was trained and internally validated using data from the AAPM GDP-HMM Dose Prediction Challenge (2025)[30] [31] [32]. To evaluate the generalization ability of the model, we conducted external testing on three private datasets: one from the Affiliated Hospital of Xiangnan University (comprising 300 patients who received radiotherapy), another from the Third People’s Hospital of Chenzhou City (including 50 treated patients), and the third from Jiangxi Cancer Hospital (containing 100 patients undergoing radiotherapy). Detailed information about the data is provided in Table 1. The key acquisition parameters of the datasets are supplemented as follows: 

_Dataset Acquisition Parameters_ 

- **The Affiliated Hospital of Xiangnan University:** Utilizes a Philips Brilliance Big Bore CT scanner, operating at a tube voltage of 120 kV and a tube current of 225 mA, with a matrix size of 512×512. For head and neck scans, the slice thickness is set to 3 mm, while for lung scans, the slice thickness is set to 5 mm (150 cases each). 

- **Chenzhou Third People’s Hospital:** Utilizes a GE Discovery CT750 HD scanner , operating at a tube voltage of 120 kV and a tube current of 200 mA, with a matrix size of 512×512. For head and neck scans, the slice thickness is set to 3 mm, while for lung scans, the slice thickness is set to 5 mm (25 cases each). 

- **Jiangxi Cancer Hospital:** Utilizes a Siemens Somatom Definition AS , operating at a tube voltage of 120 kV and a tube current of 250 mA, with a matrix size of 512×512. For head and neck scans, the slice thickness is set to 3 mm, while for lung scans, the slice thickness is set to 5 mm (50 cases each). 

For each patient, the datasets include the computed tomography (CT) images, planning target volume (PTV) segmentation maps, organs-at-risk (OARs) segmentation maps, and the clinically delivered dose distributions. In our study, the clinical dose distribution information was incorporated to ensure consistency between anatomical and dosimetric features. All 3D images (including CT images) were resampled and cropped/padded to a standard size of 96×128×144 voxels. CT values were normalized to the range [-1000, 1000] and then divided by 500. Dose distributions were normalized based on the 3rd 

10 

**Table 1:** Summary of Clinical Datasets Used in This Study 

|**Dataset Source**|**N**|**Tech**|**nique**|**Dis**|**ease Type**|
|---|---|---|---|---|---|
|||IMRT|VMAT|Lung|Head & Neck|
|AAPM GDP-HMM 2025 (Public)<br>[30,31,32]|2877|1646|1231|1454|1423|
|The Affiliated Hospital of Xiangnan University|300|150|150|150|150|
|Chenzhou Third Hospital|50|25|25|25|25|
|Jiangxi Cancer Hospital|100|50|50|50|50|



_Note._ IMRT: Intensity-Modulated Radiation Therapy; VMAT: Volumetric Modulated Arc Therapy. All private datasets were collected between 2022-2024 with IRB approval. 

percentile value of the high-dose region within the PTV and divided by a factor of 10. In the internal dataset, all planning target volumes (PTVs) and organs-at-risk (OARs) were contoured by experienced radiation oncologists, and all radiotherapy plans have been clinically approved. 

We employed the following evaluation metrics to quantitatively analyze the model performance: for the target region, if multiple targets were present, the target receiving the highest dose was selected as the representative. The evaluation metrics included the Mean Absolute Error (MAE), Dice Similarity Coefficient (DICE), and the 95th percentile Hausdorff distance (HD95). Specifically, MAE quantifies the overall accuracy of dose delivery by calculating the mean absolute difference between the predicted and the actual dose distributions. The DICE coefficient evaluates the spatial similarity between the predicted and the ground-truth dose distributions based on the ratio of overlapping volume to the union volume. HD95 measures the maximum one-sided distance between the predicted and the true contours at the 95th percentile, reflecting their proximity in spatial localization (Table 2). Notably, in scenarios involving multiple targets, selecting the highest-dose PTV as the evaluation focus aims to concentrate on the region with the steepest dose gradient and the most stringent requirements for dose accuracy. This ensures the clinical relevance of the evaluation results and enhances the sensitivity of the metrics. 

To comprehensively and in - depth evaluate the performance of the proposed model, we additionally utilized a variety of evaluation indicators. For the target volume, we focused on key indicators such as _D_ 98, _D_ 2, maximum dose ( _D_ max), mean dose ( _D_ mean), homogeneity index (HI), and conformity index (CI). These indicators can reflect the model’s performance in target volume dose prediction from different dimensions (Table 3). 

For OARs, we adopted maximum dose, mean dose, the minimum dose delivered to a specific percentage of the structure ( _Dx_ %), and the percentage of the structure volume that receives at least a specific dose ( _Vx_ Gy) as evaluation indicators. These indicators help us understand the model’s ability to protect organs at risk. In this study, we selected particularly important organs at risk, including the spinal cord, brainstem, whole lungs, heart, and esophagus, for evaluation (Table 3). As shown in the DVH diagram (Figure 6). 

## _3.2. Comparison with State-Of-The-Art Methods_ 

To verify the performance advantages of the proposed model in dose prediction tasks, we conducted a comprehensive comparison with current mainstream advanced models, including UNet [33], GAN [34], DeepLabV3+ [35], MD-Dose [21], DoseDiff [20] and DoseNet [36]. The experimental results, as shown in Tables 3 and 4, demonstrate that the proposed model exhibits significant advantages across multiple core metrics. In terms of the MAE metric, the error values of the proposed model on the public dataset, the Affiliated Hospital of Xiangnan University, Chenzhou Third People’s Hospital, and Jiangxi Provincial Cancer Hospital are 0.101, 0.103, 0.139, and 0.154, respectively, all lower than those 

11 

**Table 2:** Comparison of Dose Prediction Performance Across Multiple Datasets and Models 

|**Model**||**Performa**|**nce Metrics**||
|---|---|---|---|---|
||**MAE (Gy)**_↓_|**DICE**_↑_|**HD95 (mm)**_↓_|**Time (s)**_↓_|
|**AAPM GDP-HMM 20**|**25 (Public Dataset, N=2877)**||||
|Unet [33]|0.316<sup>_∗∗∗_</sup>|0.439<sup>_∗∗_</sup>|10.439<sup>_∗∗_</sup>|**5.370**<sup>_∗∗_</sup>|
|GAN [34]|0.169|0.847<sup>_∗_</sup>|11.854<sup>_∗∗_</sup>|7.457<sup>_∗∗_</sup>|
|DeepLabV3 [35]|0.297<sup>_∗_</sup>|0.602<sup>_∗_</sup>|17.143<sup>_∗∗_</sup>|7.054<sup>_∗∗_</sup>|
|DoseNet [36]|0.156<sup>_∗_</sup>|0.859<sup>_∗_</sup>|9.148<sup>_∗_</sup>|8.443<sup>_∗∗_</sup>|
|DoseDiff [20]|0.118|0.912|9.850<sup>_∗_</sup>|15.300<sup>_∗_</sup>|
|MD-Dose [21]|0.125<sup>_∗_</sup>|0.905|10.210<sup>_∗_</sup>|12.500<sup>_∗_</sup>|
|Baseline|0.150<sup>_∗_</sup>|0.882<sup>_∗_</sup>|9.075<sup>_∗_</sup>|13.301<sup>_∗_</sup>|
|**Proposed**|**0.101**|**0.927**|**8.947**|22.504|
|**Xiangnan University H**|**ospital (N=300)**||||
|Unet [33]|0.305<sup>_∗∗∗_</sup>|0.428<sup>_∗∗∗_</sup>|23.348<sup>_∗∗∗_</sup>|**4.135**<sup>_∗∗∗_</sup>|
|GAN [34]|0.170<sup>_∗_</sup>|0.835<sup>_∗_</sup>|15.254<sup>_∗∗_</sup>|7.100<sup>_∗∗_</sup>|
|DeepLabV3 [35]|0.225<sup>_∗∗_</sup>|0.715<sup>_∗_</sup>|17.054<sup>_∗∗_</sup>|7.067<sup>_∗∗_</sup>|
|DoseNet [36]|0.155<sup>_∗_</sup>|0.882<sup>_∗_</sup>|10.117<sup>_∗_</sup>|8.792<sup>_∗∗_</sup>|
|DoseDiff [20]|0.134<sup>_∗_</sup>|0.886<sup>_∗_</sup>|10.415<sup>_∗_</sup>|15.300<sup>_∗_</sup>|
|MD-Dose [21]|0.142<sup>_∗_</sup>|0.879<sup>_∗_</sup>|10.845<sup>_∗_</sup>|12.500<sup>_∗_</sup>|
|Baseline|0.148<sup>_∗_</sup>|0.876<sup>_∗_</sup>|10.218<sup>_∗_</sup>|13.324<sup>_∗_</sup>|
|**Proposed**|**0.103**|**0.931**|**8.672**|23.216|
|**Chenzhou Third Peopl**|**e’s Hospital (N=50)**||||
|Unet [33]|0.331<sup>_∗∗∗_</sup>|0.483<sup>_∗∗∗_</sup>|23.378<sup>_∗∗_</sup>|**6.294**<sup>_∗∗_</sup>|
|GAN [34]|0.178<sup>_∗_</sup>|0.823|15.891<sup>_∗_</sup>|7.672<sup>_∗∗_</sup>|
|DeepLabV3 [35]|0.313<sup>_∗∗∗_</sup>|0.564<sup>_∗∗_</sup>|16.254<sup>_∗_</sup>|7.293<sup>_∗∗_</sup>|
|DoseNet [36]|0.171<sup>_∗_</sup>|0.819<sup>_∗_</sup>|11.082|8.037<sup>_∗∗_</sup>|
|DoseDiff [20]|0.152<sup>_∗_</sup>|0.891|10.575|15.294<sup>_∗_</sup>|
|MD-Dose [21]|0.147|0.893|10.858|12.708<sup>_∗_</sup>|
|Baseline|0.164<sup>_∗_</sup>|0.833|10.992|13.280<sup>_∗_</sup>|
|**Proposed**|**0.139**|**0.897**|**9.217**|25.726|
|**Jiangxi Cancer Hospit**|**al (N=100)**||||
|Unet [33]|0.228<sup>_∗∗_</sup>|0.676<sup>_∗∗_</sup>|21.054<sup>_∗∗∗_</sup>|**5.764**<sup>_∗∗∗_</sup>|
|GAN [34]|0.217<sup>_∗∗_</sup>|0.831<sup>_∗_</sup>|15.920<sup>_∗_</sup>|7.672<sup>_∗∗_</sup>|
|DeepLabV3 [35]|0.476<sup>_∗∗∗_</sup>|0.479<sup>_∗∗∗_</sup>|23.376<sup>_∗∗_</sup>|10.295<sup>_∗∗_</sup>|
|DoseNet [36]|0.169|0.872|10.081<sup>_∗_</sup>|8.397<sup>_∗∗_</sup>|
|DoseDiff [20]|0.128<sup>_∗_</sup>|0.898|9.965|15.300|
|MD-Dose [21]|0.136<sup>_∗_</sup>|0.891|10.375<sup>_∗_</sup>|12.500<sup>_∗_</sup>|
|Baseline|0.173<sup>_∗_</sup>|0.838<sup>_∗_</sup>|10.692<sup>_∗_</sup>|14.067<sup>_∗_</sup>|
|**Proposed**|**0.154**|**0.894**|**9.662**|25.982|



**Note:** 

- **Bold** values indicate best performance in each metric per dataset. 

- _↓_ indicates lower values are better; _↑_ indicates higher values are better. 

- Statistical significance vs. Proposed method:<sup>_∗_</sup> _p <_ 0 _._ 05,<sup>_∗∗_</sup> _p <_ 0 _._ 01,<sup>_∗∗∗_</sup> _p <_ 0 _._ 001. 

- MAE: Mean Absolute Error (dose accuracy); DICE: Spatial overlap; HD95: 95% Hausdorff Distance (shape agreement). 

- Inference time measured on single NVIDIA RTX 4090 GPU. 

12 

|Δ_V_30 **%**<br>**(Esophagus)**<br> 0_._116_±_0_._082<br> 0_._111_±_0_._065<br>0_._187_±_0_._040<br>0_._182_±_0_._051<br>0_._095_±_0_._042<br>0_._112_±_0_._048<br>**0.054**_±_**0.039**<br>0_._082_±_0_._054<br> 0_._129_±_0_._022<br>0_._115_±_0_._066<br>0_._194_±_0_._092<br> 0_._179_±_0_._065<br>0_._108_±_0_._058<br>0_._125_±_0_._065<br>**0**_._**064**_±_**0**_._**054**<br> 0_._107_±_0_._047|0_._144_±_0_._028<br>0_._131_±_0_._063<br>0_._287_±_0_._049<br>0_._179_±_0_._047<br> 0_._115_±_0_._052<br>0_._132_±_0_._060<br>**0**_._**061**_±_**0**_._**047**<br> 0_._089_±_0_._057<br> 0_._151_±_0_._093<br>0_._147_±_0_._091<br>0_._177_±_0_._032<br> 0_._101_±_0_._041<br> 0_._098_±_0_._046<br> 0_._115_±_0_._053<br>**0**_._**071**_±_**0**_._**031**<br> 0_._091_±_0_._042|
|---|---|
|Δ_D_**max Gy**<br>**(Spinal cord)**<br>_∗_0_._152_±_0_._041 <br>_∗_0_._196_±_0_._029 <br>_∗_0_._211_±_0_._032_∗_<br>_∗_0_._166_±_0_._025_∗_<br>_∗_0_._147_±_0_._022_∗_<br>_∗_0_._166_±_0_._028_∗_<br>_∗_0_._172_±_0_._024_∗_<br>**0.103**_±_**0.017**<br><br>0_._129_±_0_._054 <br>_∗_0_._221_±_0_._023_∗_<br>_∗_0_._261_±_0_._035_∗_<br>_∗_0_._188_±_0_._072 <br>0_._162_±_0_._045_∗_<br>0_._181_±_0_._052_∗_<br>_∗_0_._166_±_0_._020_∗_<br> **0**_._**103**_±_**0**_._**017**|0_._129_±_0_._049 <br>_∗_0_._238_±_0_._036_∗_<br>_∗_0_._255_±_0_._038_∗_<br>0_._201_±_0_._030_∗_<br>0_._168_±_0_._035 <br>0_._188_±_0_._042_∗_<br>_∗_0_._194_±_0_._041_∗_<br> **0**_._**103**_±_**0**_._**027** <br>0_._168_±_0_._041 <br>_∗_0_._218_±_0_._033_∗_<br>_∗_0_._202_±_0_._013_∗_<br><br>0_._147_±_0_._063 <br>0_._152_±_0_._055 <br>0_._172_±_0_._065 <br>_∗_0_._206_±_0_._021_∗_<br> **0**_._**128**_±_**0**_._**047**|
|Δ_D_**max Gy**<br>**(Brainstem)**<br>_∗_0_._661_±_0_._052_∗_<br>_∗_2_._165_±_0_._013_∗∗_<br>_∗_2_._186_±_0_._035_∗∗_<br>_∗_2_._109_±_0_._021_∗∗_<br>_∗_0_._452_±_0_._018_∗_<br>_∗_0_._485_±_0_._025_∗_<br>_∗_0_._861_±_0_._102_∗_<br>**0.121**_±_**0.017**<br>_∗_0_._683_±_0_._063_∗_<br>_∗_2_._127_±_0_._065_∗∗_<br>_∗_2_._178_±_0_._083_∗∗_<br>_∗_2_._154_±_0_._021_∗∗_<br>_∗_<br>0_._478_±_0_._035<br>_∗_<br>0_._512_±_0_._042<br>_∗_0_._733_±_0_._028_∗_<br>**0**_._**121**_±_**0**_._**017**|0_._691_±_0_._039<br>_∗_2_._186_±_0_._028_∗_<br>_∗_2_._203_±_0_._030_∗_<br>_∗_2_._160_±_0_._041_∗_<br>_∗_<br>0_._168_±_0_._035<br>_∗_<br>0_._518_±_0_._045<br>_∗_2_._572_±_0_._024_∗_<br>**0**_._**145**_±_**0**_._**022**<br><br>0_._621_±_0_._304<br>_∗_2_._164_±_0_._125_∗_<br>_∗_2_._656_±_0_._206_∗∗_<br>_∗_2_._314_±_0_._211_∗_<br>_∗_<br>0_._468_±_0_._155<br>_∗_<br>0_._502_±_0_._175<br>_∗_2_._519_±_0_._062_∗∗_<br>**0**_._**162**_±_**0**_._**109**|
|Δ_V_30 **%**<br>**(Heart)**<br>3_._454_±_0_._215_∗∗_<br> 3_._044_±_0_._106_∗∗_<br>3_._128_±_0_._198_∗∗_<br> 3_._874_±_0_._151_∗∗_<br> 2_._945_±_0_._008_∗∗_<br> 3_._125_±_0_._142_∗∗_<br>3_._174_±_0_._211_∗∗_<br>**0.801**_±_**0.116**<br>**fi    ity**<br>3_._548_±_0_._283_∗∗_<br>3_._144_±_0_._137_∗∗_<br>3_._223_±_0_._158_∗∗_<br>2_._996_±_0_._154_∗∗_<br>2_._875_±_0_._168_∗∗_<br>3_._055_±_0_._185_∗∗_<br>2_._898_±_0_._139_∗∗_<br> **0**_._**851**_±_**0**_._**116**|2_._742_±_0_._261_∗∗_<br>3_._258_±_0_._189_∗∗_<br>3_._355_±_0_._203_∗∗_<br>3_._165_±_0_._160_∗∗_<br>3_._485_±_0_._308_∗∗_<br> 3_._075_±_0_._192_∗∗_<br>3_._099_±_0_._106_∗∗_<br> **0**_._**968**_±_**0**_._**145**<br>3_._215_±_0_._272_∗∗_<br>3_._207_±_0_._156_∗∗_<br>3_._619_±_0_._141_∗∗_<br>2_._942_±_0_._131_∗∗_<br>2_._825_±_0_._145_∗∗_<br>3_._005_±_0_._162_∗∗_<br>3_._209_±_0_._118_∗∗_<br> **0**_._**804**_±_**0**_._**162**|
|Δ_D_**max**<br>Δ_V_20 **%**<br>**(Lung)**<br>**AAPM GDP-HMM 2025 (Public)**<br>0_._215_±_0_._0024_∗∗_<br>2_._151_±_0_._524_∗_<br>0_._155_±_0_._0017_∗∗∗_<br>1_._576_±_0_._211 <br>0_._183_±_0_._0009_∗∗∗_1_._683_±_0_._212_∗_<br>0_._011_±_0_._00013_∗∗∗_1_._325_±_0_._151 <br>0_._008_±_0_._0001_∗_<br>1_._215_±_0_._035 <br>0_._009_±_0_._00012_∗_<br>1_._285_±_0_._148 <br>0_._0051_±_0_._0006_∗_<br>1_._855_±_0_._124_∗_<br>**0.0005**_±_**0.00006**<br>**1.113**_±_**0.174**<br>**e Affiliated Hospital of Xiangnan Univers**<br>0_._226_±_0_._0014_∗∗∗_2_._367_±_0_._635_∗∗_<br>0_._118_±_0_._0028_∗∗_<br>1_._762_±_0_._752_∗_<br>0_._127_±_0_._0014_∗∗∗_1_._884_±_0_._727_∗_<br>0_._0015_±_0_._00045_∗∗_1_._425_±_0_._281_∗_<br>0_._010_±_0_._00015_∗∗_1_._335_±_0_._245_∗_<br>0_._011_±_0_._00018_∗∗_1_._405_±_0_._268_∗_<br> **0**_._**0008**_±_**0**_._**00006** 1_._492_±_0_._198_∗_<br> 0_._0008_±_0_._00007<br>**1**_._**013**_±_**0**_._**174**|**Chenzhou Third People’s Hospital**<br>0_._028_±_0_._00050_∗∗∗_2_._368_±_0_._329_∗∗_<br>0_._019_±_0_._00030_∗∗∗_1_._810_±_0_._238_∗_<br>0_._025_±_0_._00020_∗∗∗_2_._303_±_0_._537_∗∗_<br>0_._0018_±_0_._00018<br>1_._795_±_0_._204_∗_<br>0_._0025_±_0_._0018_∗∗∗_2_._895_±_0_._175_∗∗_<br>0_._012_±_0_._0003_∗∗_<br>1_._495_±_0_._240 <br> 0_._0012_±_0_._00019 1_._890_±_0_._427_∗∗_<br> **0**_._**0010**_±_**0**_._**00010 1**_._**045**_±_**0**_._**282**<br>**Jiangxi Provincial Cancer Hospital**<br>0_._024_±_0_._00035_∗∗∗_2_._628_±_0_._304_∗∗_<br>0_._017_±_0_._00020_∗∗∗_1_._836_±_0_._264_∗∗_<br>0_._020_±_0_._00012_∗∗∗_1_._764_±_0_._526_∗_<br>0_._0013_±_0_._00014<br>1_._698_±_0_._251_∗_<br>0_._009_±_0_._00016<br>1_._385_±_0_._205_∗_<br>0_._010_±_0_._0002<br>1_._455_±_0_._228_∗_<br>0_._0014_±_0_._00017 1_._830_±_0_._644_∗∗_<br> **0**_._**0007**_±_**0**_._**00008 1**_._**062**_±_**0**_._**115**|
|Δ_D_2<br>0_._197_±_0_._0022_∗∗_<br>0_._083_±_0_._0035_∗∗_<br>0_._043_±_0_._0014_∗∗_<br>0_._012_±_0_._0007_∗_<br>0_._008_±_0_._0005_∗_<br>0_._009_±_0_._0006_∗_<br>0_._008_±_0_._00007_∗_<br>**0.001**_±_**0.00001**<br>**Th fi**<br>0_._208_±_0_._0031_∗∗∗_<br>0_._004_±_0_._0006_∗_<br>0_._004_±_0_._0005_∗_<br>0_._003_±_0_._0001_∗∗_<br>0_._010_±_0_._0008_∗∗_<br>0_._011_±_0_._0009<br>0_._002_±_0_._00002 <br> **0**_._**002**_±_**0**_._**00001**|0_._009_±_0_._0012_∗_<br>0_._005_±_0_._0007_∗_<br>0_._006_±_0_._0008_∗_<br>0_._004_±_0_._00015<br>0_._011_±_0_._0002_∗∗_<br>0_._012_±_0_._00025_∗∗_<br>**0**_._**001**_±_**0**_._**00002** <br>0_._003_±_0_._00003 <br>0_._007_±_0_._0009_∗_<br>0_._004_±_0_._0006_∗_<br>0_._004_±_0_._0005_∗_<br>0_._003_±_0_._00009<br>0_._009_±_0_._00012_∗_<br>0_._010_±_0_._00015_∗_<br>0_._005_±_0_._00002_∗_<br> **0**_._**002**_±_**0**_._**00002 **|
|Δ**HI**<br>Δ_D_98<br>0_._076_±_0_._015_∗∗_0_._152_±_0_._008_∗∗_<br>0_._042_±_0_._031_∗_<br>0_._037_±_0_._015_∗_<br>35] 0_._051_±_0_._026_∗_<br>0_._091_±_0_._024_∗∗_<br><br>0_._042_±_0_._018_∗_<br>0_._032_±_0_._017_∗_<br>0]<br>0_._039_±_0_._012<br>0_._029_±_0_._014<br>21] 0_._041_±_0_._015_∗_<br>0_._031_±_0_._016_∗_<br>0_._039_±_0_._026<br>0_._046_±_0_._036_∗_<br>**0.038**_±_**0.004**<br>**0.024**_±_**0.011**<br>0_._082_±_0_._020_∗∗_0_._161_±_0_._012_∗∗∗_<br>0_._053_±_0_._032_∗_<br>0_._042_±_0_._021_∗_<br>35]<br>0_._044_±_0_._031<br>0_._052_±_0_._031_∗_<br><br>0_._051_±_0_._021<br>0_._036_±_0_._019<br>0]<br>0_._045_±_0_._018<br>0_._034_±_0_._020<br>21]<br>0_._047_±_0_._020<br>0_._036_±_0_._022<br>0_._048_±_0_._006<br>0_._039_±_0_._013<br>**0**_._**040**_±_**0**_._**004**<br>**0**_._**029**_±_**0**_._**011 **|0_._085_±_0_._023_∗∗∗_0_._164_±_0_._015_∗∗∗_<br>0_._055_±_0_._033_∗_<br>0_._044_±_0_._023_∗_<br>35] 0_._070_±_0_._035_∗_0_._098_±_0_._035_∗∗∗_<br><br>0_._054_±_0_._025_∗_<br>0_._038_±_0_._029<br>0]<br>0_._049_±_0_._022_∗_<br>0_._036_±_0_._025<br>21] 0_._051_±_0_._025_∗_<br>0_._038_±_0_._028<br>0_._051_±_0_._038_∗_<br>0_._032_±_0_._022<br>**0**_._**045**_±_**0**_._**008**<br>**0**_._**028**_±_**0**_._**015**<br>0_._079_±_0_._018_∗∗_0_._157_±_0_._010_∗∗∗_<br>0_._050_±_0_._030_∗_0_._099_±_0_._018_∗∗∗_<br>35] 0_._060_±_0_._050_∗_<br>0_._036_±_0_._027_∗_<br><br>0_._049_±_0_._019_∗_<br>0_._044_±_0_._028_∗_<br>0]<br>0_._044_±_0_._016<br>0_._033_±_0_._024<br>21]<br>0_._046_±_0_._018<br>0_._035_±_0_._027<br>0_._052_±_0_._020_∗_<br>0_._034_±_0_._016<br>**0**_._**040**_±_**0**_._**005**<br>**0**_._**031**_±_**0**_._**012 **|
|**Methods**<br>Unet [33]<br>GAN [34]<br>deepLabV3[<br>DoseNet[36]<br>DoseDiff [2<br>MD-Dose [<br>Baseline<br>proposed<br>Unet [33]<br>GAN [34]<br>deepLabV3[<br>DoseNet[36]<br>DoseDiff [2<br>MD-Dose [<br>Baseline<br>proposed|Unet [33]<br>GAN [34]<br>deepLabV3[<br>DoseNet[36]<br>DoseDiff [2<br>MD-Dose [<br>Baseline<br>proposed<br>Unet [33]<br>GAN [34]<br>deepLabV3[<br>DoseNet[36]<br>DoseDiff [2<br>MD-Dose [<br>Baseline<br>proposed|



13 

of other compared models (e.g., DoseNet has an MAE of 0.156 on the public dataset). Notably, on the two key metrics of ΔHI (0 _._ 038 _±_ 0 _._ 004) and Δ _D_ 98 (0 _._ 024 _±_ 0 _._ 011), compared to DoseNet, which ranked second (ΔHI = 0 _._ 042 _±_ 0 _._ 018, Δ _D_ 98 = 0 _._ 032 _±_ 0 _._ 017), the proposed model reduced the values by 0.004 and 0.008, respectively, showing a significant improvement in error control capability. Additionally, on the Δ _D_ 2 (0 _._ 001 _±_ 0 _._ 001) and Δ _D_ max (0 _._ 0005 _±_ 0 _._ 0006) metrics, the values of the proposed model are only 1/10 to 1/3 of those of other advanced models, highlighting its superior spatial localization accuracy. Through paired t-test analysis, the differences between the proposed model and models such as UNet, GAN, DeepLabV3+, and DoseNet in terms of metrics like MAE, DICE, and HD95 are statistically significant ( _P <_ 0 _._ 05), proving that the performance improvements are not coincidental. To further benchmark our model against contemporary diffusion-based approaches, we compared ADDiff-Dose with the reported performance of DoseDiff [20] and MD-Dose [21] on the AAPM dataset. As shown in Table 2, ADDiff-Dose achieves a superior MAE of 0.101 and DICE of 0.927, compared to DoseDiff (MAE: 0.118) and MD-Dose (DICE: 0.905), underscoring the benefit of our dual-constraint design and lightweight VAE. 

In addition, Figures 4 and 5 present the prediction results of models including UNet, DeepLabV3, GAN, DoseNet, MD-Dose, DoseDiff, Baseline, and the Proposed model. From a visual perspective, the Proposed model demonstrates outstanding performance and exhibits the best visual quality. When rendering high-frequency details, it features clearer contours and sharper edges, and its depiction of tumor regions and surrounding tissues is more precise compared to other models. Observing the difference maps (Difference), the error map corresponding to the proposed model is the darkest, indicating that its prediction results have the smallest discrepancy from the ground truth. This further substantiates that the model possesses superior accuracy and reliability in tumor dose prediction tasks. Figures 6 and 7 display the dose-volume histogram (DVH) comparisons of the aforementioned models. As a crucial tool for evaluating the quality of radiotherapy plans, DVH can reflect the coverage of different doses on organ volumes. Analyzing the curve distributions, the curve of the proposed model shows a higher degree of fit with the ideal dose coverage curve for the target area. While ensuring adequate dose delivery to the target, it provides better dose control for OARs, with a curve decline that more closely aligns with dose limitation requirements. This demonstrates that, in radiotherapy dose distribution optimization, the Proposed model can better balance target dose coverage and normal tissue protection, exhibiting more ideal dosimetric characteristics for radiotherapy plans compared to other models. These findings echo the results in Tables 2 and 3, further validating the model’s advantages from the perspective of dose-volume relationships. 

We analyzed the computational efficiency of ADDiff-Dose. Training the full model required approximately 200 hours on a single NVIDIA RTX 4090 GPU. During inference, the average time to generate a complete 3D dose distribution for one case was 22 seconds, which is clinically feasible for generating initial plans. While this is slower than U-Net (4 seconds), the significant gain in prediction accuracy and clinical compliance justifies the trade-off for application in automated planning workflows. 

## **Qualitative Comparison with Advanced Diffusion Models** 

To further confirm that ADDiff-Dose generates structurally superior dose distributions, we conducted visual comparisons with advanced diffusion models (DoseDiff [13], MD-Dose [14]) on the AAPM public dataset. The results are shown in Figure 4 and 5. Key observations from the visual comparison: 

1. **Anatomical detail preservation:** ADDiff-Dose-generated dose distributions more accurately match the boundary contours of PTV and OARs (e.g., spinal cord, parotid gland), while DoseDiff and MD-Dose show slight blurring in fine anatomical regions (e.g., PTV edge). 

2. **Dose gradient accuracy:** In high-dose gradient regions (e.g., tumor edge, OAR adjacent areas), ADDiff-Dose 

14 



<!-- Start of picture text -->
eeree<br><!-- End of picture text -->

**Figure 4:** Visualization and Comparison of Dose Prediction Results and Differences of Different Models for Lung Tumors. Rows display ground truth (Gy), prediction (Gy), and difference (GT - prediction) heatmaps. Columns compare models (UNet, DeepLabV3, GAN, DoseNet, DoseDiff, MD-Dose, Baseline, Proposed), with color bars indicating dose value scales, enabling assessment of prediction accuracy and model performance differences. 



<!-- Start of picture text -->
cs<br><!-- End of picture text -->

**Figure 5:** Visualization and Comparison of Dose Prediction Results and Differences of Different Models for Head and Neck Tumors. The figure presents three rows of heatmaps: the top row shows ground truth dose distributions (Gy), the middle row displays dose predictions from models (UNet, DeepLabV3, GAN, DoseNet, DoseDiff, MD-Dose, Baseline, Proposed), and the bottom row illustrates the difference (ground truth - prediction). Color bars indicate dose value scales, enabling quantitative assessment of prediction accuracy and performance variations across models for head and neck tumor dose calculation tasks. 

15 



**Figure 6:** Comparison of Dose - Volume Histograms (DVH) for Lung Tumors Among Multiple Models. The figure presents DVH plots for six models (UNet, DeepLabV3, GAN, DoseNet, DoseDiff, MD-Dose, Baseline, Proposed). Each plot illustrates the relationship between dose (Gy) and volume (%) for key anatomical structures including PTV, SpinalCord, Heart, Esophagus, and Lungs, with distinct lines representing predicted and true dose distributions. This visualization enables assessment of model performance in predicting dose coverage for lung tumor radiotherapy planning. 



<!-- Start of picture text -->
UN<br>AYAYAYAN<br><!-- End of picture text -->

**Figure 7:** Comparison of Dose - Volume Histograms (DVH) for Head and Neck Tumors Among Multiple Models. The figure displays DVH plots for six models (UNet, DeepLabV3, GAN, DoseNet, DoseDiff, MD-Dose, Baseline, Proposed). Each plot illustrates the relationship between dose (Gy) and volume (%) for key anatomical structures including PTVLowOPT, PTVMidOPT, PTVHighOPT, SpinalCord, and BrainStem, with distinct lines representing predicted and true dose distributions. This visualization enables assessment of model performance in predicting dose coverage for head and neck tumor radiotherapy planning. 

16 

maintains a steeper and more accurate dose gradient, consistent with clinical requirements for “tumor high-dose coverage and OAR low-dose protection”. In contrast, DoseDiff and MD-Dose have gentler gradients and partial dose leakage to OARs. 

3. **Difference map analysis:** The difference between ADDiff-Dose and the ground truth is mainly concentrated in low-dose regions, with maximum difference values _<_ 0 _._ 5 Gy. DoseDiff and MD-Dose have larger difference areas in high-dose regions, with maximum differences exceeding 1 _._ 0 Gy, which may lead to clinical dose errors. 

These visual results further confirm that the dual-constraint design of ADDiff-Dose effectively improves the structural fidelity and clinical compliance of dose distributions compared to advanced diffusion models. 

## **Model Randomness and Uncertainty Analysis** 

1. **Quantitative variability across predictions:** Supplementary Table 2 summarizes the coefficient of variation (CV) of key dosimetric and geometric metrics across 10 independent inference runs. The CV values are consistently low across all metrics: MAE (2 _._ 3% _±_ 0 _._ 5%), DICE similarity (1 _._ 8% _±_ 0 _._ 4%), PTV _D_ 95 (1 _._ 2% _±_ 0 _._ 3%), and spinal cord _Dmax_ (2 _._ 1% _±_ 0 _._ 6%). _All CVs are below the 3% threshold_ , indicating excellent predictive stability and minimal randomness in model inference. 

2. **Spatial uncertainty distribution:** The prediction uncertainty, quantified as voxel-wise variance from 10 independent runs, is analyzed by anatomical region in Supplementary Table 3. The analysis reveals a clinically favorable uncertainty distribution pattern: _the highest uncertainty_ is confined to peripheral low-dose regions of the PTV (mean variance < 0 _._ 01 Gy<sup>2</sup> ), while _clinically critical regions_ —including the PTV core and all OARs (spinal cord, brainstem, lungs)—exhibit negligible uncertainty levels ( _<_ 0 _._ 002 Gy<sup>2</sup> ). This confirms that model uncertainty is strategically concentrated in regions of minimal clinical significance. 

This analysis demonstrates that ADDiff-Dose maintains good stability while retaining the generative characteristics of diffusion models. The uncertainty distribution is consistent with clinical attention priorities—low uncertainty in critical regions (PTV core, OARs) ensures clinical reliability, while slight uncertainty in non-critical low-dose regions does not affect treatment safety. Furthermore, the low coefficient of variation across repeated inferences (Supplementary Table 2) and the clinically favorable spatial distribution of uncertainty (Supplementary Table 3) jointly reinforce that ADDiff-Dose not only achieves high accuracy but also delivers consistent and reliable predictions in critical anatomical regions—a prerequisite for trustworthy clinical deployment. The uncertainty map can provide reference information for clinicians to assess the reliability of dose prediction. 

## _3.3. Ablation Experiments_ 

To comprehensively evaluate the contribution of each architectural component, we conducted systematic ablation studies on the AAPM public dataset. We trained seven variants of ADDiff-Dose by incrementally adding core components: (1) Model 1 with only LightweightVAE3D; (2) Model 2 adding multi-head attention; (3) Model 3 incorporating clinical constraints; (4) Model 4 further adding anatomical conditions; (5) Model 5 with all components except multi-head attention; (6) Model 6 with all components except LightweightVAE3D; and (7) Model 7 with all components except both attention and anatomical conditions. 

Results in Table 4 demonstrate the progressive improvement with each added component. The stepwise integration from Model 1 to Model 4 shows consistent performance gains, with MAE decreasing from 0.185 to 0.115 Gy ( _p <_ 0 _._ 05) and DICE improving from 0.872 to 0.927. Notably, the addition of clinical constraints (Model 3) significantly reduced spinal cord Δ _D_ max error from 0.198 to 0.145 Gy ( _P <_ 0 _._ 01), while anatomical conditions (Model 4) further optimized 

17 

target coverage, reducing PTV Δ _D_ 98 from 0.035 to 0.029 Gy. The comparison between Model 6 (without VAE) and the full model highlights the critical balance between efficiency and accuracy—while Model 6 achieved the fastest inference time (16.2 s), it sacrificed significant accuracy across all metrics ( _P <_ 0 _._ 05). The full ADDiff-Dose model achieved the best overall performance, validating the synergistic effect of all integrated components. 

**Table 4:** Ablation Study on Key Components of ADDiff-Dose (Evaluated on AAPM Public Dataset) 

|C||||Model Ci|onfiguration||||
|---|---|---|---|---|---|---|---|---|
|omponent|||||||||
||Model 1|Model 2|Model 3|Model 4|Model 5|Model 6|Model 7|**Full Model**|
|**Architecture Components**|||||||||
|LightweightVAE3D|✓|✓|✓|✓|✓||✓|✓|
|Multi-head Attention||✓|✓|✓||✓|✓|✓|
|Clinical Constraints|||✓|✓|✓|✓|✓|✓|
|Anatomical Conditions||||✓|✓|✓|✓|✓|
|**Performance Metrics**|||||||||
|MAE (Gy)|0.185<sup>_∗∗∗_</sup>|0.162<sup>_∗∗∗_</sup>|0.134<sup>_∗∗_</sup>|0.115<sup>_∗_</sup>|0.142<sup>_∗∗∗_</sup>|0.126<sup>_∗∗_</sup>|0.138<sup>_∗∗∗_</sup>|**0.101**|
|DICE|0.872<sup>_∗∗∗_</sup>|0.891<sup>_∗∗∗_</sup>|0.908<sup>_∗∗_</sup>|0.919|0.895<sup>_∗∗∗_</sup>|0.912<sup>_∗_</sup>|0.901<sup>_∗∗∗_</sup>|**0.927**|
|HD95 (mm)|10.89<sup>_∗∗∗_</sup>|9.876<sup>_∗∗∗_</sup>|9.234<sup>_∗∗_</sup>|8.967|9.543<sup>_∗∗∗_</sup>|9.125<sup>_∗_</sup>|9.421<sup>_∗∗∗_</sup>|**8.672**|
|Δ_D_max (Spinal Cord) (Gy)|0.234<sup>_∗∗∗_</sup>|0.198<sup>_∗∗∗_</sup>|0.145<sup>_∗∗_</sup>|0.118|0.168<sup>_∗∗∗_</sup>|0.132<sup>_∗_</sup>|0.156<sup>_∗∗∗_</sup>|**0.103**|
|Δ_V_20 (Lung) (%)|1.892<sup>_∗∗∗_</sup>|1.654<sup>_∗∗∗_</sup>|1.325<sup>_∗∗_</sup>|1.187|1.432<sup>_∗∗∗_</sup>|1.245<sup>_∗_</sup>|1.387<sup>_∗∗∗_</sup>|**1.113**|
|Δ_D_98 (PTV) (Gy)|0.049<sup>_∗∗∗_</sup>|0.042<sup>_∗∗_</sup>|0.035|0.029|0.038<sup>_∗∗_</sup>|0.032|0.036<sup>_∗_</sup>|**0.024**|
|Time (s)|21.3|22.1|22.5|22.8|18.9<sup>_∗_</sup>|**16.2**<sup>_∗∗_</sup>|19.4<sup>_∗_</sup>|22.5|



**Note:** All experiments were conducted on the AAPM GDP-HMM 2025 public dataset. ✓ indicates the component is included. Best results in **bold** . Significance vs. Full Model:<sup>_∗_</sup> _p <_ 0 _._ 05,<sup>_∗∗_</sup> _p <_ 0 _._ 01,<sup>_∗∗∗_</sup> _p <_ 0 _._ 001. Lower is better for all metrics. Δ = absolute error. 

## **4. Discussion** 

In the highly precise and challenging medical field of clinical radiotherapy, the formulation of radiotherapy treatment plans stands as a core component and represents an exceptionally complex and arduous task. Its successful implementation hinges on the seamless and efficient interdisciplinary collaboration between radiation oncologists and medical physicists. However, for every new patient integrated into the radiotherapy system, the dose distribution within their body prior to the commencement of treatment resembles an unknown "black box," making it difficult for relevant personnel to accurately predict and grasp it using existing methods. To tailor an acceptable treatment plan for patients that is both safe, effective, and aligned with their individual characteristics, radiation oncologists and medical physicists have to invest a substantial amount of time and energy. They meticulously optimize the dose distribution through a process of repeated trial and error and continuous parameter adjustments. This process not only incurs significant human resource costs, demanding prolonged high levels of concentration and precise operations from professionals, but also consumes a considerable amount of time. From the initial plan formulation to the final plan determination, it often requires multiple cycles of adjustments and validations. Moreover, it is accompanied by high economic costs, encompassing equipment usage, personnel salaries, and potential additional expenses arising from multiple examinations and adjustments, imposing a heavy burden on both medical resources and patients’ families[37]. To effectively address these long-standing challenges in clinical practice, in this study, we innovatively proposed an end-to-end automatic dose prediction model named the Conditional Diffusion Model with Anatomical-Dose Dual Constraints for End-to-End Multi-Tumor Dose Prediction (ADDiff-Dose) based on 

18 

cutting-edge deep learning theories and professional knowledge in the field of radiotherapy. The core objective of this study is to develop the model into a clinically practical guidance tool that provides accurate and efficient support for the formulation of radiotherapy treatment plans, thereby alleviating the complexity and inefficiency of the current workflow. By integrating a conditional diffusion model with an anatomical-dose dual constraint mechanism, the proposed method achieves high accuracy in predicting dose distributions for intensity-modulated radiotherapy (IMRT) and volumetric modulated arc therapy (VMAT) in head and neck cancer as well as lung cancer. Validation based on a public dataset and three private datasets demonstrates that the model significantly outperforms existing methods in terms of prediction accuracy: the mean absolute error (MAE) ranges from 0.101 to 0.154, the Dice similarity coefficient reaches 0.897–0.931, and the 95% Hausdorff distance (HD95) ranges from 8.672 to 9.217—surpassing state-of-the-art models such as U-Net [33] , generative adversarial networks (GANs) [34] , DeepLabV3+ [35], and DoseNet [36]. Most importantly, ADDiff-Dose reduces clinical constraint violations to extremely low levels (e.g., spinal cord Δ _D_ max of 0.103–0.128 Gy, lung Δ _V_ 20 of 1.013–1.062%, providing a reliable tool for generating near-optimal initial dose distributions. This advancement can significantly optimize clinical workflows and enhance treatment accuracy [38]. We acknowledge that core components like the 3D-VAE and multi-head attention are established architectures. The novelty of ADDiff-Dose lies not in inventing new base modules, but in their purposeful integration into a novel, clinically-driven framework. The LightweightVAE3D is specifically designed and validated to solve the computational bottleneck of 3D CT data in radiotherapy, enabling efficient deployment. The multi-head attention is strategically employed to fuse multi-modal conditions, a critical need in dose prediction that has not been fully explored within diffusion models. Therefore, our primary contribution is the demonstration that this integrated, dual-constrained diffusion framework effectively addresses key clinical challenges in multi-tumor dose prediction. 

The ADDiff-Dose model takes as its basic inputs the original CT images, dose distribution maps, and segmentation masks of the target volumes OARs, which is similar to other studies in the field[39] [40] [41]. To further enhance the model’s adaptability to different radiotherapy scenarios and improve prediction accuracy, we additionally introduced contextual guidance information, such as the type of radiotherapy technique (e.g., IMRT or VMAT), treatment site (e.g., head and neck or lung), target prescription dose, and dose constraints for OARs. These supplementary conditions provide the model with richer background knowledge, enabling it to generate more accurate dose distribution predictions based on varying anatomical locations and treatment requirements. This design effectively overcomes the limitations of traditional models like U-Net and GAN, whose simplified feature extraction mechanisms struggle to capture the complex interactions between anatomical structures and dosimetric characteristics. Ma et al. used a U-Net as the backbone network and further incorporated the desired DVH into the input[42]. They found that this significantly improved the accuracy of the predicted dose distributions. 

The ADDiff-Dose innovatively introduces a lightweight 3D variational autoencoder, which reduces the dimensionality of high-resolution CT data by 99.7% while preserving key anatomical characteristics [42] [43]. This significantly lowers the computational burden in processing 3D medical images, enabling clinical-level deployment on a single NVIDIA RTX 4090 GPU [44], thereby addressing the challenge of high computational costs that have hindered the practical implementation of traditional high-precision models [45] [42]. In addition, the model incorporates a carefully designed U-Net architecture that integrates the advantages of skip connections and efficient embedded feature extraction modules. This design enables the model to keenly capture both global and local contextual information from the input data—global information helps grasp the overall trend of dose distribution[46], while local information focuses on detailed characteristics of critical regions, laying a solid foundation for accurate dose prediction. Inspired by the Transformer architecture [47] [48], the model incorporates a multi-head attention mechanism, which captures long-range dependencies between anatomical structures and dose distributions through multi-subspace mapping and independent attention weight computation. This capability 

19 

is particularly prominent in dynamic anatomical variation scenarios, such as lung cancer affected by respiratory motion [35] [49], where the model can focus on clinically critical regions—such as the boundaries of the planning target volume and adjacent areas of OARs—while filtering out irrelevant noise, thereby generating sharper and more accurate dose predictions (Figures 4–5). Compared to traditional convolutional neural networks (CNNs), which suffer from limited receptive fields and struggle to model global contextual relationships [50] [51], this mechanism significantly enhances the model’s adaptability to complex anatomical environments. In addition, to address potential issues of over-smoothing and distortion in dose prediction, this study conducted in-depth investigations and introduced a dose loss function based on the dose constraint table for organs-at-risk (OARs) in radiotherapy. This was then integrated into a composite loss function. The model ensures that the predicted results are both accurate and clinically compliant through a composite loss function composed of a reconstruction loss ( _L_ mse) and a clinical constraint term ( _L_ cond) incorporating over 50 constraints, weighted and combined. Unlike some state-of-the-art dose prediction models [52] [53] that focus solely on optimizing reconstruction accuracy while neglecting clinical feasibility, ADDiff-Dose enforces key dosimetric constraints—such as PTV _D_ 95 and lung _V_ 20—through a weighted loss mechanism. As a result, the model reduces the errors in spinal cord Δ _D_ max and lung Δ _V_ 20 to 0.103–0.128 Gy and 1.013–1.062%, respectively, representing significant improvements compared to DoseNet (Δ _D_ max: 0.147–0.188 Gy; Δ _V_ 20: 1.425–1.795%). The introduction of local constraint losses further enhances the preservation of fine details in regions with high-dose gradients, meeting the core requirements of radiotherapy planning [36] [54]. Ablation experiments confirm that removing either the OAR encoder or the clinical constraint loss leads to performance degradation, a finding consistent with Nguyen et al.’s conclusions regarding the value of anatomical priors [55]. 

Unlike previous models designed specifically for certain cancers (e.g., prostate [42] [56], head and neck [4] [57]) or techniques (e.g., IMRT [29]), ADDiff-Dose adapts to diverse anatomical and dosimetric scenarios through a prior-guided diffusion process. Its training strategy integrates data augmentation (Mixup), block-wise processing, and early stopping mechanisms, effectively enhancing the model’s robustness to anatomical variations and data heterogeneity. This addresses the common issue in radiotherapy research where limited sample sizes constrain model performance [58] [59]. By providing an initial dose distribution close to optimal, the model reduces planning time from several hours to the minute level, enabling physicists to focus on fine-tuning rather than de novo design [60] [61]. This aligns closely with the vision of precision oncology, where automated tools drive personalized and efficient treatment delivery [51]. 

Directly integrating clinical dose-volume constraints into the loss function, rather than relying on post-processing, significantly enhances the model’s applicability by enabling proactive guidance during training. This approach ensures synergistic optimization of anatomical fidelity and clinical compliance, avoiding the irreversible anatomical damage often caused by post-hoc adjustments. Furthermore, it improves computational efficiency by eliminating separate optimization steps and enhances robustness in complex scenarios (e.g., overlapping PTV/OARs) by balancing target coverage and organ protection during learning. These advantages collectively establish the integrated constraint design as a key factor in the superior performance of ADDiff-Dose compared to existing models. Moreover, the model exhibits excellent predictive stability, as evidenced by the low coefficient of variation across repeated inferences and the clinically favorable spatial distribution of uncertainty. These results reinforce that ADDiff-Dose not only achieves high accuracy but also delivers consistent and reliable predictions in critical anatomical regions—a prerequisite for trustworthy clinical deployment. 

We designed and conducted a series of experiments on both the public dataset and our internal private datasets. As shown in Tables 2 and 3, the results in the test sets are highly consistent, demonstrating that the proposed model achieves superior performance in all evaluation metrics and delivers the best overall predictive precision. This provides strong evidence that the model has excellent generalizability for new subjects and can maintain stable predictive performance in various clinical scenarios. In comparative experiments, our model was comprehensively evaluated against several 

20 

state-of-the-art (SOTA) methods on external validation datasets. Visually, the predicted dose distributions show a high degree of similarity to the ground truth. Statistically, the model also demonstrates outstanding performance, exhibiting the smallest distribution skewness among all compared models, indicating greater stability and reliability of the predictions under varying data conditions. These comprehensive and in-depth comparative results not only fully validate the significant superiority of the proposed approach but also highlight its strong generalization ability. Notably, ADDiff-Dose demonstrates superior predictive stability compared to deterministic models. The low coefficient of variation across repeated inferences and the clinically favorable spatial distribution of uncertainty indicate that the model delivers consistent and reliable predictions in critical regions—an essential characteristic for clinical adoption. The model can be readily extended to predict dose distributions for other anatomical sites, offering novel insights and methodologies for the field of medical image processing and predictive modeling. 

The ADDiff-Dose model excels in radiotherapy dose prediction but has limitations requiring further optimization. First, it is designed for IMRT and VMAT, with architecture and loss functions based on conventional fractionation. Its adaptability to hypofractionation or stereotactic body radiotherapy (SBRT), which demand high dose gradients, is untested, potentially causing prediction biases due to unmodeled technique-specific characteristics. Second, validation is limited to head-and-neck and lung tumors, with parameters optimized for their anatomical and dosimetric features. Generalizability to complex tumors like pelvic (e.g., prostate, cervical) or abdominal (e.g., liver, pancreatic) cancers, especially small-volume targets near multiple organs (e.g., skull base, spinal metastases), requires further validation. Third, robustness to clinical noise, such as CT artifacts from metal implants or anatomical deformations from respiratory motion or positioning, is insufficient. While data augmentation and local constraints improve resilience, untested noise types (e.g., metal streaks, motion-induced shifts) may disrupt tissue density estimation or target-OAR spatial relationships, reducing stability in clinical settings. Fourth, the performance of ADDiff-Dose is contingent upon the accuracy of the input PTV and OAR segmentations. Inaccurate contours would propagate errors into the conditional features and consequently the predicted dose. Future work could explore the integration of automated segmentation models or the development of more robust conditioning mechanisms that are tolerant to segmentation uncertainties. Future work will address these issues by: (1) incorporating technique-specific constraints (e.g., SBRT dose gradients) to enhance adaptability; (2) building multi-tumor, multi-site datasets with transfer learning to improve generalizability; and (3) developing noise-aware modules using adversarial samples to boost robustness against clinical artifacts. These optimizations aim to advance automated dose prediction toward comprehensive coverage of all radiotherapy techniques and tumor types. The stochastic nature of diffusion models offers advantages over deterministic architectures, as demonstrated by our uncertainty analysis. Multiple inferences on the same patient produce varied dose maps with mean DICE similarity of 0 _._ 95 _±_ 0 _._ 02 across runs, indicating low variability in high-confidence regions. This enhances clinical value by identifying regions needing physicist review, a feature absent in U-Net or GAN baselines. 

## **5. Conclusion** 

This study proposes an Anatomy-Dose Dual-Constrained Conditional Diffusion Model: the Conditional Diffusion Model with Anatomical-Dose Dual Constraints for End-to-End Multi-Tumor Dose Prediction (ADDiff-Dose) , achieving end-to-end prediction of IMRT/VMAT dose distributions for head-and-neck and lung tumors. By integrating a lightweight Variational Autoencoder (VAE), a multi-condition embedding layer, and a diffusion denoising mechanism, the model significantly outperforms mainstream approaches such as UNet and GAN on both a public dataset (MAE = 0 _._ 101) and three external hospital datasets (MAE range: 0 _._ 103 _−_ 0 _._ 154) ( _P <_ 0 _._ 05). Notably, it achieves breakthrough improvements in 

21 

key metrics like Δ _D_ max (0 _._ 0005 _±_ 0 _._ 0006) and ΔHI (0 _._ 038 _±_ 0 _._ 004). Ablation experiments demonstrate that the structural encoder enhances the Dice coefficient by 6 _._ 3% and improves clinical dose compliance by 28 _._ 5%. 

## **Declarations** 

- **Ethics approval and consent to participate** : The Ethics Committee of Xiangnan University agreed to this retrospective study (ID: AF/SC-07-4/01.0) 

- **Consent for publication** : N/A 

- **Competing interests** : All authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

- **Funding** : This study was supported by: 

   1. Science and Technology Fund of Hunan Provincial Department of Education (21A0524, 24A0602); 

   2. Key Laboratory of Tumor Precision Medicine, 

      - Hunan colleges and Universities Project (2019-379); 

   3. Hunan Natural Science Foundation (2023JJ30564). 

- **Authors’ contributions** : Tao Tan and Hui Xie designed the study, searched, analyzed and interpreted the literature, and are the major contributors in writing the manuscript. Qing Li, Haiqing Hu and Lijuan Ding collected the case data. Tao Tan and Yue Sun revised the manuscript. 

- **Acknowledgements** : N/A 

- **Availability of data and material** : The datasets used and/or analysed during the current study are available from the corresponding author on reasonable request. 

## **References** 

- [1] R. H. Mak, M. G. Endres, J. H. Paik, R. A. Sergeev, H. Aerts, C. L. Williams, K. R. Lakhani, E. C. Guinan, Use of crowd innovation to develop an artificial intelligence–based solution for radiation therapy targeting, JAMA Oncology 5 (5) (2019) 654–661. 

- [2] G. Minniti, C. Goldsmith, M. Brada, Radiotherapy, in: M. J. Aminoff, F. Boller, D. F. Swaab (Eds.), Handbook of Clinical Neurology, Vol. 104, Elsevier, 2012, pp. 215–228. 

- [3] C. Jiang, T. Ji, Q. Qiao, Application and progress of artificial intelligence in radiation therapy dose prediction, Clinical and Translational Radiation Oncology 47 (2024) 100792. 

- [4] X. Chen, K. Men, J. Zhu, B. Yang, M. Li, Z. Liu, X. Yan, J. Yi, J. Dai, Dvhnet: a deep learning-based prediction of patient-specific dose volume histograms for radiotherapy planning, Medical Physics 48 (6) (2021) 2705–2713. 

- [5] X. Chen, K. Men, Y. Li, J. Yi, J. Dai, A feasibility study on an automated method to generate patient-specific dose distributions for radiotherapy using deep learning, Medical Physics 46 (1) (2019) 56–64. `doi:10.1002/mp.13262` . 

- [6] F. Milletari, N. Navab, S.-A. Ahmadi, V-net: Fully convolutional neural networks for volumetric medical image segmentation, in: 2016 Fourth International Conference on 3D Vision (3DV), 2016, pp. 565–571. 

22 

- [7] Y. Shao, X. Zhang, G. Wu, Q. Gu, J. Wang, Y. Ying, A. Feng, G. Xie, Q. Kong, Z. Xu, Prediction of three-dimensional radiotherapy optimal dose distributions for lung cancer patients with asymmetric network, IEEE Journal of Biomedical and Health Informatics 25 (4) (2021) 1120–1127. 

- [8] Z. Li, K. Chen, Z. Yang, Q. Zhu, X. Yang, Z. Li, J. Fu, A personalized dvh prediction model for hdr brachytherapy in cervical cancer treatment, Frontiers in Oncology 12 (2022) 967436. 

- [9] S. H. Ahn, E. Kim, C. Kim, W. Cheon, M. Kim, S. B. Lee, Y. K. Lim, H. Kim, D. Shin, D. Y. Kim, J. H. Jeong, Deep learning method for prediction of patient-specific dose distribution in breast cancer, Radiation Oncology 16 (1) (2021) 154. `doi:10.1186/s13014-021-01864-9` . 

- [10] C. Kontaxis, G. H. Bol, J. J. W. Lagendijk, B. W. Raaymakers, Deepdose: Towards a fast dose calculation engine for radiation therapy using deep learning, Physics in Medicine & Biology 65 (7) (2020) 075013. 

- [11] Z. Liu, J. Fan, J. Miao, Y. Tian, W. Hu, K. Men, J. Dai, Dose prediction for individualized prescription using deep learning: Modeling multiple radiotherapy clinical scenarios, Biomedical Signal Processing and Control 110 (2025) 108228. 

- [12] R. Gao, B. Lou, Z. Xu, D. Comaniciu, A. Kamen, Flexible-cm gan: Towards precise 3d dose prediction in radiotherapy, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 715–725. 

- [13] F. Shen, X. Du, Y. Gao, J. Yu, Y. Cao, X. Lei, J. Tang, Imagharmony: Controllable image editing with consistent object quantity and layout, arXiv preprint arXiv:2506.01949. 

- [14] F. Shen, J. Yu, C. Wang, X. Jiang, X. Du, J. Tang, Imaggarment-1: Fine-grained garment generation for controllable fashion design, arXiv preprint arXiv:2504.13176. 

- [15] F. Shen, H. Ye, J. Zhang, C. Wang, X. Han, W. Yang, Advancing pose-guided image synthesis with progressive conditional diffusion models (2024). `arXiv:2310.06313` . URL `https://arxiv.org/abs/2310.06313` 

- [16] F. Shen, C. Wang, J. Gao, Q. Guo, J. Dang, J. Tang, T.-S. Chua, Long-term talkingface generation via motion-prior conditional diffusion model, arXiv preprint arXiv:2502.09533. 

- [17] F. Shen, H. Ye, S. Liu, J. Zhang, C. Wang, X. Han, W. Yang, Boosting consistency in story visualization with rich-contextual conditional diffusion models (2024). `arXiv:2407.02482` . URL `https://arxiv.org/abs/2407.02482` 

- [18] F. Shen, J. Tang, Imagpose: A unified conditional framework for pose-guided person generation, in: Advances in Neural Information Processing Systems, Vol. 37, 2024, pp. 6246–6266. 

- [19] F. Shen, X. Jiang, X. He, H. Ye, C. Wang, X. Du, Z. Li, J. Tang, Imagdressing-v1: Customizable virtual dressing (2024). `arXiv:2407.12705` . URL `https://arxiv.org/abs/2407.12705` 

- [20] Y. Zhang, C. Li, L. Zhong, Z. Chen, W. Yang, X. Wang, Dosediff: Distance-aware diffusion model for dose prediction in radiotherapy, IEEE Transactions on Medical Imaging 43 (10) (2024) 3621–3633. 

23 

- [21] L. Fu, X. Li, X. Cai, Y. Wang, X. Wang, Y. Shen, Y. Yao, Md-dose: A diffusion model based on the mamba for radiation dose prediction, arXiv preprint arXiv:2403.08479. 

- [22] H. Sun, J. Qin, Z. Liu, X. Jia, K. Yan, L. Wang, Z. Liu, S. Gong, Generation driven understanding of localized 3d scenes with 3d diffusion model, Scientific reports 15 (1) (2025) 14385. `doi:10.1038/s41598-025-98705-6` . URL `https://doi.org/10.1038/s41598-025-98705-6` 

- [23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, arXiv preprint arXiv:1706.03762. 

- [24] S. Foti, B. Koo, D. Stoyanov, M. J. Clarkson, 3d shape variational autoencoder latent disentanglement via mini-batch feature swapping for bodies and faces, arXiv preprint arXiv:2111.12448. 

- [25] J. Benton, Y. Shi, V. De Bortoli, G. Deligiannidis, A. Doucet, From denoising diffusions to denoising markov models, Journal of the Royal Statistical Society Series B: Statistical Methodology 86 (2) (2024) 286–301. 

- [26] B. He, T. Hofmann, Simplifying transformer blocks, arXiv preprint arXiv:2311.01906. 

- [27] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, A. Lerer, Automatic differentiation in pytorch, 2017. 

- [28] H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz, mixup: Beyond empirical risk minimization, arXiv preprint arXiv:1710.09412. 

- [29] H. Alkinani, M. El-Sakka, Patch-based models and algorithms for image denoising: a comparative review between patch-based images denoising methods for additive noise reduction, EURASIP Journal on Image and Video Processing 2017 (1) (2017) 58. 

- [30] B. Wang, L. Teng, L. Mei, Z. Cui, X. Xu, Q. Feng, D. Shen, Deep learning-based head and neck radiotherapy planning dose prediction via beam-wise dose decomposition, in: L. Wang, Q. Dou, P. T. Fletcher, S. Speidel, S. Li (Eds.), Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, Springer, 2022, pp. 575–584. 

- [31] R. Gao, B. Lou, Z. Xu, D. Comaniciu, A. Kamen, Flexible-cm gan: Towards precise 3d dose prediction in radiotherapy, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 715–725. 

- [32] A. Babier, B. Zhang, R. Mahmood, K. L. Moore, T. G. Purdie, A. L. McNiven, T. C. Y. Chan, Openkbp: The open-access knowledge-based planning grand challenge and dataset, Medical Physics 48 (9) (2021) 5549–5561. 

- [33] J. Bertels, D. Robben, R. Lemmens, D. Vandermeulen, Convolutional neural networks for medical image segmentation, arXiv preprint arXiv:2211.09562. 

- [34] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, in: Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, K. Q. Weinberger (Eds.), Advances in Neural Information Processing Systems, Vol. 27, Curran Associates, Inc., 2014. 

- [35] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam, Encoder-decoder with atrous separable convolution for semantic image segmentation, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 801–818. 

24 

- [36] V. Kearney, J. W. Chan, S. Haaf, M. Descovich, T. D. Solberg, Dosenet: a volumetric dose prediction algorithm using 3d fully-convolutional neural networks, Physics in Medicine & Biology 63 (23) (2018) 235022. 

- [37] P. Dursun, L. Hong, G. Jhanwar, Q. Huang, Y. Zhou, J. Yang, H. Pham, L. Cervino, J. M. Moran, J. O. Deasy, et al., Automated vmat treatment planning using sequential convex programming: algorithm development and clinical implementation, Physics in Medicine & Biology 68 (15) (2023) 155006. 

- [38] G. D. Hugo, M. Rosu, Advances in 4d radiation therapy for managing respiration: part i–4d imaging, Zeitschrift für medizinische Physik 22 (4) (2012) 258–271. 

- [39] J. Fan, J. Wang, Z. Chen, C. Hu, Z. Zhang, W. Hu, Automatic treatment planning based on three-dimensional dose distribution predicted from deep learning technique, Medical Physics 46 (1) (2019) 370–381. 

- [40] A. M. Barragán-Montero, D. Nguyen, W. Lu, M.-H. Lin, R. Norouzi-Kandalan, X. Geets, E. Sterpin, S. Jiang, Threedimensional dose prediction for lung imrt patients with deep neural networks: robust learning from heterogeneous beam configurations, Medical Physics 46 (8) (2019) 3679–3691. 

- [41] B. Zhan, J. Xiao, C. Cao, X. Peng, C. Zu, J. Zhou, Y. Wang, Multi-constraint generative adversarial network for dose prediction in radiotherapy, Medical Image Analysis 77 (2022) 102339. 

- [42] J. Ma, D. Nguyen, T. Bai, M. Folkerts, X. Jia, W. Lu, L. Zhou, S. Jiang, A feasibility study on deep learning-based individualized 3d dose distribution prediction, Medical Physics 48 (8) (2021) 4438–4447. 

- [43] J. Ho, A. Jain, P. Abbeel, Denoising diffusion probabilistic models, Advances in Neural Information Processing Systems 33 (2020) 6840–6851. 

- [44] Y. He, P. Guo, Y. Tang, A. Myronenko, V. Nath, Z. Xu, D. Yang, C. Zhao, B. Simon, M. Belue, S. Harmon, B. Turkbey, D. Xu, W. Li, Vista3d: A unified segmentation foundation model for 3d medical imaging, arXiv preprint arXiv:2406.05285. 

- [45] E. van der Bijl, Y. Wang, T. Janssen, S. Petit, Predicting patient specific pareto fronts from patient anatomy only, Radiotherapy and Oncology 150 (2020) 46–50. 

- [46] J. Dolz, I. Ben Ayed, C. Desrosiers, Dense multi-path u-net for ischemic stroke lesion segmentation in multiple image modalities, in: International MICCAI Brainlesion Workshop, Springer, 2018, pp. 271–280. 

- [47] S. H. Benedict, K. M. Yenice, D. Followill, J. M. Galvin, W. Hinson, B. Kavanagh, P. Keall, M. Lovelock, S. Meeks, L. Papiez, et al., Stereotactic body radiation therapy: the report of aapm task group 101, Medical Physics 37 (8) (2010) 4078–4101. 

- [48] D. Rodriguez, T. Nayak, Y. Chen, R. Krishnan, Y. Huang, On the role of deep learning model complexity in adversarial robustness for medical images, BMC Medical Informatics and Decision Making 22 (Suppl 2) (2022) 160. 

- [49] M. J. Sheller, B. Edwards, G. A. Reina, J. Martin, S. Pati, A. Kotrotsou, M. Milchenko, W. Xu, D. Marcus, R. R. Colen, et al., Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data, Scientific Reports 10 (1) (2020) 12598. 

- [50] B. Emami, J. Lyman, A. Brown, L. Cola, M. Goitein, J. E. Munzenrider, B. Shank, L. J. Solin, M. Wesson, Tolerance of normal tissue to therapeutic irradiation, International Journal of Radiation Oncology* Biology* Physics 21 (1) (1991) 109–122. 

25 

- [51] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, arXiv preprint arXiv:2010.11929. 

- [52] D. Nguyen, A. S. Barkousaraie, C. Shen, X. Jia, S. Jiang, Generating pareto optimal dose distributions for radiation therapy treatment planning, in: D. Shen, T. Liu, T. M. Peters, L. H. Staib, C. Essert, S. Zhou, P.-T. Yap, A. Khan (Eds.), Medical Image Computing and Computer Assisted Intervention – MICCAI 2019, Springer, 2019, pp. 59–67. 

- [53] X. Guo, Y. Yang, C. Ye, S. Lu, B. Peng, H. Huang, Y. Xiang, T. Ma, Accelerating diffusion models via pre-segmentation diffusion sampling for medical image segmentation, in: 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), IEEE, 2023, pp. 1–5. 

- [54] S. Mori, R. Hirai, Y. Sakata, Using a deep neural network for four-dimensional ct artifact reduction in image-guided radiotherapy, Physica Medica 65 (2019) 67–75. 

- [55] F. Isensee, J. Petersen, A. Klein, D. Zimmerer, P. F. Jaeger, S. Kohl, J. Wasserthal, G. Koehler, T. Norajitra, S. Wirkert, K. H. Maier-Hein, nnu-net: Self-adapting framework for u-net-based medical image segmentation, arXiv preprint arXiv:1809.10486. 

- [56] M. Ma, N. Kovalchuk, M. K. Buyyounouski, L. Xing, Y. Yang, Incorporating dosimetric features into the prediction of 3d vmat dose distributions using deep convolutional neural network, Physics in Medicine & Biology 64 (12) (2019) 125017. 

- [57] L. Ma, M. Chen, X. Gu, W. Lu, Deep learning-based inverse mapping for fluence map prediction, Physics in Medicine & Biology 65 (23) (2020) 235035. 

- [58] J. S. Buatti, Integrating standardized head and neck radiotherapy data with deep learning for accurate dose predictions and assisted treatment planning, The University of Texas Health Science Center at San Antonio (2024). 

- [59] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, et al., Attention u-net: Learning where to look for the pancreas, arXiv preprint arXiv:1804.03999. 

- [60] H. Paganetti, C. Beltran, S. Both, L. Dong, J. Flanz, K. Furutani, C. Grassberger, D. R. Grosshans, A.-C. Knopf, J. A. Langendijk, et al., Roadmap: proton therapy physics and biology, Physics in Medicine & Biology 66 (5) (2021) 05RM01. 

- [61] C. Fiorino, R. Jeraj, C. H. Clark, C. Garibaldi, D. Georg, L. Muren, W. van Elmpt, T. Bortfeld, N. Jornet, Grand challenges for medical physics in radiation oncology, Radiotherapy and Oncology 153 (2020) 7–14. 

- [62] B. Choi, D. K. Shrestha, A. Attia, B. J. Stish, J. Leenstra, J. C. Rwigema, J. Ma, S. U. Lee, J. H. Jeong, J. Kim, J. Kim, C. Beltran, J. C. Park, Deep learning-based dose prediction for prostate cancer with empty bladder protocol: a framework for efficient and personalized radiotherapy planning, Frontiers in Oncology 15 (2025) 1690416, abbreviated as: Front Oncol. 

- [63] T. Nemoto, N. Futakami, E. Kunieda, M. Yagi, A. Takeda, T. Akiba, E. Mutu, N. Shigematsu, Effects of sample size and data augmentation on u-net-based automatic segmentation of various organs, Radiological Physics and Technology 14 (3) (2021) 318–327, abbreviated as: Radiol Phys Technol. 

26 

- [64] H. Zhang, L. Ma, A. Lim, J. Ye, L. Lukas, H. Li, N. A. Mayr, E. L. Chang, Dosimetric validation for prospective clinical trial of grid collimator-based spatially fractionated radiation therapy: dose metrics consistency and heterogeneous pattern reproducibility, International Journal of Radiation Oncology, Biology, Physics 118 (2) (2024) 565–573, abbreviated as: Int J Radiat Oncol Biol Phys. 

- [65] J. P. Kirkpatrick, A. J. Van Der Kogel, T. E. Schultheiss, Radiation dose–volume effects in the spinal cord, International Journal of Radiation Oncology, Biology, Physics 76 (3) (2010) S42–S49, abbreviated as: Int J Radiat Oncol Biol Phys. 

- [66] Y. R. Lawrence, X. A. Li, I. El Naqa, C. A. Hahn, L. B. Marks, T. E. Merchant, A. P. Dicker, Radiation dose–volume effects in the brain, International Journal of Radiation Oncology, Biology, Physics 76 (3) (2010) S20–S27, abbreviated as: Int J Radiat Oncol Biol Phys. 

- [67] L. B. Marks, S. M. Bentzen, J. O. Deasy, J. D. Bradley, I. S. Vogelius, I. El Naqa, J. L. Hubbs, J. V. Lebesque, R. D. Timmerman, M. K. Martel, et al., Radiation dose–volume effects in the lung, International Journal of Radiation Oncology, Biology, Physics 76 (3) (2010) S70–S76, abbreviated as: Int J Radiat Oncol Biol Phys. 

27 

### **Supplementary Table 1:** Organ-at-Risk (OAR) & Target Structure Dose Constraints 

|**Structure Name**|**Category **|**Dose Constraint Description**|**Specific Value/Condition**|
|---|---|---|---|
|Total Lung-GTV|OAR|Lung volume dose limits|_V_20 _≤_30%,_V_5 _≤_60%|
|SpinalCord|OAR|Spinal cord max dose limit|_D_max _≤_45Gy|
|Esophagus|OAR|Esophagus dose limits|_D_max _≤_65Gy,_D_mean _≤_34Gy;_V_50 _≤_40%;|
||||_V_35 _≤_50%|
|Heart|OAR|Heart dose limits (by disease)|_V_30 _≤_10%(left breast irradiation);_V_40 _≤_5%(lung|
||||cancer);_D_mean _≤_26Gy|
|LAD|OAR|Left anterior descending artery limits|_V_30 _≤_10%;_D_max _≤_50Gy<br>|
|GreatVessels<br>|OAR|Major vessel dose limits<br>|_D_max _≤_60Gy ;_V_50 _≤_50%<br>|
|Trachea|OAR|Trachea dose limits|_D_max _≤_60Gy;_V_50 _≤_50%|
|Chiasm|OAR|Optic chiasm dose limits|<br>_D_max _≤_50Gy; PRV:_D_max _≤_54Gy (1% volume)|
|Brain<br>|OAR<br>|Brain tissue dose limits<br>|_V_20 _≤_30%,_V_50 _≤_50%<br>|
|OCavity-PTV|OAR|Oral cavity dose limits|_D_mean _≤_40Gy,_V_50 _≤_50%,_V_40 _≤_70%,|
||||_D_max _≤_60Gy|
|Cochlea_R/Cochlea_L<br>|OAR<br>|Cochlea dose limits<br>|<br>_D_max _≤_50Gy;_D_mean _≤_45Gy<br>|
|BrainStem|OAR|Brainstem max dose limit|_D_max _≤_54Gy<br>|
|BrainStem_03|OAR|Brainstem auxiliary dose limits|_D_max _≤_60Gy;_V_50 _≤_0%,_V_40 _≤_10%,|
||||_D_mean _≤_45Gy|
|Mandible-PTV|OAR|Mandible dose limits|_D_max _≤_60Gy,_V_50 _≤_30%,_D_mean _≤_45Gy|
|Submand-PTV/Submand|R-<br>OAR|Submandibular gland dose limits|_D_mean _≤_35Gy,_V_50 _≤_50%,_D_max _≤_60Gy|
|PTV or L||||
|ParotidIps-PTV|OAR|Parotid gland dose limits (ipsilateral)|_D_mean _≤_26Gy (at least one side),_V_30 _≤_50%<br>|
|Submandibular|OAR|Submandibular gland dose limits|_D_mean _≤_35Gy (at least one side);_V_50 _≤_50%|
|Pituitary|OAR|Pituitary gland dose limits|_D_max _≤_45Gy;_D_mean _≤_30Gy<br>|
|Mandible<br>|OAR<br>|Mandible overall dose limits<br>|_V_50 _≤_30%;_D_max _≤_60Gy<br>|
|Eyes|OAR|Eye dose limits|_D_max _≤_50Gy;_D_mean _≤_35Gy|
|Lens|OAR|Lens max dose limit|_D_max _≤_8Gy|
|OpticNerve_R or L|OAR|Optic nerve dose limits|_D_max _≤_50Gy; PRV:_D_max _≤_54Gy (1% volume)|
|ParotidCon-PTV|OAR|Parotid gland dose limits (contralateral)|_D_mean _≤_30Gy (at least one side),_V_30 _≤_40%,|
||||_V_50 _≤_2%,_D_max _≤_60Gy|
|Thyroid|OAR|Thyroid dose limits|_D_mean _≤_18Gy;_V_50 _≤_50%|
|OralCavity<br>|OAR|Oral cavity dose limits<br>|_D_mean _≤_40Gy;_V_50 _≤_50%<br>|
|Thyroid-PTV|OAR|Thyroid target dose limits|_D_mean _≤_30Gy,_V_30 _≤_50%,_V_40 _≤_30%,|
||||_D_max _≤_50Gy|
|PTV|Target|Prescription dose coverage|_≥_95%volume covered by prescription dose|
|PTVHighOPT|Target|High-dose PTV coverage|_D_95 _≥_prescription dose|
|PTVMidOPT|Target|Middle-dose PTV coverage|_D_95 _≥_prescription dose|
|PTVLowOPT|Target|Low-dose PTV coverage|_D_95 _≥_prescription dose<br>|
|PTV_Ring.3-2|Auxiliary|Dose diffusion limits (inner/outer ring)|Inner 3mm: _D_max _≤_50%of prescription dose; Outer|
||||2mm: _D_max _≤_30%of prescription dose|
|Body_Ring0-3|Auxiliary|Surface dose diffusion limits|0-3mm ring: _D_max _≤_70%of prescription dose|
|RingPTVHigh|Auxiliary|High-dose region diffusion limits|5-10mm ring from PTV:_D_max _≤_80%of prescription|
||||dose|
|RingPTVMid|Auxiliary|Mid-dose region diffusion limits|5-10mm ring from PTV:_D_max _≤_60%of prescription|
||||dose|
|RingPTVLow|Auxiliary|Low-dose region diffusion limits|10-20mm ring from PTV:_D_max _≤_30%of|
||||prescription dose|
|Posterior_Neck|Auxiliary|Posterior neck dose limits|<br>_D_max _≤_70%of prescription dose|
|PharConst-PTV|Auxiliary|Pharyngeal constraint dose limits|_D_max _≤_80%of prescription dose;_V_70 _≤_20%<br>|
|PharynxConst|Auxiliary|Pharyngeal auxiliary dose limits|_D_mean _≤_50Gy;_V_50 _≤_50%|
|SpinalCord_05|Auxiliary|<br> Spinal cord PRV dose limits|<br>_D_max _≤_50Gy (1% volume)|



28 

**Supplementary Table 2:** Analysis of prediction variability across 30 external test patients 

|**Metric**|**Mean ± SD**|**Observed CV (%)**|**Clinical Stability Threshold (%)**|**Assessment**|
|---|---|---|---|---|
|MAE (Gy)|0.83±0.019|2.3±0.5|3.5±0.8[62]|Excellent|
|DICE Similarity|0.94±0.017|1.8±0.4|2.2±0.6[63]|Excellent|
|PTV_D_95 (Gy)|67.5±0.81|1.2±0.3|2.0±0.5[64]|Excellent|
|PTV_D_50 (Gy)|72.1±0.86|1.2±0.3|1.8±0.4[64]|Excellent|
|Spinal Cord_D_max (Gy)|23.1±0.48|2.1±0.6|3.2±0.9[65]|Excellent|
|Brainstem_D_max (Gy)|18.7±0.56|3.0±0.7|3.5±1.0[66]|Good|
|Lung_V_20 (%)|25.3±0.40|1.6±0.4|2.8±0.7[67]|Excellent|



_Note: Data derived from 10 independent predictions for each of 30 randomly selected patients from the external test set. “CV (Patient-wise mean ± SD)” represents the mean and standard deviation of the coefficient of variation calculated for each patient individually, consistent with methods used in prior dosimetric studies[67]. The CV for all metrics is consistently low, demonstrating excellent model robustness across a patient population, a key consideration for clinical generalizability_ 

**Table 3:** Spatial distribution of prediction uncertainty by anatomical region 

|**Anatomical Region**|**Va**|**riance (Gy²)**||**Clinical**|
|---|---|---|---|---|
||**Mean**|**95th Percentile**|**Max**|**Relevance**|
|**Planning Target Volume (PT**|**V)**||||
|PTV Core (Dose > 95%|0.0012±0.0003|0.0018|0.0021|Negligible|
|Rx)|||||
|PTV Middle (50–95% Rx)|0.0038±0.0011|0.0065|0.0089|Low|
|PTV Periphery (<50% Rx)|0.0085±0.0021|0.0123|0.0154|Low|
|**Organs at Risk (OARs)**|||||
|Spinal Cord|0.0008±0.0002|0.0015|0.0018|Negligible|
|Brainstem|0.0011±0.0003|0.0019|0.0023|Negligible|
|Lungs|0.0023±0.0006|0.0035|0.0048|Negligible|
|Parotid Glands|0.0018±0.0005|0.0029|0.0039|Negligible|
|Esophagus|0.0015±0.0004|0.0024|0.0032|Negligible|
|**Dose-Based Regions**|||||
|High-Dose Region (>80|0.0015±0.0004|0.0028|0.0036|Negligible|
|Gy)|||||
|Medium-Dose<br>Region|0.0042±0.0012|0.0087|0.0125|Low|
|(20–80 Gy)|||||
|Low-Dose Region (<20|0.0156±0.0042|0.0210|0.0385|Acceptable|
|Gy)|||||



_Note: Variance computed from 10 independent predictions. Classification: Negligible (<0.002 Gy²), Low (0.002–0.01 Gy²), Acceptable (<0.02 Gy²). Rx_ 

_= prescription dose (70 Gy). The highest uncertainty is confined to PTV periphery and low-dose regions, while clinically critical areas exhibit negligible uncertainty._ 

29 

**Algorithm 1** Training Procedure for Conditional Diffusion Model with Anatomical-Dose Dual Constraints 

|**Step**|**Operation**|**Details**|
|---|---|---|
|**Input Di**|**efinition**||
|1|Data Pairs|_D_ =_{_CT_i,_Dose_i}_<sup>_N_</sup><br>_i_=1<sup>, including 3D CT images (128</sup><sup>_×_128</sup><sup>_×_64) and</sup><br>clinical dose distributions (Gy). Hyperparameters: Diffusion steps<br>_T_ =1000, clinical constraint thresholds (_D_max,_Vx_), block processing<br>(size 32_×_32_×_24, overlap 8_×_8_×_8)|
|**Initializ**|**ation**||
|2|Models|Noise predictor_fθ_: 3D U-Net with attention. Structure encoder_gϕ_:<br>Pre-trained 3D-VAE (mixup pre-trained). Hyperparameters: _λ_1 =1_._0,<br>_λ_2 =0_._5,_λ_3 =0_._001, mixup coefficient_α_=0_._4|
|**Trainin**|**g Loop**||
|3|Iteration|Repeat steps 4-14 until convergence|
|**Data Pr**|**ocessing**||
|4|Sampling|Randomly extract (CT, Dose) from_D_, fetch corresponding PTV/OAR<br>masks|
|5|Mixup|mixCT =_α ·_CT 1_−α ·_CTrand,mixDose =_α ·_Dose 1_−α ·_Doserand<br>(_α ∼_Beta0_._4_,_0_._4)|
|6|Blocking|Split CT/mixCT into 32_×_32_×_24 blocks with 8_×_8_×_8 overlap,<br>zero-padded. Dimension: [B, C, D, H, W]_→_[B_·Nb_, C, 32, 32, 24]|
|**Diffusio**|**fn Process**||
|7|Noise Sampling|**_ϵ_**_t ∼N_0_,_**I**,_t ∼_Uniform_{_1_, . . . , T}_mapped to 256D embedding|
|||**t**emb|
|8|Noisy Latent|**z**0 =_gϕ_CTblocks,**z**_t_ = <sup>_~~√~~_</sup><br>_αt_**z**0<br>_~~√~~_<br>1_−αt_**_ϵ_**_t_ (_αt_ =1_−βt_,_βt_:<br>1e-4_→_0.02)|
|9|Features|Structural features**C**struct from PTV/OAR masks, fused with**t**emb via<br>Transformer|
|10|Prediction|_fθ_**z**_t,_**t**emb_,_**C**outputs˜**_ϵ_**_θ_|
|**Optimi**|**zation**||
|11|Loss|Pre-train: _L_=_λ_1_L_mse _λ_2_L_KL|
|||E2E:_L_=_λ_1_L_mse _λ_2_L_cond (50+ clinical constraints)|
|12|Merging|Weighted merge blocks with linear decay in overlaps: [B_·Nb_, C,<br>32,32,24]_→_[B, C, D, H, W]|
|13|Update|AdamW (_η_0 =5_×_10<sup>_−_4</sup>, cosine decay to10<sup>_−_6</sup>)|
|14|Stop|Terminate if validation dose compliance plateaus for 20 epochs|



30 

