## Week 2: introduction 
This week we were introduced to the course and the comming semester 


## Week 3: Smoothing and Interpolation with B-Splines

This week focused on **smoothing** and **interpolation** in 1D and 2D — key preprocessing steps in medical image analysis.

### Key Concepts

1. **Purpose**
   - *Smoothing:* reduces noise in medical images.  
   - *Interpolation:* estimates intensity values at non-grid points, crucial for image registration.

2. **Model**
   - Both methods use **linear regression** with non-linear **basis functions** φ_m(x) and learnable weights w.

3. **B-Spline Basis**
   - Order 0 → nearest-neighbor interpolation  
   - Order 1 → linear interpolation  
   - Order 3 → cubic (smooth and preferred)

4. **Controlling Smoothness**
   - Fewer basis functions (low M) → stronger smoothing, less detail.  
   - More basis functions (high M) → weaker smoothing, follows noise.

5. **Efficiency in Higher Dimensions**
   - 2D basis functions formed by Kronecker product: Φ = Φ₂ ⊗ Φ₁  
   - Exploits **separability**, allowing row- and column-wise filtering instead of large matrix inversion — reducing memory and computation.


## Week 4: Image Registration with Linear Transformations

This week focused on **Image Registration** — aligning medical images acquired at different times or orientations — using **linear spatial transformations** and the **landmark-based method**.

---

### 1. Coordinate Systems

- **Voxel vs. World Coordinates:**  
  *Voxels (v)* are discrete grid indices, while *world coordinates (x)* represent real-world positions (e.g., in mm).

- **Voxel-to-World Mapping:**  
  x = A·v + t  
  where **A** encodes scaling and orientation, and **t** is translation.

- **Homogeneous Coordinates:**  
  Combining into one affine 4×4 matrix **M** simplifies applying multiple transformations.

- **Resampling:**  
  Before registration, the moving image must be resampled to the fixed image grid:  
  (v_T2, 1)ᵀ = M_T2⁻¹ · M_T1 · (v_T1, 1)ᵀ  

- **Interpolation:**  
  Non-integer voxel locations are estimated using **cubic B-spline interpolation (order 3)** for smooth intensity values.

---

### 2. Landmark-Based Registration

- **Concept:**  
  Uses **corresponding anatomical landmarks** (xₙ in fixed image, yₙ in moving image).

- **Goal:**  
  Find transformation parameters **w** that minimize the squared distance between point pairs.

- **Accuracy:**  
  Depends heavily on precise, anatomically consistent landmark selection.

---

### 3. Linear Transformation Models

| Model | Description | Solution | Best Use |
|-------|--------------|-----------|-----------|
| **Affine** | Includes translation, rotation, scaling, and skew (12 DOF in 3D). | Closed-form linear regression. | Flexible but sensitive to landmark errors — may distort geometry. |
| **Rigid** | Only translation + rotation (y = R·x + t). Constraints: RᵀR = I, det(R)=1. | Solved via **SVD** from centered landmarks. | Ideal for rigid anatomy (e.g., skull); preserves shape and size. |

---

**Summary:**  
Linear registration aligns medical images by mapping voxel to world coordinates, resampling via B-splines, and optimizing transformations — rigid for shape-preserving cases, affine for more flexible alignment.


## Week 5: Intensity-Based Image Registration

This week focused on **Intensity-Based Registration**, an automatic approach for aligning medical images — especially useful across different imaging modalities.

---

### 1. Core Principles and Motivation
- **Automatic Registration:** Aligns images by minimizing an **energy function** derived directly from image intensities.  
- **Advantage:** No manual landmark selection — faster and more accurate.  
- **Goal:** Find transformation parameters **w** that minimize the energy E(w).

---

### 2. Choice of Similarity Measure
Depends on the intensity relationship between images:

| Registration Type | Use Case | Similarity Measure |
|--------------------|----------|--------------------|
| **Intra-modal** (same modality) | e.g., two CT scans | **Sum of Squared Differences (SSD)** |
| **Inter-modal** (different modalities) | e.g., T1 ↔ T2 MRI, or MRI ↔ CT | **Mutual Information (MI)** |

---

### 3. Mutual Information (MI)
- **Definition:** Measures statistical dependency between intensities in fixed (F) and moving (M) images.  
- **Optimization:**  
  Maximize MI = H_F + H_M − H_FM  
  or equivalently minimize  
  E(w) = H_FM − H_F − H_M  
- **Joint Histogram:** Built from intensity pairs (f, m) at corresponding locations.  
- **Joint Entropy (H_FM):** Indicates alignment quality — lower entropy → better alignment.

---

### 4. Implementation and Preprocessing
- **Resampling:** Moving image (e.g., T2) is resampled to the fixed image grid (e.g., T1) using affine scanner matrices.  
- **Interpolation:** **Cubic B-spline (order 3)** interpolation estimates intensities at transformed non-integer coordinates.  
- **Optimization:** No closed-form solution — parameters are refined iteratively (e.g., via grid search or gradient-based methods).

---

**Summary:**  
Intensity-based registration, particularly **Mutual Information (MI)**, enables robust, automatic alignment of multimodal images, outperforming rigid or affine landmark-based approaches.

## Week 6: Non-linear Image Registration

This week focused on **Non-linear Deformations**, extending registration beyond rigid and affine transformations to model **tissue deformation** in medical images.

---

### 1. Purpose of Non-linear Transformations
Linear models cannot capture local deformations caused by breathing, organ motion, or tumor growth.  
**Non-linear registration** models these complex, flexible deformations.

---

### 2. Model Structure (Residual Deformation)
After global alignment (affine registration), only the **residual deformation** is modeled:

y_d(x, w_d) = x_d + δ_d(x, w_d)

where **δ_d** describes local displacements between corresponding anatomical points.

---

### 3. Deformation Basis Functions
The deformation field δ_d(x, w_d) is expressed as a linear combination of **M basis functions** ϕ_m(x) with weights w_d,m:

δ_d(x, w_d) = Σₘ w_d,m · ϕ_m(x)

- **Basis choice:** typically **separable B-spline functions** (as in Week 2).  
- **3D case:** deformation parameters are separate for each axis →  
  w = (w₁, w₂, w₃)ᵀ.

---

### 4. Optimization Challenge
No closed-form solution exists for minimizing the energy function (e.g., SSD).  
Instead, **iterative numerical optimization** is required to estimate the parameters w.

---

### 5. Gauss–Newton Optimization
- **Idea:** Linearize the moving image M(y(x, w)) around current estimates of w.  
- **Solution:** Solve for update ε in closed form:  
  ε = (ΨᵀΨ)⁻¹Ψᵀτ  
- **Iterative update:**  
  w ← w + ε  

This method efficiently approximates parameter updates for non-linear problems.

---

### 6. Stabilization with Levenberg–Marquardt
If the update ε is too large, the energy may increase.  
To stabilize, modify the update:

ε = (ΨᵀΨ + λI)⁻¹Ψᵀτ

- **λ (lambda):** adjustable damping factor.  
  - If energy increases → increase λ (acts like gradient descent).  
  - If energy decreases → reduce λ for faster convergence.  
This ensures stable, monotonic energy reduction during optimization.

---

**Summary:**  
Non-linear registration enables realistic modeling of tissue deformation by combining **B-spline basis functions** with **iterative optimization (Gauss–Newton + Levenberg–Marquardt)**, providing flexible yet stable image alignment.

---
## Week 7

1. Motivation: Limitations of Volume-Based Methods
Volume-based approaches struggle with:

Partial Volume Effects and noise, which complicate precise local thickness measurements.

Registration issues, as volume-based alignment can mix neighboring tissue types, hindering accurate comparison of local anatomical features like cortical thickness across subjects.

2. Aim of Surface-Based Approaches
Surface methods introduce anatomical constraints to ensure accuracy:

Thickness Estimation: Surfaces assume smooth boundaries, which helps correct for partial volume effects and ensures topological correctness (inner and outer boundaries always appear together).

Registration: Surfaces enforce the constraint that gray matter must be aligned with gray matter for valid comparisons.

3. Surface Reconstruction and Thickness Measurement
The process involves creating and analyzing surfaces (e.g., boundaries of the cerebral cortex).

Reconstruction: Methods like Marching Cubes start from a binary mask.

Topology and Curvature: Reconstructed surfaces (e.g., from FreeSurfer) are processed to fix topology and smooth the result. A closed surface is vital for defining inside/outside and calculating surface curvature (e.g., concave vs. convex).

Cortical Thickness: Measured by "growing" a surface from the inner (white matter) boundary to the outer (CSF) boundary. Thickness is approximated by the Euclidean distance between corresponding vertex pairs.

4. Surface-Based Registration and Alignment
This enables the spatial comparison of complex cortical folding patterns:

Spherical Registration: The cortical surface is first expanded to a sphere (only works for surfaces with an Euler characteristic of 2).

Curvature Alignment: The sphere is then distorted so that the pattern of positive/negative curvatures aligns with a template (e.g., fsaverage). This places individual measurements onto a common template for averaging.

5. Other Metrics and Applications
Surfaces support specialized measurements and applications:

Geodesic Distance: Measures the distance along the surface, which is more accurate than Euclidean distance for two cortical regions and correlates well with cortical organization.

M/EEG Localization: Surfaces are used to correctly place current dipoles for source localization.

Augmented Reality: Surface models are crucial for visualization in image-guided surgical procedures.

6. Surface Processing Operations
Standard geometrical operations are applied to refine surfaces:

Smoothing: Techniques like Taubin smoothing use iterated Gaussian smoothing steps with opposite signs.

Surface Normals: Calculated by the Cross-product of adjacent edge vectors, determining the surface's direction.

Summary: Surface-based metrics provide a robust framework for analyzing cortical features by employing anatomical constraints during reconstruction and registration, enabling the accurate measurement of metrics like cortical thickness and geodesic distance on a common spherical template.
