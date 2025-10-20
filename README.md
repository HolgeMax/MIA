## Week 2: Smoothing and Interpolation with B-Splines

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


## Week 3: Image Registration with Linear Transformations

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
