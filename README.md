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

