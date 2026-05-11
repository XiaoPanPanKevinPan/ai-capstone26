# Homework 4 Report: Robot Manipulation Framework

**Student ID:** [Your Student ID]  
**Name:** [Your Name]

---

## 1. About Task 1: Forward Kinematics (FK)

### 1.1 Implementation Explanation
The `your_fk()` function was implemented using the **Classic D-H Convention**. The process involves:
1. **Kinematic Chain Construction**: Starting from the base transformation, we iterate through all 6 joints. For each joint $i$, we compute the individual transformation matrix $A_i$ using the D-H parameters ($a_i, d_i, \alpha_i$) and the joint angle $q_i$.
   - $A_i = Rot_z(q_i) \cdot Trans_z(d_i) \cdot Trans_x(a_i) \cdot Rot_x(\alpha_i)$
2. **Intermediate Frame Tracking**: During the iteration, we store the z-axis ($z_{i-1}$) and the origin ($p_{i-1}$) of each joint frame in the base coordinate system. These are extracted from the 3rd and 4th columns of the accumulated transformation matrix.
3. **End-Effector Pose**: The final transformation matrix $T$ represents the end-effector pose. We apply a required "adjustment" matrix to the rotation part and convert the result into a 7D pose (3D position + 4D quaternion).
4. **Jacobian Computation**: The geometric Jacobian $J$ is a $6 \times 6$ matrix where each column $i$ corresponds to joint $i$:
   - Linear part: $J_{v,i} = z_{i-1} \times (p_{end} - p_{i-1})$
   - Angular part: $J_{\omega,i} = z_{i-1}$

### 1.2 Difference between D-H Convention and Craig's Convention
- **Classic D-H Convention**:
    - The joint $i$ axis is aligned with $z_{i-1}$.
    - The parameters $a_i$ and $\alpha_i$ describe the relationship between $z_{i-1}$ and $z_i$ relative to frame $i$.
    - Transformation order: $Rot_z(\theta_i) \rightarrow Trans_z(d_i) \rightarrow Trans_x(a_i) \rightarrow Rot_x(\alpha_i)$.
- **Modified D-H (Craig's) Convention**:
    - The joint $i$ axis is aligned with $z_i$.
    - The parameters $a_{i-1}$ and $\alpha_{i-1}$ describe the relationship between $z_{i-1}$ and $z_i$ relative to frame $i-1$.
    - Transformation order: $Rot_x(\alpha_{i-1}) \rightarrow Trans_x(a_{i-1}) \rightarrow Rot_z(\theta_i) \rightarrow Trans_z(d_i)$.

### 1.3 D-H Table (Classic Convention)
Based on the parameters provided in `fk.py`, the D-H table is as follows:

| Joint | $a_i$ (m) | $\alpha_i$ (rad) | $d_i$ (m) | $\theta_i$ (rad) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0 | $\pi/2$ | 0.0892 | $q_1$ |
| 2 | -0.425 | 0 | 0 | $q_2$ |
| 3 | -0.392 | 0 | 0 | $q_3$ |
| 4 | 0 | $\pi/2$ | 0.1093 | $q_4$ |
| 5 | 0 | $-\pi/2$ | 0.09475 | $q_5$ |
| 6 | 0 | 0 | 0.2023 | $q_6$ |

---

## 2. About Task 2: Inverse Kinematics (IK)

### 2.1 Implementation Explanation
The `your_ik()` function implements an **Iterative Inverse Kinematics** solver using the **Jacobian Pseudo-inverse Method**.
1. **Iterative Loop**: The solver runs for a maximum of `max_iters` or until the error norm falls below `stop_thresh`.
2. **Error Calculation**:
   - **Position Error**: $\Delta p = p_{target} - p_{current}$.
   - **Orientation Error**: Computed by finding the relative rotation $R_{rel} = R_{target} R_{current}^T$ and converting it to a rotation vector (axis-angle representation) $\Delta \Phi$.
   - These are combined into a 6D error vector $\Delta x = [\Delta p, \Delta \Phi]^T$.
3. **Joint Update**: The joint increment is calculated as $\Delta q = \alpha \cdot J^{\dagger} \Delta x$, where $J^{\dagger}$ is the Moore-Penrose pseudo-inverse of the Jacobian and $\alpha$ is a step rate (set to 0.5 for stability).
4. **Constraints**: After each update, the joint angles are clipped to the provided `joint_limits`.

### 2.2 Problems Encountered and Solutions
- **Singularities**: Near singular configurations, the Jacobian becomes ill-conditioned. Using `scipy.linalg.pinv` (SVD-based pseudo-inverse) ensures numerical stability by ignoring very small singular values.
- **Orientation Continuity**: Representing orientation error directly with quaternions or Euler angles can lead to discontinuities. Using the rotation vector from the relative rotation matrix $R_{target} R_{curr}^T$ provides a robust and mathematically consistent error term for the geometric Jacobian.
- **Oscillations**: Large updates can cause the solver to overshoot the target. A step rate of 0.5 was introduced to dampen the updates and ensure smooth convergence.

### 2.3 Bonus: Other IK Methods
While the primary implementation uses the Pseudo-inverse method, another common method is the **Jacobian Transpose** method ($\Delta q = \alpha J^T \Delta x$). Although slower to converge, it is computationally cheaper and avoids the matrix inversion entirely, making it very stable even near singularities. However, the Pseudo-inverse method was chosen for its faster convergence and better handling of the specific UR5 kinematics in this assignment.
