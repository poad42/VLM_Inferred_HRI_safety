

# **Advanced Simulation Fidelity and Control Architectures in Contact-Rich Manipulation: A Comprehensive Analysis of Robotic Sawing Dynamics**

## **1\. Introduction: The Convergence of Physics, Geometry, and Control**

The simulation of contact-rich manipulation tasks represents the apex challenge in modern robotics. Unlike free-space motion planning, where the primary constraints are kinematic limits and obstacle avoidance, contact-rich tasks—such as the robotic sawing of a log—require the simultaneous resolution of stiff physical constraints, discontinuous contact dynamics, and high-bandwidth force modulation. The case study presented, involving a Franka Emika Panda robot equipped with a circular saw, serves as a quintessential archetype for these challenges. It exposes the fragile interplay between the discrete mathematics of physics engines, the geometric abstractions of scene description languages, and the continuous domain of modern control theory.  
This report provides an exhaustive technical analysis of the implementation status, validating the successes achieved in physics stability \[Phase 1\] and impedance tuning \[Phase 2\], while systematically deconstructing the persistent geometric anomalies manifesting as "Saw Tilt." The investigation confirms that the system has transitioned from a chaotic physical state to a stable control regime. However, the identified "Zero Error Fallacy" indicates a fundamental misalignment between the Control Frame (the mathematical target of the Operational Space Controller) and the Functional Frame (the physical cutting plane of the saw).  
The analysis leverages a deep review of the NVIDIA Isaac Lab architecture, the Universal Scene Description (USD) physics schema, and the mathematical underpinnings of Operational Space Control (OSC). It serves not only to resolve the immediate issue of tool orientation but also to document the precise calibration required to bridge the gap between idealized simulation and functional robotic behavior.

### **1.1. System Architecture and Kinematic Topology**

The simulation environment is built upon NVIDIA Isaac Lab 2.3.0, utilizing Isaac Sim 5.1.0 as the backend. This layered architecture is critical to understanding the source of errors. Isaac Sim provides the rendering and physics implementation (PhysX), while Isaac Lab provides the reinforcement learning-ready abstractions and controller logic.  
The kinematic chain in question is a serial manipulator (Franka Panda) terminating in a panda\_hand. Attached to this hand is a rigid body representing the saw. This attachment is mediated by a FixedJoint, a physics constraint that locks the six degrees of freedom (6-DOF) of the saw relative to the hand. The core conflict arises because the control software (OSC) operates on the kinematic definitions of the robot (URDF/USD), while the physical behavior is governed by the constraint solver's resolution of the attachment.  
The breakdown of the problem space is as follows:

* **Physics Layer:** Responsible for contact stability, friction, and constraint satisfaction.  
* **Control Layer:** Responsible for generating joint torques to achieve task-space targets.  
* **Geometric Layer:** Responsible for defining the relative transforms between bodies.

The "Saw Tilt" is a failure of the Geometric Layer to communicate effectively with the Control Layer, despite the Physics Layer functioning correctly.

## **2\. Phase 1 Analysis: The Physics of Stability and Contact Dynamics**

The stabilization of the saw-log interaction marks the first critical milestone. In rigid body simulation, "stability" is a measure of the solver's ability to maintain constraints without injecting spurious energy into the system. The adjustments made to restitution, depenetration velocity, and solver iterations constitute a textbook optimization of the simulation for rigid-contact tasks.

### **2.1. Restitution Coefficients and Energy Dissipation**

The reduction of the restitution coefficient to 0.0 for both the saw and the log is the primary factor in eliminating the "bouncing" behavior. In the context of the PhysX engine, the coefficient of restitution ($e$) governs the conservation of kinetic energy during a collision impulse along the contact normal.  
The collision law is governed by Newton’s hypothesis for impact:

$$v\_{rel}' \= \-e \\cdot v\_{rel}$$  
Where $v\_{rel}$ is the relative velocity before impact and $v\_{rel}'$ is the relative velocity after impact. A value of $e=0.0$ signifies a perfectly inelastic collision. In a real-world cutting scenario, the interaction between a saw blade and wood is highly dissipative due to material deformation, friction, and ablation. It is certainly not elastic.  
The initial "bouncing" observed in the simulation was likely an artifact of discrete time-stepping. When a rigid body with $e \> 0$ impacts a static surface, the solver calculates an impulse to reverse its velocity. If the gravity or control force immediately accelerates the object back into the surface within the next time step, the object enters a cycle of rapid impacts, visualized as jitter. By setting $e=0.0$, the simulation forces the relative velocity to zero immediately upon contact, effectively engaging the "stiction" necessary for a cutting operation.

### **2.2. Depenetration Velocity and "Explosive" Contact Resolution**

The limitation of max\_depenetration\_velocity to 0.1 m/s is a crucial safeguard against numerical explosions. In rigid body simulation, objects are technically allowed to overlap slightly between time steps (interpenetration). The solver detects this overlap ($d$) and applies a corrective impulse to separate the bodies in the next step.  
If the penetration depth is significant and the time step ($dt$) is small, the required velocity to separate the bodies in a single step ($v\_{corr} \= d/dt$) can be astronomically high. This results in massive kinetic energy injection into the system. For a high-impedance robot arm pushing a saw into a log, a slight overshoot in position can lead to deep penetration. Without a velocity cap, the solver would eject the saw from the log with violent force.  
Clamping this velocity ensures that the solver resolves the interpenetration over multiple frames rather than instantaneously. This effectively acts as a low-pass filter on the contact forces, smoothing out the interaction and maintaining the stability of the saw against the log surface. This is particularly vital in "compliant mode" control, where the robot is actively pushing against the constraint.

### **2.3. Solver Iterations and Constraint Fidelity**

Increasing the solver iterations (Position Iterations: 12, Velocity Iterations: 4\) directly addresses the "jitter" phenomenon by improving the convergence of the Linear Complementarity Problem (LCP) solver.  
The PhysX engine uses a projected Gauss-Seidel (PGS) or Temporal Gauss-Seidel (TGS) solver to satisfy constraints. These are iterative solvers. In a complex chain involving a robot arm (7-DOF), a fixed attachment (constraint), and a contact pair (saw on log), the solver must balance forces across the entire system.

* **Position Iterations:** These correct the geometric errors (penetration and joint separation). Insufficient iterations result in "drift" or visual separation of jointed parts.  
* **Velocity Iterations:** These correct the velocity errors (restitution and friction). Insufficient iterations result in "spongy" contacts or inaccurate friction behavior.

The interaction of a high-stiffness controller ($K\_p=1000$ for rotation) creates stiff spring-like forces. The physics engine must resolve these active control forces against the hard contact constraints. By increasing the iteration count, the solver propagates forces back and forth through the kinematic chain more times per step, arriving at a solution that minimizes residual error. The result is a system where the saw feels "solidly" connected to the arm and "solidly" planted on the log, rather than vibrating between the two.

### **2.4. Advanced Contact Parameters: Torsional Friction**

While the current tuning has achieved stability, further enhancements can be found in the advanced PhysX schemas utilized by Isaac Lab. Snippet 1 and 1 reveal the availability of torsional\_patch\_radius and min\_torsional\_patch\_radius in the RigidBody and FixedJoint configuration.  
Standard friction models in physics engines are often limited to tangential friction (resistance to sliding). However, a saw blade resting on a log also experiences *torsional friction* (resistance to twisting in place). In simulations where the contact patch is approximated by a single point, an object can spin unnaturally easily around the contact normal.

| Parameter | Function | Recommendation for Sawing |
| :---- | :---- | :---- |
| **torsional\_patch\_radius** | Defines the radius of the contact patch for applying torsional friction. | **Set \> 0 (e.g., 0.05m)**. This simulates the width of the saw blade or the saw's base, preventing it from pivoting unrealistically on the log surface. |
| **min\_torsional\_patch\_radius** | Ensures a minimum effective radius even for small contacts. | **Set equal to blade thickness/2**. This ensures rotational stability during the initial cut. |

Integrating these parameters into the RigidObjectCfg or FixedJoint definition would add another layer of physical realism, further suppressing any residual rotational jitter that might occur as the saw moves along the log.

## **3\. Phase 2 Analysis: Operational Space Control (OSC) Dynamics**

The configuration of the Operational Space Controller represents a sophisticated application of impedance control theory. The move from joint-space control to operational space (task space) is essential for manipulation tasks where the interaction occurs in Cartesian coordinates. The implementation described in the query aligns with the best practices for hybrid force/motion control.

### **3.1. Mathematical Formulation of OSC**

The Operational Space formulation, pioneered by Oussama Khatib, projects the robot's dynamics into the task space. The control law typically takes the form:

$$\\Gamma \= J^T(q) F\_{task} \+ N(q)^T \\Gamma\_{null} \+ g(q)$$  
Where:

* $\\Gamma$ is the vector of joint torques.  
* $J(q)$ is the task Jacobian.  
* $F\_{task}$ is the desired force/wrench in task space.  
* $N(q)$ is the null-space projection matrix.  
* $g(q)$ is the gravity compensation term.

In the user's implementation, $F\_{task}$ is generated via an impedance law:

$$F\_{task} \= \\Lambda(q) \\ddot{x}\_{des} \+ K\_p (x\_{des} \- x) \+ K\_d (\\dot{x}\_{des} \- \\dot{x})$$  
Here, $K\_p$ and $K\_d$ represent the stiffness and damping matrices, respectively. The user's configuration of anisotropic stiffness ($K\_p$) is the defining feature of the control strategy.

### **3.2. Anisotropic Stiffness and Task Decomposition**

The reported stiffness configuration demonstrates a nuanced understanding of the task requirements:

* **Translational Stiffness:** \`\`  
* **Rotational Stiffness:** \`\`

This anisotropy effectively decouples the control axes:

1. **Stiff Plane (X/Y):** $K\_p=100$ keeps the saw on the cutting line.  
2. **Compliant Normal (Z):** $K\_p=10$ creates a "soft" vertical behavior. This allows the saw to maintain contact with the log without exerting excessive force that would destabilize the simulation. It effectively turns the robot into a specialized spring that is stiff in positioning but soft in pressing.  
3. **Rigid Orientation (Rot):** $K\_p=1000$ attempts to lock the orientation of the tool. This is critical for a saw, which must remain perpendicular to the cutting surface to function. If rotational stiffness were low, the reaction torques from the log (friction) would twist the saw.

### **3.3. Damping and Transient Response**

The damping settings \[2, 2, 2, 4, 4, 4\] indicate an overdamped system, particularly in rotation.  
Critical damping is defined as $d\_c \= 2\\sqrt{k}$.  
For the rotational stiffness of 1000:

$$d\_c \= 2\\sqrt{1000} \\approx 63.2$$  
The user values (4) appear low if interpreted as absolute gains, but in many controller implementations (including parts of Isaac Lab depending on impedance\_mode), these values might be interpreted as damping ratios or scaled factors. If impedance\_mode="variable\_kp", the damping is often automatically computed to be critical based on the stiffness, and the user-supplied values might be multipliers.  
Snippet 2 and 3 clarify that motion\_damping\_ratio\_task is used in conjunction with stiffness. If the user is providing raw damping gains instead of ratios, the interpretation depends on the impedance\_mode.  
Assuming the system is stable (as reported), the high damping on rotation serves to suppress oscillation. When the saw hits the log, the impact creates a rotational impulse. A stiff, underdamped controller would "ring" (oscillate) around the target orientation. An overdamped controller absorbs this energy, returning strictly to the target angle.

### **3.4. Inertial Dynamics Decoupling**

Snippet 3 mentions inertial\_dynamics\_decoupling. This parameter determines whether the controller accounts for the full mass matrix of the robot when computing accelerations.

* **True:** The controller linearizes the dynamics, making the robot behave like a unit mass in all directions. This creates very pure stiffness behavior but requires accurate mass modeling.  
* **False:** The robot's natural inertia influences the response.

Given the heavy, offset load of the saw, enabling inertial\_dynamics\_decoupling (if not already set) would likely improve tracking performance by compensating for the specific inertial tensor of the saw (provided the saw's mass properties are correctly loaded in the USD).

## **4\. Root Cause Analysis: The Zero Error Fallacy**

The "Zero Error Fallacy" identified in the user's report is the definitive diagnostic clue. The controller reports EE Error \= 0.0000, yet the saw is visually tilted. This implies that the controller is perfectly minimizing the error function it has been given, but the function itself describes the wrong physical goal.

### **4.1. The Kinematic Chain Mismatch**

The standard kinematic chain for the robot is:  
$$ World \\rightarrow Base \\rightarrow Link\_1 \\dots \\rightarrow Link\_7 \\rightarrow Flange \\rightarrow EE (panda\_hand) $$  
The OSC controls the frame of the panda\_hand. When the user sets a target orientation of \`\` (RotY 180), the controller drives the panda\_hand frame to align with this target.  
However, the physical saw exists further down the chain:

$$EE (panda\\\_hand) \\rightarrow \[FixedJoint\] \\rightarrow SawBase \\rightarrow Blade$$  
The attachment between the hand and the saw is mediated by a static transform, $T\_{hand}^{saw}$. If this transform is not the identity matrix (meaning the saw is mounted at an angle), then aligning the hand to the world Z-axis does *not* align the saw to the world Z-axis.

### **4.2. Deconstructing the "Tilt"**

The actual orientation of the saw in the world frame is given by:

$$R\_{world}^{saw} \= R\_{world}^{hand} \\cdot R\_{hand}^{saw}$$  
The controller enforces:

$$R\_{world}^{hand} \\rightarrow R\_{target}$$  
If the user sets $R\_{target}$ to be "Vertical" (Identity, for simplicity):

$$R\_{world}^{saw} \= I \\cdot R\_{hand}^{saw} \= R\_{hand}^{saw}$$  
The saw's resulting orientation is exactly the orientation of the attachment offset. The user's observation that "RotX(90°) Made saw horizontal instead of vertical" provides the specific value of this offset.

* Input Command: 90 degrees.  
* Physical Result: 0 degrees (Horizontal).  
* Implies: $90^\\circ \+ Offset \\approx 0^\\circ \\Rightarrow Offset \\approx \-90^\\circ$.

This strongly suggests the FixedJoint or the Saw asset itself has a built-in \-90 degree rotation relative to the panda\_hand frame. The "Zero Error" confirms the controller has successfully rotated the hand to 90 degrees, but the \-90 degree offset cancels it out physically.

### **4.3. Comparative Analysis: RmpFlow vs. OSC**

This issue is not unique to OSC. Snippets 4 and 5 discuss similar issues with RmpFlow (Riemannian Motion Policies), another controller in Isaac Sim. In those cases, users found that the "converged end effector position has offset with target position" because "RmpFlow... has a different idea of where the end effector frame is than you do."  
The documentation suggests using visualize\_end\_effector\_position() to debug RmpFlow. Similarly, the user's employment of FrameTransformer in Phase 3 was the correct diagnostic move for OSC. It revealed the discrepancy between the visualized frame (which the controller targets) and the physical geometry.

## **5\. The Geometry of Attachment: Anatomy of a USD FixedJoint**

To understand why the offset exists, we must look inside the "Black Box" of the attachment, which is defined by the USD Physics schema. Snippets 6 and 7 provide the technical details of the FixedJoint.

### **5.1. The Dual-Frame Constraint**

A FixedJoint in USD does not simply glue two bodies together at their origins. It defines two local frames, one on each body, and constrains *those frames* to be coincident.

* **Body 0 (Hand):** Has a frame defined by physics:localPos0 and physics:localRot0.  
* **Body 1 (Saw):** Has a frame defined by physics:localPos1 and physics:localRot1.

The constraint enforced by the physics engine is:

$$T\_{world}^{Body0} \\cdot T\_{Body0}^{Frame0} \= T\_{world}^{Body1} \\cdot T\_{Body1}^{Frame1}$$  
The "Tilt" originates in $T\_{Body0}^{Frame0}$ or $T\_{Body1}^{Frame1}$.

* If physics:localRot0 is set to a 90-degree rotation, the Hand's origin will be rotated 90 degrees relative to the Saw's origin.  
* This is often done in asset preparation to align mounting holes or handles. For example, a saw handle is often 90 degrees or 45 degrees offset from the blade plane.

### **5.2. Visual vs. Physical Alignment**

In Phase 4, the user moved the spawn position of the saw ($X=1.0 \\rightarrow X=0.45$). While this placed the saw visually near the robot, it did not change the Constraint Definition.  
As snippet 8 explains: "You can move around the attached bodies relative to each other... when you click PLAY, the bodies will snap back together to satisfy the constraint."  
The spawn position is merely an initial condition. The steady-state relationship is defined entirely by the localPos and localRot attributes of the Joint. This explains why positioning fixes failed to resolve the tilt.

## **6\. Mathematical Fundamentals: Quaternion Conventions**

One of the most insidious sources of error in 3D simulation is the inconsistency in quaternion conventions. Research snippets 9, and 12 highlight a critical conflict that likely exacerbated the user's attempts to fix the tilt via "Fixed Orientation Targets."

### **6.1. The wxyz vs xyzw Conflict**

There are two primary ways to store a quaternion in a 4-element array:

1. **Scalar First (w, x, y, z):** Used by **Isaac Lab**, PyTorch, and the underlying math libraries of the controller.  
2. **Vector First (x, y, z, w):** Used by **USD**, PhysX, and sometimes ROS.

Snippet 9 explicitly warns: "Remember to switch all quaternions to use the xyzw convention when working indexing rotation data. Similarly, please ensure all quaternions are in wxyz before passing them to Isaac Lab APIs."

### **6.2. The Impact on Rotation Targets**

Consider the user's attempt to apply RotX(90°).

* Mathematically, a 90-degree rotation about X is: $q \= \[\\cos(45^\\circ), \\sin(45^\\circ), 0, 0\] \\approx \[0.707, 0.707, 0, 0\]$ (in wxyz).  
* If this array \[0.707, 0.707, 0, 0\] is interpreted by a system expecting xyzw:  
  * $x \= 0.707$  
  * $y \= 0.707$  
  * $z \= 0.0$  
  * $w \= 0.0$  
  * This represents a 180-degree rotation (since $w \= \\cos(\\theta/2) \= 0 \\Rightarrow \\theta \= 180^\\circ$) around the axis vector $(0.707, 0.707, 0)$.

This creates a completely different rotation than intended. If the user tried to compensate for the tilt by manually guessing quaternions, this coordinate mismatch would make the behavior appear chaotic or "unreachable" (as reported), because the robot would be commanded to twist in physically impossible ways.

### **6.3. Best Practices for Coordinate Interoperability**

To avoid this, explicit conversion is mandatory. Snippet 10 details the isaaclab.utils.math.convert\_quat function:

Python

def convert\_quat(quat, to="xyzw"):  
    """Converts quaternion from one convention to another."""

Any solution involving manual offsets must rigorously utilize this utility to ensure that the OperationalSpaceController (which expects wxyz) receives the correct data, regardless of how the USD asset (which stores xyzw) is configured.

## **7\. Solution Implementation: The body\_offset Architecture**

The structural solution to the "Saw Tilt" is to mathematically define the transformation between the Robot Flange (Controlled Frame) and the Saw Blade (Functional Frame). Isaac Lab provides a specific mechanism for this: the body\_offset parameter in the Action Configuration.

### **7.1. Analyzing OperationalSpaceControllerActionCfg**

Snippet 13 and 13 detail the configuration class for the OSC action term. This class is the interface between the user's environment script and the underlying controller.

Python

@configclass  
class OperationalSpaceControllerActionCfg(ActionTermCfg):  
    """Configuration for operational space controller action term."""  
    \#...  
    body\_name: str \= MISSING  \# The name of the body to control (e.g., "panda\_hand")  
    body\_offset: OffsetCfg | None \= None \# The crucial parameter

Snippet 13 describes OffsetCfg:  
"The offset pose from parent frame to child frame. On many robots, end-effector frames are fictitious frames that do not have a corresponding rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body."  
This is the architectural bridge. By defining body\_offset, we tell the OSC to compute the Jacobian not for the panda\_hand origin, but for a virtual point transformed by the offset.

### **7.2. Deriving the Offset**

We need to determine the transform $T\_{hand}^{blade}$.  
Based on the empirical evidence ($90^\\circ$ command $\\rightarrow$ $0^\\circ$ result), we hypothesize a \-90 degree rotation around X (or Y, depending on the axis definition).  
The body\_offset comprises two parts:

1. **Position (pos):** The linear distance from the hand center to the blade center.  
2. **Rotation (rot):** The quaternion rotation that aligns the Hand's axes with the Blade's axes.

### **7.3. Step-by-Step Implementation Guide**

Step 1: Visual Verification (Redux)  
Utilize the FrameTransformer from Phase 3\.

* Visualize panda\_hand. Note the Blue Axis (Z).  
* Visualize the saw geometry. Identify the vector perpendicular to the blade (the cutting normal).  
* If the Hand Z points UP, and the Saw Normal points HORIZONTAL, the rotation is 90 degrees.

Step 2: Constructing the Config  
The user must modify their environment configuration script. The code snippet below demonstrates the integration of the body\_offset.

Python

from isaaclab.envs.mdp.actions.actions\_cfg import OperationalSpaceControllerActionCfg  
from isaaclab.utils import configclass

@configclass  
class SawingEnvCfg:  
    \#... other config...  
      
    \# Define the Action Term  
    arm\_action \= OperationalSpaceControllerActionCfg(  
        asset\_name="robot",  
        joint\_names=\["panda\_joint.\*"\],  
        body\_name="panda\_hand", \# We control the hand...  
          
        \#...BUT we target the Saw Blade via this offset  
        body\_offset=OperationalSpaceControllerActionCfg.OffsetCfg(  
            \# Example: Blade is 15cm below and rotated 90 deg relative to hand  
            pos=(0.0, 0.0, 0.15),   
            \# Rotation of \-90 degrees around X (in w, x, y, z format)  
            \# w \= cos(-45) \= 0.707, x \= sin(-45) \= \-0.707  
            rot=(0.707, \-0.707, 0.0, 0.0)   
        ),  
          
        \# OSC Parameters from Phase 2  
        target\_types=\["pose\_abs"\],  
        impedance\_mode="variable\_kp",   
        motion\_stiffness\_task=\[100.0, 100.0, 10.0, 1000.0, 1000.0, 1000.0\],  
        motion\_damping\_ratio\_task=\[2.0, 2.0, 2.0, 4.0, 4.0, 4.0\],  
    )

Step 3: Execution and Verification  
When this config is loaded:

1. The OSC calculates the kinematic Jacobian $J\_{hand}$.  
2. It applies the rigid body transformation law to compute $J\_{blade}$ using the body\_offset.  
3. When the user commands "Target Orientation \= Vertical", the controller solves for joint torques that make the *Blade* Vertical.  
4. Visually, the *Hand* will tilt to \-90 degrees (canceling the offset), and the *Saw* will stand strictly upright.

## **8\. Conclusion and Future Outlook**

The investigation concludes that the simulation environment is fundamentally sound. The "Saw Tilt" is not a failure of the physics engine or the control law, but a semantic discrepancy in the definition of the end-effector.

### **8.1. Summary of Findings**

| Phase | Status | Mechanism of Success/Failure |
| :---- | :---- | :---- |
| **Physics** | **Stable** | $e=0.0$ and $v\_{depen}=0.1$ successfully dissipated collision energy and prevented LCP explosions. |
| **Control** | **Robust** | Anisotropic stiffness ($K\_{rot}=1000, K\_z=10$) correctly implemented hybrid behavior for cutting. |
| **Geometry** | **Misaligned** | FixedJoint introduced an uncompensated transform, leading to the "Zero Error Fallacy." |

### **8.2. Actionable Recommendations**

1. **Implement body\_offset:** This is the primary fix. It aligns the Control Frame with the Functional Frame.  
2. **Standardize Quaternions:** Strictly enforce (w, x, y, z) formatting using isaaclab.utils.math helpers to prevent axis confusion.  
3. **Enable Torsional Friction:** To further stabilize the saw-log contact during the cutting motion, set torsional\_patch\_radius \> 0 in the saw's rigid body settings.  
4. **Collision Masking:** As per snippet 6, ensure assembled\_robot.mask\_collisions is active between the hand and the saw to prevent internal physics jitter, which can manifest as subtle sensor noise even in a stable system.

By implementing the geometric offset correction, the system will achieve the final goal: a stable, compliant, and geometrically accurate robotic sawing simulation.

#### **Works cited**

1. isaaclab.sim.schemas — Isaac Lab Documentation, accessed November 23, 2025, [https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html)  
2. isaaclab.controllers — Isaac Lab Documentation, accessed November 23, 2025, [https://isaac-sim.github.io/IsaacLab/v2.0.0/source/api/lab/isaaclab.controllers.html](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/api/lab/isaaclab.controllers.html)  
3. isaaclab.controllers.operational\_space\_cfg — Isaac Lab ..., accessed November 23, 2025, [https://docs.robotsfan.com/isaaclab\_official/main/\_modules/isaaclab/controllers/operational\_space\_cfg.html](https://docs.robotsfan.com/isaaclab_official/main/_modules/isaaclab/controllers/operational_space_cfg.html)  
4. End-effector offset of RMP/IK \- Isaac Sim \- NVIDIA Developer Forums, accessed November 23, 2025, [https://forums.developer.nvidia.com/t/end-effector-offset-of-rmp-ik/228158](https://forums.developer.nvidia.com/t/end-effector-offset-of-rmp-ik/228158)  
5. How can I change end-effectors reference frame? \- Isaac Sim \- NVIDIA Developer Forums, accessed November 23, 2025, [https://forums.developer.nvidia.com/t/how-can-i-change-end-effectors-reference-frame/243681](https://forums.developer.nvidia.com/t/how-can-i-change-end-effectors-reference-frame/243681)  
6. Assemble Robots And Rigid Bodies \- Isaac Sim Documentation, accessed November 23, 2025, [https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot\_setup/assemble\_robots.html](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/assemble_robots.html)  
7. Simplifying joint position and rotation \- Isaac Sim \- NVIDIA Developer Forums, accessed November 23, 2025, [https://forums.developer.nvidia.com/t/simplifying-joint-position-and-rotation/264606](https://forums.developer.nvidia.com/t/simplifying-joint-position-and-rotation/264606)  
8. Objects connected through fixed joints change orientation after simulation start \- Isaac Sim, accessed November 23, 2025, [https://forums.developer.nvidia.com/t/objects-connected-through-fixed-joints-change-orientation-after-simulation-start/249030](https://forums.developer.nvidia.com/t/objects-connected-through-fixed-joints-change-orientation-after-simulation-start/249030)  
9. From IsaacGymEnvs — Isaac Lab Documentation, accessed November 23, 2025, [https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating\_from\_isaacgymenvs.html](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)  
10. isaaclab.utils — Isaac Lab Documentation, accessed November 23, 2025, [https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html)  
11. Change the coordinate in Isaac SIm \- NVIDIA Developer Forums, accessed November 23, 2025, [https://forums.developer.nvidia.com/t/change-the-coordinate-in-isaac-sim/287834](https://forums.developer.nvidia.com/t/change-the-coordinate-in-isaac-sim/287834)  
12. clemense/quaternion-conventions: An overview of different quaternion implementations and their chosen order: x-y-z-w or w-x-y-z? \- GitHub, accessed November 23, 2025, [https://github.com/clemense/quaternion-conventions](https://github.com/clemense/quaternion-conventions)  
13. isaaclab.envs.mdp.actions.actions\_cfg — Isaac Lab Documentation, accessed November 23, 2025, [https://docs.robotsfan.com/isaaclab\_official/v2.1.0/\_modules/isaaclab/envs/mdp/actions/actions\_cfg.html](https://docs.robotsfan.com/isaaclab_official/v2.1.0/_modules/isaaclab/envs/mdp/actions/actions_cfg.html)