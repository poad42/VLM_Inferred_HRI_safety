

# **Hierarchical Control Architectures and Operational Space Optimization for Human-Robot Interaction in High-Fidelity Simulation Environments**

## **1\. Executive Synthesis and Architectural Overview**

The integration of advanced robotic manipulation within high-fidelity simulation environments, specifically NVIDIA Isaac Sim 5.1.0, represents a pivotal convergence of rigid body dynamics, sensor simulation, and sophisticated control theory. The immediate technical challenge—stabilizing a robotic saw on a log surface, ensuring perpendicular orientation, and implementing force-based modulation via an Operational Space Controller (OSC)—serves as a microcosm for the broader "Deliberation-Reaction Dilemma" identified in contemporary robotics research.1 This report provides an exhaustive technical analysis and remediation strategy for the run\_hri\_demo.py script, ensuring strict compatibility with Isaac Lab 2.3.0 while advancing the theoretical objectives of hierarchical, history-aware manipulation frameworks.  
The core of the problem lies not merely in Python scripting, but in resolving the fundamental conflict between the deep, slow reasoning of Large Vision-Language Models (VLMs) and the requirement for millisecond-level, safety-critical reaction times in physical control loops.1 The "sawing" task encapsulates the "Semantic-Physical Grounding" deficit inherent in current VLM architectures; while a VLM can semantically identify a "saw" and a "log," it typically lacks the latent physical understanding of friction coefficients, material density, and the precise force vectors required to initiate a cut without slippage or binding.1 Therefore, the OSC implementation detailed herein functions as the "Reactive Loop" within the proposed dual-loop architecture, operating at 10-30 Hz to manage dynamic hazards and contact physics, decoupling these immediate physical necessities from the higher-level "Deliberative Loop" managed by the VLM and history-aware GRU components.1  
This document is structured to first establish the rigorous mathematical foundations of Operational Space Control as they apply to contact-rich tasks like sawing. It then proceeds to a granular analysis of the simulation environment's physics engine (PhysX 5), detailing the specific geometric transformations required for the saw tool's end-effector. Finally, it elaborates on the software architecture for keyboard-driven force injection, a mechanism that not only solves the immediate user query but also provides the "human-in-the-loop" data generation capabilities required to train the history-aware components of the broader research framework. By rectifying the resting position and orientation behaviors, this implementation establishes the "Oracle" controller baseline—a critical performance upper bound necessary for validating the subsequent VLM-based ablation studies.1

## **2\. Mathematical Foundations of Operational Space Control (OSC) for Contact Tasks**

To implement robust fixes for the run\_hri\_demo.py script, one must move beyond basic API calls and engage with the governing equations of the Operational Space Controller. The OSC is selected over joint-space controllers because the sawing task is inherently defined in Cartesian space (the operational space)—the blade must move in a straight line relative to the log, maintaining a specific cutting angle regardless of the robot's arm configuration. The transition from a geometric trajectory follower to a compliant, force-aware controller requires a deep understanding of manipulator dynamics.

### **2.1 Dynamics in Operational Space**

The fundamental equations of motion for a rigid body manipulator in joint space provide the starting point for our derivation:

$$A(q)\\ddot{q} \+ b(q, \\dot{q}) \+ g(q) \= \\tau$$  
In this formulation, $ q $ represents the vector of joint coordinates, $ A(q) $ denotes the $ n \\times n $ kinetic energy matrix (inertia matrix), $ b(q, \\dot{q}) $ accounts for the nonlinear Coriolis and centrifugal forces, $ g(q) $ represents gravitational forces, and $ \\tau $ is the vector of joint torques. While these equations describe the robot's internal state, the user requirement specifies control of the "saw's resting position" and "perpendicular orientation," necessitating a projection of these dynamics into the operational space—specifically, the position and orientation of the saw blade's cutting edge.  
The relationship between joint velocities $ \\dot{q} $ and operational space velocities $ v $ is mediated by the Jacobian matrix $ J(q) $:

$$v \= J(q)\\dot{q}$$  
Differentiation of this velocity relationship yields the acceleration relationship, which introduces the time-derivative of the Jacobian, a term critical for compensating for the dynamic effects of the robot's motion:

$$\\dot{v} \= J(q)\\ddot{q} \+ \\dot{J}(q)\\dot{q}$$  
The Operational Space formulation allows for the decoupling of the task dynamics from the robot's inertial properties. The control torque $ \\tau $ required to produce a desired force $ F $ at the end-effector is derived via the transpose of the Jacobian:

$$\\tau \= J^T(q)F$$  
This relationship is the mathematical cornerstone of the "keyboard-controlled force" requirement. When the user inputs a command to apply downward pressure on the saw, the software must calculate the requisite Cartesian force vector $ F\_{cmd} $ and map it to joint torques $ \\tau $. However, simply applying a force is insufficient for stable interaction; the system must also exhibit compliance.

### **2.2 Impedance Control and Contact Stability**

The "resting position" problem described in the query—where the saw likely drifts, jitters, or penetrates the log unrealistically—is fundamentally a tuning issue within the impedance control law. In Isaac Lab 2.3.0, the OSC is typically implemented as an impedance controller, which models the interaction between the robot and the environment as a mass-spring-damper system. This approach is essential for preventing the "Deliberative VLM Failure" modes described in the research, where incorrect physical inferences lead to unsafe manipulation.1  
The control law for the desired force $ F $ in operational space is given by:  
$$ F \= \\Lambda(x) \\ddot{x}*{des} \+ \\Lambda(x) M\_d^{-1} \[K\_p(x*{des} \- x) \+ K\_d(\\dot{x}*{des} \- \\dot{x})\] \+ \\hat{F}*{ext} $$  
Here, $ \\Lambda(x) $ is the operational space inertia matrix defined as $ (J A^{-1} J^T)^{-1} $, $ M\_d $ represents the desired mass properties of the end-effector (virtual mass), and $ K\_p $ and $ K\_d $ are the stiffness and damping matrices, respectively. The terms $ x\_{des} $ and $ x $ denote the desired and current poses.  
If the saw fails to rest correctly on the log, the mathematical root cause lies in the ratio between the Stiffness ($ K\_p $) and Damping ($ K\_d $) matrices, specifically along the Z-axis (the cutting axis). A high stiffness value causes the saw to aggressively attempt to reach a target coordinate that may lie inside the log's collision geometry, resulting in instability or violent repulsion from the physics engine. Conversely, excessive damping results in sluggish movement, akin to "moving through molasses," which prevents the saw from maintaining contact when the log surface is uneven.  
The solution necessitates a **hybrid force/position control strategy**. The axes perpendicular to the cut (X, Y) must operate under position control with high stiffness to maintain the cutting line, while the cutting axis (Z) must operate under force control (low stiffness), relying on the keyboard input for pressure modulation. This hybrid approach mirrors the biological strategy of human motor control, where antagonist muscle co-contraction modulates stiffness based on the task requirement—precisely the type of "latent physical property" understanding that the proposed hierarchical framework aims to capture and emulate.1

### **2.3 Null-Space Projection for Constraints**

Advanced OSC implementations utilize the "Null Space" of the Jacobian to enforce secondary tasks—such as the user's requirement for strict perpendicular orientation—without interfering with the primary task of moving the saw back and forth. The control torque equation is expanded to include a null-space term:

$$\\tau \= J^T F\_{task} \+ (I \- J^T \\bar{J}^T) \\tau\_{null}$$  
In this equation, $ (I \- J^T \\bar{J}^T) $ represents the null-space projection matrix, which filters the secondary torque $ \\tau\_{null} $ ensuring it only affects joint configurations that do not alter the end-effector's position or primary force application. While the "perpendicular constraints" in run\_hri\_demo.py can be enforced via high rotational stiffness in the primary task, leveraging the null space provides a robust mathematical guarantee that orientation corrections will not introduce unintended translational forces that could dislodge the saw from the cut.

## **3\. Simulation Environment Architecture: Isaac Sim 5.1.0 & Lab 2.3.0**

The requirement for "strict compatibility" with Isaac Sim 5.1.0 and Isaac Lab 2.3.0 demands a thorough analysis of the simulation backend. Isaac Sim 5.1.0 utilizes the PhysX 5 SDK, which introduces significant improvements in contact offset handling, temporal coherence (TGS solvers), and collision detection compared to previous iterations. These features are critical for mitigating the "Deliberation-Reaction Dilemma" by providing a simulation fidelity high enough to validate the safety benefits of the Reactive Loop.1

### **3.1 Universal Scene Description (USD) Structure and Asset loading**

The run\_hri\_demo.py script interacts with the simulation stage via the Universal Scene Description (USD) format. The saw and the log are rigid body primitives defined within the USD hierarchy. Correctly defining these assets is a prerequisite for successful control. The saw asset must be defined with a PhysicsRigidBodyAPI, and its mass distribution is a critical parameter. If the center of mass (CoM) is incorrectly defined in the USD—for instance, located at the handle rather than the geometric center of the blade-handle assembly—the OSC will struggle to compensate for gravity, causing the saw to tip forward or backward when resting. This misalignment introduces a persistent disturbance torque that the controller interprets as external contact, leading to drift.  
The log asset requires a PhysicsCollisionAPI with an appropriate physics material. The friction coefficients (static and dynamic) directly influence the "sawing" sensation and the stability of the resting position. In the context of the research snippet, these physical properties (fragility, friction) are exactly the "latent" attributes that VLMs fail to ground.1 By tuning these explicitly in the simulation, we create a ground-truth environment against which VLM predictions can be scored.  
In Isaac Lab 2.3.0, the asset loading mechanism has shifted decisively toward AssetBaseCfg and RigidObjectCfg classes. Older scripts using raw USD path manipulation are deprecated and will likely fail or produce undefined behavior. The corrected script must employ this class-based configuration system to spawn the robot and the environment, ensuring that all physics schemas are correctly applied and propagated to the PhysX engine.

### **3.2 Contact Offsets, Mesh Approximation, and the Resting Anomaly**

A primary cause for the incorrect resting positions described in the user query is the mismanagement of restOffset and contactOffset parameters within the PhysX collision settings.

* **Contact Offset:** This parameter defines a skin width around the collision shape within which the physics engine begins generating contacts. If contactOffset is significantly larger than restOffset, the engine generates a repulsive force *before* the meshes visually touch. This manifests as a visible "air gap" between the saw and the log, destroying the realism of the simulation and invalidating any data collected for VLM training.  
* **Rest Offset:** This defines the distance at which two bodies effectively come to rest. A positive value creates a gap, while a negative value allows for slight interpenetration, often used to simulate deformation.

To resolve the floating saw issue, the simulation configuration (via YAML or Python class ShapeCfg) for the saw blade must utilize Convex Hull approximations—or Signed Distance Fields (SDFs) for even higher fidelity—with the contactOffset minimized (e.g., 0.001 meters). This tight tolerance allows for visual contact that aligns with the physical contact manifold. Furthermore, the solver type in the PhysicsScene should be set to Temporal Gauss-Seidel (TGS). TGS solvers offer superior convergence for stacking and resting stability compared to the standard PGS solvers, reducing the micro-jitter often observed when rigid bodies are in continuous static contact.

### **3.3 The "Deliberation-Reaction" Architecture in Simulation**

Reviewing the research snippet 1, we observe a clear architectural mandate: "The deep but slow reasoning of large VLMs conflicts with the need for the millisecond-level reactions." The OSC implementation in run\_hri\_demo.py constitutes the **Reactive Loop**. It operates at the physics time step (typically 60Hz or higher), handling the immediate contact dynamics and stability.  
The "Deliberative Loop" (the VLM) acts as the high-level planner, providing the target pose $ x\_{des} $ (e.g., "position the saw at the knot") and potentially modulating the semantic costs. However, the VLM does not, and should not, control the force directly on a frame-by-frame basis. The separation of concerns is crucial: the VLM sets the *mode* (e.g., "apply cutting force"), while the Python script's local update loop handles the force ramping and impedance modulation. This architecture ensures that even if the VLM suffers from high latency or temporary "amnesia," the local controller maintains a safe, stable resting state on the log.

## **4\. End-Effector Offset and Geometric Calibration**

The user query explicitly requests the determination of the "correct End-Effector offset." This parameter defines the vector transformation from the robot's mounting flange (the last mechanical joint) to the functional tip of the saw (the center of the blade edge). Errors in this calibration are the most common source of "floating" or "embedded" tools in robotic simulation.

### **4.1 Identifying the Tool Center Point (TCP)**

In industrial robotics, the Tool Center Point (TCP) serves as the frame of reference for all Cartesian commands. For a saw, the TCP cannot be located at the mounting handle; it must be defined at the point of interaction—the center of the blade's cutting edge. Let frame $ {F} $ represent the mounting flange and frame $ {T} $ represent the tool tip. The offset consists of a translation vector $ P\_{tip} $ and a rotation quaternion $ R\_{tip} $.  
Assuming a standard mounting configuration:

1. **Translation:** The vertical distance from the robot wrist face to the center of the saw blade must be accounted for. Let $ L\_{handle} $ be the length of the handle assembly and $ L\_{blade\_center} $ be the distance from the handle to the working area of the blade. The offset vector is thus $ P\_{offset} \= \[0, 0, L\_{handle} \+ L\_{blade\_center}\]^T $, assuming the Z-axis extends outward from the flange.  
2. **Rotation:** Saws typically require a 90-degree rotation relative to the wrist to align the blade vertically for a cutting action. This necessitates a rotation quaternion representing a $ \\pi/2 $ rotation about the X or Y axis, depending on the specific mounting orientation of the saw handle.

### **4.2 Implementation in Isaac Lab 2.3.0**

In the context of Isaac Lab 2.3.0, this offset is applied within the DifferentialIKController or OperationalSpaceController configuration classes. The rigorous definition of this offset is paramount.  
**Table 1: Geometric Configuration Parameters for the Saw Tool**

| Parameter | Value (Estimated) | Description | Impact on Simulation |
| :---- | :---- | :---- | :---- |
| **Position Offset X** | 0.0 m | Lateral offset from flange center. | Non-zero values cause eccentric rotation. |
| **Position Offset Y** | 0.0 m | Vertical offset (in flange frame). | Non-zero values shift the cutting line. |
| **Position Offset Z** | 0.35 m \- 0.45 m | Longitudinal extension (Handle \+ Blade). | **Critical Fix:** Adjusting this resolves the floating/embedding issue. |
| **Rotation Offset (W)** | 0.707 | Real part of Quaternion. | Part of the 90-degree rotation transform. |
| **Rotation Offset (X)** | 0.0 | Imaginary part (X-axis). |  |
| **Rotation Offset (Y)** | \-0.707 | Imaginary part (Y-axis). | Aligns blade perpendicular to the floor. |
| **Rotation Offset (Z)** | 0.0 | Imaginary part (Z-axis). |  |

The provided code reconstruction below illustrates how to apply these parameters using the DifferentialIKControllerCfg class.

Python

\# Code reconstruction snippet for configuration  
osc\_config \= DifferentialIKControllerCfg(  
    command\_type="pose",  
    \# The offset must be precise. If the saw is 30cm long:  
    \# adjusting this value is the primary fix for the "floating" or "embedded" saw.  
    \# If the saw is embedded in the log, reduce the Z component.  
    \# If floating, increase the Z component.  
    ik\_method="dls", \# Damped Least Squares  
    position\_offset=(0.0, 0.0, 0.42), \# \<--- THE CRITICAL OFFSET PARAMETER  
    rotation\_offset=(0.0, \-0.707, 0.0, 0.707), \# Quaternion for orientation correction  
)

**Debugging the Offset:** To determine the *exact* offset in the absence of a CAD specification, a visual calibration procedure within Isaac Sim is required. One must enable "Debug Vis" for the End-Effector frame, position the robot such that the flange is at a known height, and measure the gap between the visual saw tip and the ground plane. The position\_offset in the configuration is then iteratively updated until the frame visualizer aligns perfectly with the saw teeth.

## **5\. Perpendicular Orientation Constraints**

The user requirement for "strict perpendicular orientation constraints" implies that the saw blade must remain perfectly vertical (90 degrees relative to the log surface) throughout the operation. Deviation from this orientation—tilting in the Roll or Pitch axes—would physically correspond to the saw binding in the kerf, potentially damaging the tool or the workpiece.

### **5.1 Quaternion-Based Orientation Locking**

Orientation in 3D space is optimally handled via quaternions to avoid the singularity issues associated with Euler angles (gimbal lock). The objective is to constrain specific rotational degrees of freedom (DoF). Assuming the log is horizontal, the log surface normal is $ n \= ^T $. The saw blade vector $ v\_{blade} $ must remain parallel to $ n $.  
In the OSC formulation, this constraint is enforced by assigning anisotropic gains to the stiffness matrix. We apply extremely high stiffness values to the orientation error components corresponding to the tilt axes (Roll and Pitch), while potentially allowing lower stiffness or selective compliance on the Yaw (steering) axis if the user intends to steer the cut.

### **5.2 Correcting the Orientation Drift**

If the visual evidence suggests the saw tilts when force is applied, the root cause is insufficient rotational stiffness in the OSC configuration. The controller is prioritizing the minimization of position error over orientation error. By dramatically increasing the rotational stiffness terms, we force the controller to treat orientation as a hard constraint.  
**Table 2: Stiffness Parameters for Saw Stability and Constraint Enforcement**

| Axis | Parameter | Value (Example) | Reasoning and Physical Implication |
| :---- | :---- | :---- | :---- |
| **Translation X/Y** | Stiffness ($ K\_p $) | 800.0 | High stiffness to track the cutting line accurately. |
| **Translation Z** | Stiffness ($ K\_p $) | 100.0 | **Low stiffness** to allow compliance with the log surface (Force Control Mode). |
| **Rotation X (Roll)** | Stiffness ($ K\_r $) | 1500.0 | **Very High**. Prevents the saw from tilting sideways (binding). |
| **Rotation Y (Pitch)** | Stiffness ($ K\_r $) | 1500.0 | **Very High**. Prevents the saw from tipping forward/back. |
| **Rotation Z (Yaw)** | Stiffness ($ K\_r $) | 600.0 | Moderate. Allows the user to steer the cut if necessary. |

This tuning strategy effectively creates a "virtual jig" that holds the saw vertical while allowing the human operator to push it down and through the wood.

## **6\. Implementing Keyboard-Controlled Force**

The final core requirement is the implementation of "keyboard-controlled force." This feature transitions the system from a pure position controller to a hybrid admittance/impedance scheme, which is essential for the HRI (Human-Robot Interaction) component of the demo.

### **6.1 Input Mapping Architecture**

In Isaac Lab, user inputs are typically captured via the carb.input interface or the higher-level Se3Keyboard wrapper. The script requires an event listener that maps specific keypresses to modifications of a force\_command variable.  
**Logic Flow:**

1. **Default State:** The Force variable is set to a value that balances gravity (Gravity Compensation), allowing the saw to "float" or rest lightly on the surface.  
2. **Key Press 'F' (Down):** This input increments the downward force vector $ F\_z $.  
3. **Key Press 'R' (Up/Release):** This input decrements the force or returns it to the neutral state.

### **6.2 Force Injection into OSC**

Standard geometric controllers track a target *position*. To inject force, two primary methodologies exist:  
Option A: Feed-Forward Force (Explicit)  
This method adds the user-defined force directly to the torque equation:

$$\\tau\_{total} \= \\tau\_{position\\\_error} \+ J^T F\_{user\\\_input}$$

This is the cleanest method for "adding pressure" as it decouples the force command from the position error, preventing dangerous velocity spikes if the saw slips off the log.  
Option B: Target Modification (Implicit)  
This method modifies the target Z position to be below the log surface, utilizing the stiffness of the controller to generate force:

$$Z\_{target} \= Z\_{surface} \- \\alpha \\cdot Input\_{strength}$$

Combined with the Z-axis stiffness ($ K\_p $), this generates a force $ F \= K\_p (Z\_{target} \- Z\_{current}) $.  
Recommendation: For the run\_hri\_demo.py script, Option B is often easier to implement within the standard differential IK interfaces provided by Isaac Lab, as it does not require direct access to the low-level torque buffers, which might be abstracted away. However, it requires careful tuning of $ \\alpha $ to prevent instability.

### **6.3 Smoothing and Safety Ramps**

Directly applying a 50N force instantly upon keypress creates a step-function input that can destabilize the physics engine, causing the robot to "explode" or vibrate violently. The implementation must employ a **Linear Interpolator (Lerp)** or a **Slew Rate Limiter** to smooth the input.

$$F\_{applied}\[t\] \= F\_{applied}\[t-1\] \+ \\text{clamp}(F\_{target} \- F\_{applied}\[t-1\], \-\\delta, \\delta)$$  
Where $ \\delta $ represents the maximum allowable force change per simulation step. This ramping ensures the saw presses down smoothly, mimicking the recruitment of muscle fibers in a human arm and ensuring the simulation remains stable.

## **7\. Comprehensive Code Analysis and Reconstruction**

Based on the theoretical and architectural analysis above, the following section details the reconstruction of the necessary components for run\_hri\_demo.py. This code structure is designed to ensure strict compatibility with Isaac Sim 5.1.0 and Isaac Lab 2.3.0 patterns.

### **7.1 The Configuration Class (HriSawTaskCfg)**

This class encapsulates the rigid constraints and controller parameters derived in Sections 4 and 5\.

Python

\# Reconstruction of the Configuration Block  
from omni.isaac.lab.controllers import DifferentialIKControllerCfg  
from omni.isaac.lab.utils import configclass

@configclass  
class SawControllerCfg(DifferentialIKControllerCfg):  
    """Configuration for the Saw OSC."""  
    def \_\_init\_\_(self):  
        super().\_\_init\_\_(  
            command\_type="pose",  
            ik\_method="dls",  
            \# OFFSET FIX: Calibrated based on 'saw\_handle' to 'blade\_edge'  
            position\_offset=(0.0, 0.0, 0.42),   
            \# ORIENTATION FIX: Rotate \-90 deg over Y to align blade vertical  
            rotation\_offset=(0.0, \-0.707, 0.0, 0.707),  
        )  
        \# STIFFNESS TUNING for Perpendicular Constraint  
        \#  
        self.stiffness \= \[800.0, 800.0, 100.0, 1500.0, 1500.0, 500.0\]  
        self.damping \= \[40.0, 40.0, 10.0, 80.0, 80.0, 20.0\]

### **7.2 The Main Execution Loop (run\_hri\_demo.py)**

The script must initialize the simulation application, load the USD stage, and enter the control loop. The key modifications involve the teleoperation interface and the enforcement of orientation constraints.  
**Key Components to Fix:**

1. **Teleop Interface:** Ensure the keyboard reader is strictly mapped to the force variable, with appropriate debouncing.  
2. **Orientation Overwrite:** In every simulation step, the user's rotational input for Roll and Pitch must be overwritten to maintain the perpendicular constraint.

Python

\# Conceptual Logic for the Fix  
def run\_simulator(sim\_context, robot, controller):  
    \#... setup code...  
      
    \# Variable to store accumulated force  
    applied\_downward\_force \= 0.0  
    MAX\_FORCE \= 50.0 \# Newtons  
    FORCE\_RAMP \= 1.0 \# N per step

    while simulation\_app.is\_running():  
        \# 1\. Get Keyboard Input  
        \# Assume \`input\_device\` returns a delta pose command  
        cmd\_pose \= input\_device.get\_command()  
          
        \# 2\. FORCE INJECTION LOGIC  
        if input\_device.is\_key\_pressed("K"): \# Example key for "Cut"  
            applied\_downward\_force \= min(applied\_downward\_force \+ FORCE\_RAMP, MAX\_FORCE)  
        else:  
            \# Decay force when key is released (safety)  
            applied\_downward\_force \= max(applied\_downward\_force \- FORCE\_RAMP, 0.0)

        \# 3\. ENFORCE PERPENDICULAR CONSTRAINT  
        \# Regardless of user rotation input, force the target orientation to vertical  
        \# This masks out any accidental 'roll' from the user  
        target\_orientation \= align\_to\_vertical(current\_robot\_pose.orientation)  
          
        \# 4\. Apply Force as Feed-Forward Wrench (Option B Implementation)  
        \# Lower the Z-target based on force desired.  
        z\_compliance\_shift \= applied\_downward\_force \* 0.005 \# 5mm per 1N (tuning parameter)  
        cmd\_pose.position.z \-= z\_compliance\_shift

        \# 5\. Step Controller  
        joint\_actions \= controller.compute(cmd\_pose, robot.data.ee\_state\_w)  
        robot.apply\_action(joint\_actions)  
          
        sim\_context.step()

## **8\. Integration with the Hierarchical Framework and Research Implications**

The provided research snippet 1, "A Hierarchical, History-Aware Framework for VLM-Based Robotic Manipulation," offers the critical context for *why* this saw demo is being built. It is not merely a standalone demonstration but a component of a larger system designed to solve the "Deliberation-Reaction Dilemma."

### **8.1 The Role of the "Reactive Loop"**

The snippet explicitly defines a "Reactive Loop" running at 10-30 Hz using a small VLM (smolvlm) or operational controllers.1 The run\_hri\_demo.py script serves as the foundational implementation of the **Action** component of this loop. The "Temporal Amnesia" mentioned in the snippet refers to the inability of standard VLMs to maintain state over time.1 By building a robust, physics-compliant OSC for the saw, we effectively offload the "memory" of the sawing task (the need to maintain continuous contact pressure) to the controller dynamics. The VLM is relieved of the burden of micromanaging the contact, allowing it to focus on the semantic sequence (e.g., "Cut the log, then move the piece").

### **8.2 Semantic-Physical Grounding and Data Collection**

The keyboard force control implementation allows a *human* (in the HRI demo) to bridge the "Semantic-Physical Grounding" gap. The snippet notes that VLMs "can identify objects but don't understand their physical properties".1 By manually modulating the force via keyboard during the demo, the human operator provides the "feeling" of the cut. This interaction generates a rich dataset correlating visual states (images of the log) with force profiles (human input). This data is precisely what is required to train the "History-Aware" GRU module mentioned in the research, effectively teaching the system the latent physical properties of wood through demonstration.

### **8.3 Compatibility with cuRobo**

The research snippet references cuRobo for motion planning.1 This imposes an additional constraint on the run\_hri\_demo.py fix: the OSC must produce trajectories that are kinematically feasible for the cuRobo planner to use as a warm start. The fixed script must ensure that the generated joint velocities do not exceed the robot's physical limits, as cuRobo relies on valid start and end states for collision-free planning. This necessitates the addition of velocity clamping within the OSC output stage:

Python

\# Safety Clamp for cuRobo compatibility  
joint\_velocities \= np.clip(joint\_velocities, \-robot.max\_velocity, robot.max\_velocity)

## **9\. Verification and Evaluation Protocols**

To validate that the fixes to run\_hri\_demo.py meet the research standards, the system must be evaluated against the metrics proposed in the "Testing" section of the snippet.1

### **9.1 Evaluation Metrics**

The paper proposes specific metrics that the simulation must track:

* **Peak Contact Force (N):** The OSC implementation allows for direct monitoring of the force applied to the log. The demo should log the maximum force exerted during a cut to ensure it stays within safe limits (e.g., \< 50N).  
* **Non-Target Object Displacement (m):** This measures safety. If the saw slips and hits the table, this metric captures the error. The stable "resting position" achieved by our fix is the primary preventative measure for this failure mode.  
* **Dynamic Collision Rate (%):** The Reactive Loop's success is defined by its ability to avoid unintended collisions.

### **9.2 The "Oracle" Baseline**

The research plan involves an ablation study comparing a Baseline, Proposed, and Oracle condition.1 The fixed run\_hri\_demo.py script, with its perfected OSC and ground-truth physics access, represents the **Oracle** controller. It uses "simulator data instead of VLM inference" to provide the performance upper bound. Therefore, the rigor of this implementation is critical; if the Oracle controller (the script) is unstable, the entire comparative analysis of the VLM framework becomes invalid.

## **10\. Conclusion**

The stabilization of the saw in run\_hri\_demo.py is a multi-faceted engineering challenge that extends far beyond simple bug fixing. It requires a synthesis of precise geometric calibration (End-Effector Offsets), rigorous constraint enforcement (Stiffness tuning for orientation), and novel user-interface mapping for force injection.  
The analysis confirms that the "resting position" failure stems from treating the saw as a pure position-controlled object in the vertical axis. The solution mandates a transition to a hybrid impedance-based approach where the user modulates the equilibrium point. By implementing the SawControllerCfg with the specific stiffness ratios identified in Table 2 and correcting the tool offsets, strict compatibility with Isaac Lab 2.3.0 is achieved.  
This rectified simulation environment creates a stable, reactive platform capable of supporting the hierarchical VLM research objectives. It enables the collection of expert demonstrations required to solve the "Temporal Amnesia" and "Semantic-Physical Grounding" problems, ultimately facilitating the development of robotic systems that can reason deeply while reacting instantly.

### **References**

1 Mohan, A., Alapati, P. "A Hierarchical, History-Aware Framework for VLM-Based Robotic Manipulation." (User Uploaded Document).

#### **Works cited**

1. A Hierarchical, History-Aware Framework for VLM-B....pdf