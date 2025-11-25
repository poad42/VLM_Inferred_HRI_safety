

# **Adaptive Impedance Modulation and Kinematic Extension in Robotic Manipulation: A Hierarchical VLM Framework for Contact-Rich Tasks**

## **Executive Summary**

The convergence of large-scale semantic reasoning, embodied by Vision-Language Models (VLMs), and high-frequency robotic control, typified by Operational Space Controllers (OSC), presents a paradigm shift in autonomous manipulation. This report provides an exhaustive technical analysis and implementation roadmap for integrating VLMs into the NVIDIA Isaac Lab simulation environment to address complex, contact-rich tasks such as sawing. Specifically, it addresses the modification of standard Human-Robot Interaction (HRI) scripts (e.g., run\_hri\_demo.py) to achieve two distinct objectives: the modulation of physical control parameters (stiffness and damping) based on semantic visual cues, and the rigid enforcement of prismatic kinematic constraints (rail systems) to stabilize manipulation trajectories.  
The analysis draws heavily from the provided research on "A Hierarchical, History-Aware Framework for VLM-Based Robotic Manipulation" 1, synthesizing its theoretical propositions—specifically the Deliberative-Reaction Dilemma and Semantic-Physical Grounding—with the practical software engineering patterns required for Isaac Sim 5.1.0. The report identifies that the user's observed "push-pull" instability in the sawing task is a fundamental limitation of static impedance parameters in OSC, which blindly attempt to minimize position error against a constraint (the material being sawed).  
We propose a dual-loop architecture where a VLM serves as a "Parameter Supervisor," dynamically adjusting the stiffness matrix of the controller based on material identification (e.g., differentiating between sawing wood vs. metal) and safety context. Furthermore, the report details the Universal Scene Description (USD) and PhysX implementation details required to ground these manipulators on prismatic rails. This extends the robot's workspace and linearizes its kinematic capabilities, directly addressing the user's need for straighter sawing trajectories. The proposed solution replaces static gain scheduling with a dynamic, VLM-driven impedance surface, effectively allowing the robot to "read" the fragility and dynamic properties of its environment, thereby resolving the conflict between the OSC's waypoint tracking and the physical constraints of the sawing task.  
---

## **1\. Introduction: The Intersection of Semantics and Dynamics**

The field of robotic manipulation is currently undergoing a transition from strictly geometric planning—where the world is perceived as a collection of collision meshes and coordinate frames—to semantic planning, where objects are understood in terms of their utility, fragility, and physical properties. However, a significant gap remains: the translation of high-level semantic understanding into low-level control guarantees. This report addresses this gap within the context of the NVIDIA Isaac Lab simulation environment, specifically targeting the user's query regarding the run\_hri\_demo.py script and the integration of Vision-Language Models (VLMs) for parameter modulation.

### **1.1 The Operational Context: The "Sawing" Problem**

The user has identified a specific experimental scenario: a robotic arm attempting to saw an object. In this scenario, the robot experiences a "push-pull" effect, where the Operational Space Controller (OSC) senses the resistance of the material and, treating it as a disturbance, attempts to force the end-effector back to the original trajectory waypoint. This is a classic failure mode of high-stiffness position control in contact-rich tasks. The controller is fighting the task physics rather than complying with them.  
The integration of a VLM is questioned by the user ("I struggle to find the relevance..."). This report argues that the VLM is not merely relevant but essential for generalized sawing. While a rail constraint (geometry) ensures the saw moves straight, a VLM (semantics) determines *how* the saw moves. It answers critical questions: What material is being cut? Is the blade binding? Is the cut progressing? Is a human hand interfering? Without semantic grounding, the robot is blind to the process, relying solely on proprioceptive force feedback which, as observed, leads to instability.

### **1.2 The Deliberation-Reaction Dilemma**

A recurring theme in the provided research material 1 is the "Deliberation-Reaction Dilemma." Foundation models like VLMs operate on a timescale of seconds (0.5Hz \- 2Hz), while robotic stability requires control loops operating at milliseconds (50Hz \- 1000Hz). Integrating a slow "brain" into a fast "body" creates a latency mismatch that can be catastrophic in contact tasks. If the VLM detects a change in material density but takes 2 seconds to lower the arm's stiffness, the saw blade may already have snapped or jammed.  
The solution proposed in this report mirrors the "Hierarchical, History-Aware Framework" 1, utilizing a dual-loop architecture. The VLM operates asynchronously, updating a "Context Vector," while a high-frequency recurrent neural network (GRU) or a reactive policy acts as a bridge, smoothing these updates and ensuring real-time stability.

### **1.3 Scope of the Report**

This document is structured to provide a comprehensive engineering guide.

* **Section 2** establishes the theoretical control physics, explaining why the user's OSC is failing and deriving the impedance laws.  
* **Section 3** explores the Semantic-Physical Grounding theory from the provided PDF.  
* **Section 4** details the Isaac Lab architecture.  
* **Section 5 & 6** provide the step-by-step implementation of the Prismatic Rail and VLM integration.  
* **Section 7** specifically addresses the Sawing Case Study.  
* **Section 8** outlines validation metrics.

---

## **2\. Theoretical Framework: Physics of Contact and Control**

To understand how to modify run\_hri\_demo.py, one must first master the underlying control theory implemented in Isaac Lab's OperationalSpaceController. The user's "push-pull" issue is a direct manifestation of interaction dynamics.

### **2.1 Rigid Body Dynamics and Operational Space Formulation**

The dynamics of a manipulator in joint space are described by the equation:

$$M(q)\\ddot{q} \+ C(q, \\dot{q})\\dot{q} \+ g(q) \= \\tau \+ J^T(q)F\_{ext}$$

Where:

* $M(q)$ is the mass/inertia matrix.  
* $C(q, \\dot{q})$ represents Coriolis and centrifugal forces.  
* $g(q)$ is the gravity vector.  
* $\\tau$ is the vector of joint torques.  
* $J(q)$ is the Jacobian matrix mapping joint velocities to end-effector velocities.  
* $F\_{ext}$ is the external force at the end-effector (e.g., the reaction force from the wood being sawed).

In Operational Space Control (OSC), we decouple the task dynamics from the joint dynamics. We define the task in Cartesian coordinates $x$. The control law attempts to linearize the dynamics in task space:

$$F\_{cmd} \= \\Lambda(x)\\ddot{x}\_{des} \+ \\mu(x, \\dot{x}) \+ p(x) \+ F\_{task}$$

Where $\\Lambda$ is the operational space inertia matrix (the effective mass felt at the end-effector).

### **2.2 Impedance Control and the "Push-Pull" Phenomenon**

The term $F\_{task}$ in the OSC formulation is typically governed by a PD (Proportional-Derivative) law, which essentially models a virtual spring-damper system:

$$F\_{task} \= K\_p (x\_{des} \- x) \+ K\_d (\\dot{x}\_{des} \- \\dot{x})$$

Here, $K\_p$ is the stiffness matrix and $K\_d$ is the damping matrix.  
The "Push-Pull" Mechanism:  
In the user's sawing experiment, the robot pushes the saw forward ($x\_{des}$ advances). The wood exerts a friction force $F\_{friction}$ backwards. This causes the actual position $x$ to lag behind $x\_{des}$.

* The error $(x\_{des} \- x)$ grows.  
* The term $K\_p (x\_{des} \- x)$ increases, demanding more torque to "push" back to the waypoint.  
* If the saw blade binds or sticks, the robot pushes harder.  
* Eventually, the force overcomes static friction, the saw jumps forward, overshoots, and the controller tries to "pull" it back.  
  This oscillation is the "push-pull" effect. It arises because the controller is prioritizing position over interaction.

### **2.3 Variable Impedance as a Solution**

To solve this, we must modulate $K\_p$. In the direction of the cut (say, Z-axis), we want high stiffness to maintain a straight line. However, along the feed axis (say, Y-axis), we might want *lower* stiffness or force control to maintain a constant pressure rather than a constant position.  
The relevance of the VLM becomes clear here: Parameter Modulation.  
The optimal stiffness $K\_p$ is not a constant. It depends on the environment.

* **Sawing Foam:** Low resistance. High $K\_p$ is fine.  
* **Sawing Hardwood:** High resistance. High $K\_p$ causes binding. We need lower $K\_p$ (compliance) or a switch to force control.  
* **Sawing Metal:** Requires lubricant and very slow feed rates.

Since the robot cannot inherently "know" it is sawing metal versus foam just by touching it (without complex system identification), the VLM provides the **prior** distribution. It looks at the scene, identifies "Metal Pipe," and informs the controller: "Set feed-axis stiffness to 50 N/m and target force to 20N." This is Semantic-Physical Grounding.  
---

## **3\. Semantic-Physical Grounding via VLMs**

The provided research document "A Hierarchical, History-Aware Framework for VLM-Based Robotic Manipulation" 1 offers the theoretical architecture to implement this modulation.

### **3.1 The Semantics Gap in Robotics**

Standard robotic perception uses object detectors (YOLO, MaskRCNN) which provide bounding boxes: Class: Cup. This is semantically shallow. It does not convey Fragility, Friction, Contents, or Temperature. A VLM, trained on internet-scale data, captures these latent properties. It knows that "Ceramic Cup" implies "Brittle" and "Heavy," whereas "Paper Cup" implies "Deformable" and "Light."  
The report 1 identifies the core challenge: "VLMs can identify objects but don't understand their physical properties... This gap... can lead to unsafe robot actions."  
The goal of integrating the VLM into run\_hri\_demo.py is to bridge this gap by converting semantic tokens into physical impedance gains.

### **3.2 The Deliberative-Reaction Architecture**

The PDF proposes a hierarchical structure to handle the latency of VLMs:

| Layer | Component | Frequency | Responsibility |
| :---- | :---- | :---- | :---- |
| **Deliberative** | VLM (e.g., GPT-4V, LLaVA) | 0.2 \- 1 Hz | Scene understanding, material classification, task planning, safety constraint generation. |
| **History-Aware** | GRU / RNN | 10 \- 50 Hz | Smoothing VLM outputs, maintaining context (memory), filtering hallucinations. |
| **Reactive** | OSC / Impedance Controller | 500 \- 1000 Hz | Real-time torque generation, immediate collision reflex. |

**Insight:** This architecture directly addresses the user's skepticism about VLM relevance. The VLM is not driving the saw in real-time. It is *tuning* the reflex loops of the saw. It acts as the "Coach," not the "Athlete."

### **3.3 Temporal Amnesia and the GRU**

A critical insight from the research snippet 1 is the problem of "Temporal Amnesia." VLMs are stateless. If the camera angle shifts and the saw blade is occluded, the VLM might stop reporting "Sawing." If the controller relies solely on the immediate VLM output, it might suddenly revert to "Free Motion" stiffness, causing the saw to jerk dangerously.  
The Gated Recurrent Unit (GRU) serves as a temporal anchor. It maintains a hidden state vector $h\_t$ that encodes the belief of the system.

$$h\_t \= (1 \- z\_t) \\odot h\_{t-1} \+ z\_t \\odot \\tilde{h}\_t$$

Even if the VLM misses a detection for 5 frames, the GRU's memory ($h\_{t-1}$) sustains the "Sawing Mode" context, preventing parameter discontinuities. This is essential for the smooth operation required in the sawing task.  
---

## **4\. Deep Dive: The Isaac Lab Simulation Environment**

To implement these theories, we must understand the target platform: NVIDIA Isaac Lab (built on Isaac Sim 5.1.0 and Omniverse).

### **4.1 Architecture of Isaac Lab**

Isaac Lab differs from traditional simulators (like Gazebo) by using a USD-centric data model and a GPU-accelerated physics engine (PhysX 5).

* **USD (Universal Scene Description):** The scene graph is a hierarchy of "Prims" (Primitives). Modifying the robot to add a rail involves editing this USD hierarchy, not just Python code.  
* **Fabric:** A low-latency communication layer that moves data between the PhysX engine and the Python control script without CPU-GPU copying.  
* **Cloning:** Isaac Lab allows for massive parallelization (thousands of environments). Our VLM integration must be efficient enough not to bottleneck this, or it must be implemented to run on a single "Showcase" environment while others train.

### **4.2 Anatomy of run\_hri\_demo.py**

The script provided in the user query is a standard demo script. Its execution flow is typically:

1. **App Launch:** Starts the simulator.  
2. **Design Scene:** Loads the robot USD and environment assets.  
3. **Reset:** Initializes the robot state.  
4. **Simulation Loop:**  
   * sim.step(): Advances physics.  
   * get\_observations(): Reads joint states and camera images.  
   * compute\_action(): Calculates OSC torques.  
   * apply\_action(): Sends torques to the robot.

**Integration Points:**

* **Rail Constraint:** Must be injected during the Design Scene phase. The kinematic chain must be altered *before* the physics engine compiles the articulation.  
* **VLM Inference:** Must be injected into the Simulation Loop. However, since compute\_action runs at physics frequency (e.g., 60Hz), and the VLM runs at 1Hz, the VLM call **must** be asynchronous. Blocking the simulation loop for 1 second to wait for a VLM response will cause the physics to hang and the controller to crash.

---

## **5\. Implementation Part I: Prismatic Rail Constraints**

The user's request includes a "secondary goal": research how to add a railing for the saw. This is a geometric constraint problem. In simulation, this is achieved by mounting the robot base on a prismatic joint.

### **5.1 Why a Prismatic Rail?**

For a sawing task, the motion is predominantly linear. A 6-DOF or 7-DOF robotic arm moves in arcs. To create a straight line, the arm must perform complex inverse kinematics (IK) adjustments, moving all 7 joints synchronously. This consumes the robot's workspace and places it near kinematic singularities where torque capacity drops.  
Adding a linear rail (Prismatic Joint) adds a redundant Degree of Freedom (DOF). This allows the robot to maintain a preferred arm configuration (high manipulability) while the rail handles the gross linear motion of the cut.

### **5.2 USD Composition Strategy**

In Isaac Lab, we cannot simply "add a constraint" in code to an existing robot without modifying its root. We must create a composite Articulation.  
Step 1: Define the Rail USD  
We need a base and a carriage.

Python

\# Pseudo-code for USD generation in Isaac Lab context  
from pxr import Usd, UsdGeom, UsdPhysics, Gf

def create\_rail\_prim(stage, path):  
    \# 1\. Rail Base (Static)  
    rail\_base \= UsdGeom.Cube.Define(stage, path \+ "/RailBase")  
    UsdPhysics.RigidBodyAPI.Apply(rail\_base.GetPrim())  
    \# Make it static  
    \# 2\. Carriage (Moving)  
    carriage \= UsdGeom.Cube.Define(stage, path \+ "/Carriage")  
    UsdPhysics.RigidBodyAPI.Apply(carriage.GetPrim())  
      
    \# 3\. Prismatic Joint  
    joint \= UsdPhysics.PrismaticJoint.Define(stage, path \+ "/RailJoint")  
    joint.CreateBody0Rel().AddTarget(path \+ "/RailBase")  
    joint.CreateBody1Rel().AddTarget(path \+ "/Carriage")  
    joint.CreateAxisAttr("x") \# Rail moves along X  
    joint.CreateLowerLimitAttr(-2.0) \# 2 meters travel  
    joint.CreateUpperLimitAttr(2.0)

Step 2: Parent the Robot  
The robot's USD (e.g., franka.usd) usually has a world link. We must reference the robot USD and parent it to the Carriage prim. This creates a single kinematic chain:  
World \-\> RailBase \-\> (Prismatic) \-\> Carriage \-\> (Fixed) \-\> RobotBase \-\> ArmJoints...

### **5.3 Kinematic Implications and Jacobian Augmentation**

Once the robot is on the rail, the Articulation object in Isaac Lab will report an extra DOF. If the Franka has 7 joints, the new num\_dof is 8\.  
The Controller (OSC) uses the Jacobian $J$.

$$\\dot{x} \= J \\dot{q}$$

With the rail, the Jacobian matrix expands. The first column (assuming rail is index 0\) becomes the contribution of the rail to the end-effector velocity.

$$J\_{rail} \= \\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\end{bmatrix}^T$$

(assuming the rail aligns perfectly with the world X-axis and no rotation).  
**Critical Implementation Detail:** The run\_hri\_demo.py script likely loads a configuration file (YAML) defining the robot. You **must** update this configuration to reflect the 8-DOF structure. If the OSC expects 7 DOFs and receives an 8-DOF state vector, the matrix multiplication will fail with a dimension mismatch error.

### **5.4 Rail Control Strategy for Sawing**

The user asked about "expanding the test frame." With the rail, the test frame is expanded.  
For the sawing task, we can use Null-Space Projection to control the rail.

$$\\tau \= J^T F\_{task} \+ (I \- J^T J^{\\\#}) \\tau\_{null}$$

We can set the task force $F\_{task}$ to control the saw blade. We can set the null-space behavior $\\tau\_{null}$ to keep the arm in a "comfortable" pose (e.g., elbow down, mid-range) and force the rail to do the heavy lifting of moving forward. This ensures the arm stays stiff and precise, while the rail provides the stroke.  
---

## **6\. Implementation Part II: VLM-Driven Parameter Modulation**

This section details the software integration of the VLM, addressing the core request of the user.

### **6.1 The Asynchronous VLM Class**

To solve the "Deliberation-Reaction" timing mismatch, we implement the VLM as a separate Python process or thread.

Python

import threading  
import time  
from transformers import AutoModelForCausalLM, AutoProcessor

class AsyncVLMManager:  
    def \_\_init\_\_(self, model\_id="smolvlm", device="cuda"):  
        self.device \= device  
        \# Load Model (Optimized for inference)  
        self.model \= AutoModelForCausalLM.from\_pretrained(model\_id).to(device)  
        self.processor \= AutoProcessor.from\_pretrained(model\_id)  
          
        self.current\_image \= None  
        self.latest\_context \= {"material": "unknown", "fragility": 0.5, "task\_state": "idle"}  
        self.lock \= threading.Lock()  
        self.running \= True  
          
    def start(self):  
        self.thread \= threading.Thread(target=self.\_inference\_loop)  
        self.thread.start()  
          
    def update\_image(self, image\_tensor):  
        \# Update the buffer (non-blocking)  
        with self.lock:  
            self.current\_image \= image\_tensor.cpu().numpy() \# Transfer to CPU for VLM preprocessing  
              
    def \_inference\_loop(self):  
        while self.running:  
            img \= None  
            with self.lock:  
                if self.current\_image is not None:  
                    img \= self.current\_image  
              
            if img is not None:  
                \# Construct Prompt  
                prompt \= "Analyze the image. Identify the object being sawed. " \\  
                         "Is it wood, metal, or plastic? Is there a human hand nearby? " \\  
                         "Output JSON format."  
                  
                \# Run Inference (This takes 500ms+)  
                inputs \= self.processor(text=prompt, images=img, return\_tensors="pt").to(self.device)  
                output \= self.model.generate(\*\*inputs, max\_new\_tokens=50)  
                result \= self.processor.decode(output)  
                  
                \# Parse Result and Update Context  
                parsed\_data \= self.\_parse\_json(result)  
                with self.lock:  
                    self.latest\_context \= parsed\_data  
              
            time.sleep(0.1) \# Prevent busy waiting  
              
    def get\_context(self):  
        with self.lock:  
            return self.latest\_context

### **6.2 The History-Aware GRU Module**

As suggested by 1, we feed the VLM output into a GRU to prevent jitter.

Python

import torch.nn as nn

class ContextFilter(nn.Module):  
    def \_\_init\_\_(self, input\_dim, hidden\_dim):  
        super().\_\_init\_\_()  
        self.gru \= nn.GRU(input\_dim, hidden\_dim, batch\_first=True)  
        self.decoder \= nn.Linear(hidden\_dim, 2\) \# Output:  
          
    def forward(self, vlm\_embedding, hidden\_state):  
        \# vlm\_embedding: The semantic vector from the VLM  
        out, h\_new \= self.gru(vlm\_embedding, hidden\_state)  
        physics\_params \= self.decoder(out)  
        return physics\_params, h\_new

In the main run\_hri\_demo.py loop, we call ContextFilter every frame. If the VLM has a new embedding, we pass it. If not, we pass the *previous* embedding. The GRU maintains the momentum of the context.

### **6.3 Mapping Semantics to Stiffness Matrices**

This is the core logic that connects the VLM to the Sawing Task. We define a lookup table or a learned mapping that translates Semantic Class to Impedance Parameters.

| Detected Material | Semantic Property | Stiffness (Kp​) \[N/m\] | Damping (Kd​) \[Ns/m\] | Strategy |
| :---- | :---- | :---- | :---- | :---- |
| **Soft Wood (Pine)** | Low Density, High Friction | $K\_z=800$ (High), $K\_y=200$ (Med) | Critical ($\\zeta=1.0$) | Standard Sawing |
| **Hard Wood (Oak)** | High Density, High Friction | $K\_z=1000$ (High), $K\_y=100$ (Low) | Overdamped ($\\zeta=1.5$) | Prevent Binding, Allow slow feed |
| **Metal Pipe** | Very High Density | $K\_z=2000$ (Max), $K\_y=50$ (Very Low) | Overdamped ($\\zeta=2.0$) | Force dominant, minimize chatter |
| **Foam / Styrofoam** | Low Density, Low Friction | $K\_z=400$ (Low), $K\_y=400$ (High) | Underdamped ($\\zeta=0.8$) | Fast cut, prevent crushing |
| **Human / Flesh** | Safety Hazard | $K\_{all} \= 10$ (Zero) | $K\_d \= 50$ (High) | **Emergency Stop / Compliance** |

Integration in run\_hri\_demo.py:  
Inside compute\_action():

1. Fetch latest\_context from the AsyncVLMManager.  
2. Look up target $K\_p, K\_d$ based on material.  
3. Update the OSC controller:  
   Python  
   \# Dynamic parameter update  
   self.osc\_controller.set\_gains(stiffness=target\_kp, damping=target\_kd)  
   action \= self.osc\_controller.forward(target\_pose)

---

## **7\. Case Study: Addressing the "Sawing" Specifics**

The user's query highlights a specific struggle: the "push-pull" effect. This section applies the architecture specifically to solve this.

### **7.1 The Mechanism of Sawing Instability**

The "push-pull" effect described is an oscillation caused by the interaction between the robot's stiffness and the stiction (static friction) of the saw in the material.

1. **Sticking Phase:** The saw binds. The robot is at $x\_{curr}$, target is $x\_{targ}$. Error is small. Force $F \= K\_p (x\_{targ} \- x\_{curr})$ is less than Stiction $F\_s$. The saw doesn't move.  
2. **Building Phase:** The target moves forward $x\_{targ} \+ \\delta$. Error grows. Force $F$ increases.  
3. **Slip Phase:** Eventually $F \> F\_s$. The saw surges forward. Friction drops to Kinetic Friction $F\_k$ (which is lower than $F\_s$).  
4. **Overshoot:** The accumulated spring energy releases. The saw shoots past the target.  
5. **Pull Phase:** The error is now negative. The controller pulls back. The saw binds again.

### **7.2 The Solution: Hybrid Force/Motion Control via VLM**

To fix this, we cannot use pure Position Control (High Impedance). We need Force Control along the cutting axis.  
However, we don't want Force Control all the time (it's dangerous in free space).  
The VLM Role:  
The VLM detects the "Contact State".

* **Prompt:** "Is the saw blade in contact with the material?"  
* **State:** Free Motion \-\> Use High Stiffness Position Control.  
* **State:** Contact/Sawing \-\> **Switch to Force Control.**

In Force Control, the control law changes:

$$F\_{cmd} \= F\_{desired} \+ K\_d (\\dot{x}\_{desired} \- \\dot{x})$$

We remove the $K\_p$ term along the sawing axis. We simply command a constant force $F\_{desired}$ (e.g., 20N forward). This allows the saw to move at whatever speed the material allows, completely eliminating the "push-pull" fight against the waypoint.

### **7.3 Expanding the Test Frame**

The user asks: "do I expand the test frame?"  
Recommendation: Yes.

1. **Physical Expansion:** Add the Prismatic Rail (as detailed in Section 5). This physically expands the workspace, allowing for long, continuous strokes which stabilize the sawing rhythm.  
2. **Sensory Expansion:** Move the camera or add a second camera. A wrist-mounted camera is good for close-ups, but a fixed "World Camera" is better for the VLM to see the overall context (rail position, human safety).  
3. **Temporal Expansion:** The "History-Aware" framework 1 expands the "test frame" in time. By using the GRU, the system considers the past 2 seconds of interaction, not just the current millisecond.

---

## **8\. Validation and Experimental Design**

To ensure the system meets the "Exhaustive Detail" requirement, we outline the validation methodology, integrating metrics from the provided PDF.

### **8.1 Metrics for Sawing Success**

Traditional robotics metrics (Pose Error) are useless for sawing because we expect large pose errors (lag) during the cut.  
We propose new metrics:

1. **Cut Progress Rate (mm/s):** The average velocity of the saw through the material.  
2. **Force Variance ($\\sigma\_F$):** A measure of the "push-pull" oscillation. A smooth cut has low force variance. The user's current setup likely has high variance.  
3. **Binding Events:** Count of times velocity drops to zero during the cut phase.  
4. **Semantic Awareness Score (SAS):** Accuracy of the VLM in identifying the material type compared to ground truth.

### **8.2 The "Oracle" Baseline**

The report 1 suggests comparing against an "Oracle."  
For the user's experiment, the Oracle is a controller that has perfect access to the simulator's physics engine.

* **Oracle Logic:** It reads the friction coefficient $\\mu$ directly from the PhysX API.  
* If $\\mu \> 0.8$ (Rubber/Wood), it sets Stiffness \= 200\.  
* If $\\mu \< 0.2$ (Ice/Teflon), it sets Stiffness \= 1000\.  
* **Comparison:** We measure how close the VLM-driven system gets to the Oracle's performance. This isolates the VLM's perception error from the controller's performance capability.

### **8.3 Simulation Setup in Isaac Lab**

The experiment should be configured with three parallel environments in Isaac Lab:

1. **Env 0 (Baseline):** Standard run\_hri\_demo.py OSC. Fixed stiffness.  
2. **Env 1 (VLM-Modulated):** Our proposed system. Variable stiffness based on visual material detection.  
3. **Env 2 (Oracle):** Physics-grounded gain scheduling.

This A/B/C testing framework provides the rigorous data needed to prove the utility of the VLM.  
---

## **9\. Conclusion**

The integration of Vision-Language Models into the control loop of run\_hri\_demo.py is not merely a novelty; it is a structural necessity for solving the "push-pull" instability observed in contact-rich tasks like sawing. By shifting from a purely geometric control paradigm to a **Semantically-Grounded Variable Impedance** paradigm, the robot can adapt its compliance to the material properties it perceives.  
The addition of the **Prismatic Rail Constraint** via USD composition solves the kinematic limitations of fixed-base manipulation, converting the system into a high-utility linear saw. The **Dual-Loop Architecture**—supported by a **GRU History Module**—resolves the latency and amnesia issues inherent in large foundation models, ensuring that the system is both smart enough to understand the task and fast enough to execute it safely.  
This report provides the full implementation pathway, from the theoretical derivation of VLM-modulated impedance to the specific Python and USD structures required in NVIDIA Isaac Lab. By following this roadmap, the user can transform a basic HRI demo into a state-of-the-art Embodied AI platform capable of robust, material-aware manipulation.

### **Final Recommendation for the User**

1. **Immediate Step:** Implement the **Prismatic Rail** using the USD composition guide (Section 5). This will mechanically stabilize your sawing motion.  
2. **Secondary Step:** Implement the **Hybrid Force/Motion Controller** (Section 7.2) to stop the "push-pull" effect.  
3. **Advanced Step:** Integrate the **VLM** (Section 6\) to automatically switch between stiffness modes when different materials are placed on the table, fulfilling the semantic potential of the system.

#### **Works cited**

1. A Hierarchical, History-Aware Framework for VLM-B....pdf