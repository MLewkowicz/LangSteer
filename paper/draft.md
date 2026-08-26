\documentclass{article}

\usepackage{corl_2026} % Use this for the initial submission.
\usepackage{draftfigure} % For \placeholder command
\usepackage{booktabs, colortbl, multirow, xcolor}
\usepackage{amsmath,amssymb}
\usepackage{svg}
\usepackage{wrapfig}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{listings}
\lstdefinestyle{prompt}{
basicstyle=\ttfamily\footnotesize,
backgroundcolor=\color{gray!8},
frame=single,
rulecolor=\color{gray!55},
framerule=0.4pt,
breaklines=true,
breakatwhitespace=true,
columns=fullflexible,
keepspaces=true,
showstringspaces=false,
captionpos=b,
abovecaptionskip=4pt,
aboveskip=8pt,
belowskip=8pt,
xleftmargin=4pt,
xrightmargin=4pt,
}
\renewcommand{\sectionautorefname}{Sec.}
\renewcommand{\subsectionautorefname}{Sec.}

\renewcommand{\figureautorefname}{Fig.}
\renewcommand{\tableautorefname}{Table}
\renewcommand{\appendixautorefname}{Appendix}
\setlength{\belowcaptionskip}{-1em}
% \usepackage[final]{corl_2026} % Uncomment for the camera-ready ``final'' version.
% \usepackage[preprint]{corl_2026} % Uncomment for pre-prints (e.g., arxiv); This is like ``final'', but will remove the CORL footnote.

\title{LangSteer: Inference-Time Steering of Generative Policies with Language-Derived Value Maps}

% The \author macro works with any number of authors. There are two
% commands used to separate the names and addresses of multiple
% authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to
% break the lines. Using \AND forces a line break at that point. So,
% if LaTeX puts 3 of 4 authors names on the first line, and the last
% on the second line, try using \AND instead of \And before the third
% author name.

% NOTE: authors will be visible only in the camera-ready and preprint versions (i.e., when using the option 'final' or 'preprint'). 
% 	For the initial submission the authors will be anonymized.

\author{
  Jane E.~Doe\\
  Department of Electrical Engineering and Computer Sciences\\
  University of California Berkeley 
  United States\\
  \texttt{janedoe@berkeley.edu} \\
  %% examples of more authors
  %% \And
  %% Coauthor \\
  %% Affiliation \\
  %% Address \\
  %% \texttt{email} \\
  %% \AND
  %% Coauthor \\
  %% Affiliation \\
  %% Address \\
  %% \texttt{email} \\
  %% \And
  %% Coauthor \\
  %% Affiliation \\
  %% Address \\
  %% \texttt{email} \\
  %% \And
  %% Coauthor \\
  %% Affiliation \\
  %% Address \\
  %% \texttt{email} \\
}


\begin{document}
\maketitle

%===============================================================================

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{teaser.pdf}
    \caption{The base policy supports two valid placements for the wine glass (\textbf{left}): upright in the cabinet or inverted on the rack. From the user's instruction, \textsc{LangSteer} derives a target orientation, an affordance region, and an avoidance region (\textbf{right}), and steers the frozen policy toward the scene-consistent mode.}
    \label{fig:teaser}
\end{figure}

\begin{abstract}
% combinatorial nature of how task completion interacts with environment constraints 
% Such a model must cover the combinatorial product of how an environment's constraints reshape a task and how a person phrases it, so extending it to a new situation or wording requires retraining or fine-tuning the entire policy. Our key insight is that semantic task specification and low-level motion generation should be decoupled, so that resolving an instruction against the scene happens outside the policy and leaves the learned motor skill untouched. 
A robot helping a person in unstructured environments must adapt the same skill to many situations to align with a person's directed preference. Natural language is an intuitive interface for this kind of guidance, yet the dominant approach to language-conditioned manipulation fuses semantic interpretation with motion generation into a single learned model. Such a model must cover a combinatorial space of scene constraints and phrasings, and each new situation or wording demands retraining or fine-tuning the entire policy. Our key insight is that semantic task specification and low-level motion generation should be decoupled, so that resolving an instruction against the scene happens outside the policy and does not compromise action generation. We introduce \textsc{LangSteer}, an inference-time steering framework that uses a vision-language model to convert each instruction into a set of value maps over the 3D scene, and injects gradients of those maps into the policy's denoising loop. On CALVIN, \textsc{LangSteer} holds within 4.4 percentage points of its unperturbed success rate under four systematic perturbations of the instruction, while the language-conditioned baseline loses 24.6 points on average and falls to 14.2\% on the hardest axis. Additional simulated and real-world demonstrations show that \textsc{LangSteer} expresses a greater variety of scene-grounded preferences than a language-conditioned policy can.
\end{abstract}

% Two or three meaningful keywords should be added here
\keywords{human-centered robot learning, inference-time steering with language} 

%===============================================================================

\section{Introduction}
\label{sec:intro}



Modern robot policies can execute a remarkable range of manipulation tasks, but deploying these policies in human environments requires adapting to the specific situation or person being helped. Consider \autoref{fig:teaser}, where the human asks the robot to put away the wine glass, adding that the cabinet is full. The robot may already know how to place a glass in the cabinet or on the wine rack. But with the cabinet full, only one placement is valid, and determining this requires interpreting the scene and the instruction together. In such cases, the challenge is often not generating a feasible motion, but determining which valid mode of a learned policy the human intends in the current scene.

Many language-guided robot policies, including diffusion policies and vision-language-action models (VLAs), tackle this issue by conditioning action generation on a learned embedding of the instruction, fused with the scene observations~\cite{3d-diffuser-actor,diffusion-policy,octo,openvla,pi0, rdt1b}. This coupling works well when the instruction-scene pairing matches the training distribution, but is brittle in the real world~\cite{jia2025learning,guo2026robustnessvisionlanguageactionmodelmultimodal}. People paraphrase, add irrelevant detail, or omit information they find obvious~\cite{clark1986clark, chisari2025ambiguity, peng2024plga}, and any phrasing the policy didn't see during training can degrade performance unpredictably. When this happens, the policy has no way to separate a failure to understand the instruction from a failure to execute the skill: both are buried in the same action model. The usual fix is to collect more data and retrain or fine-tune the policy~\cite{openvla}, which does not scale to the combinatorial ways of describing the situations an environment presents and the ways people naturally express their preferences.

%A natural alternative lifts language understanding out of the policy and represents it as an explicit spatial structure over the scene. For instance, a large language model (LLM) can generate affordance or avoidance fields that ground an instruction to object geometries~\cite{voxposer,rekep}. We call these fields \textbf{\textit{value maps}}: reusable and inspectable representations of the most salient features of an instruction (e.g., \textit{where} to place an object or \textit{what} to avoid). These methods then execute the instruction with a classical controller, sacrificing the dexterity of a learned visuomotor policy.

The underlying issue is that understanding what a person wants and knowing how to move are different problems, but language-conditioned policies try to solve them both with the same model. Our key insight is that semantic task specification and low-level motion generation should be \textit{decoupled}. Grounding an instruction requires reasoning over the current scene, the objects in it, and the constraints the person may leave implicit; generating the motion requires staying within the contact-rich behaviors the policy learned. By separating these two roles, the robot can ground the instruction against the scene's constraints, without having to retrain the policy or compromise its learned skills.

We introduce \textsc{\textbf{LangSteer}}, an inference-time steering framework that separates language understanding from motion generation. Rather than embedding the language directly into the policy, LangSteer uses a vision-language model (VLM) to convert each instruction-scene pair into explicit spatial value maps over a 3D workspace. These maps encode where the robot should move, what to avoid, and which orientations it should prefer. The robot then steers a pretrained policy by injecting gradients of those value maps into the policy's denoising process through classifier guidance. The pretrained policy supplies contact-rich motion, the value maps supply scene-grounded semantic structure, and the base policy's weights remain unchanged at deployment time. Unlike existing steering methods that guide policies with goal images~\cite{dynaguide}, hand-specified costs~\cite{itps}, or sparse keypoints~\cite{vls}, \textsc{LangSteer}'s dense 3D value maps express continuous spatial and rotational preferences over the scene.

We evaluate \textsc{LangSteer} on the CALVIN benchmark~\cite{calvin}, perturbing the original instructions across four tiers of difficulty. On the hardest tier, \textsc{LangSteer} maintains a success rate of 73.5\% where the language-conditioned base policy falls to 14.2\%, a 59-point absolute margin under out-of-distribution language. In a custom simulation environment and real-world environment, we steer the same base policy toward spatial preferences in a kitchenware unloading task and toward orientation preferences in the wine-glass placement of Fig.~\ref{fig:teaser}, expressing preferences that a text embedding alone cannot resolve. Together, these experiments show that dense 3D value maps can serve as an inference-time interface between language understanding and generative robot policies, steering a fixed action model toward scene-grounded intent without retraining.

%===============================================================================

\label{sec:relatedworks}
\vspace{-2mm}
\section{Related Work}
\vspace{-2mm}

\textbf{Language-Conditioned Diffusion Policies for Manipulation.} Language-conditioned diffusion policies map a scene observation and a sentence embedding to an action trajectory in a single forward pass, establishing a learned visuomotor backbone for language-conditioned manipulation \cite{3d-diffuser-actor, diffusion-policy}. VLM-backed VLAs extend this design by sharing a pretrained vision-language model with the action head, producing generalist policies whose scene reasoning and instruction understanding reside in the same parameters \cite{pi0, pi05, rdt1b, chained-diffuser, octo, openvla, gr00t}. In both cases the instruction interface is entangled with the policy weights: the parameters that generate contact-rich motion also interpret the instruction, so the mapping from instruction to action is fixed at training time and degrades on phrasings absent from the training distribution. \textsc{LangSteer} retains the visuomotor backbone of such a policy but removes its in-policy language interface, supplying instruction understanding from an external module at inference time.

\textbf{LLM-Generated Spatial Representations.} A separate line of work places language understanding outside the policy, in an explicit and inspectable spatial representation over the scene. An LLM translates the instruction into a structure the controller consumes directly, ranging from 3D voxel value maps to executable scene-API programs to relational keypoint constraints \cite{voxposer, code-as-policies, rekep, moka}. Because an instruction and its paraphrases yield the same spatial structure, these methods are robust to variation in phrasing. Their limitation lies in the downstream controller: the representation is consumed by sampling-based planning~\cite{voxposer}, optimization~\cite{rekep}, or scripted primitives~\cite{code-as-policies,moka}, none of which recover the contact-rich behavior of a learned visuomotor policy. \textsc{LangSteer} adopts this value-map representation and applies it as a steering signal for a diffusion policy.

\textbf{Classifier Guidance.} Inference-time steering injects the language signal into a frozen policy's denoising loop without retraining it, adapting classifier guidance \cite{classifier-guidance} and its training-free generalizations \cite{universal-guidance, training-free-guidance} from image generation to robot diffusion policies. Methods that use inference-time steering differ mainly in the steering signal they admit. Some methods \cite{dynaguide} use goal images which require a fresh visual exemplar for each task, while energy- and cost-based steering \cite{omniguide, itps} require the user to hand-specify every term. VLS \cite{vls} is the first to steer with language, using a VLM to derive rewards over semantically grounded keypoints. Because these keypoints anchor the reward to a sparse set of scene locations, the signal cannot express fine-grained preferences over how a trajectory moves through the scene or how it interacts with an object. \textsc{LangSteer} instead derives a continuous 3D value map from the instruction, expressing intent as a dense field over workspace coordinates rather than at a handful of keypoints.

%===============================================================================
\vspace{-2mm}
\section{Method}
\label{sec:method}
\vspace{-2mm}

\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{architecture-figure-modified.pdf}
    \caption{\textbf{\textsc{LangSteer} architecture.} The VLM composer reads the instruction $\ell$ together with a scene snapshot and labeled 3D bounding boxes, then emits a conditioning pair $(\mathbf{a}, \mathbf{o})$ for the base policy and short programs that construct the value maps $(\mathcal{V}_+, \mathcal{V}_-, \mathcal{V}_r)$. We turn these maps into a scalar cost, forecast the clean trajectory, and inject the gradient back into the noise predictor as classifier guidance.} 
    \label{fig:architecture}
\end{figure}
A diffusion policy trained on a large demonstration corpus is inherently multimodal: for a given scene it represents a distribution over many plausible behaviors, and most skills a user might request already lie somewhere in that distribution. The policy can set a block down at any of several reachable spots, or place a wine glass upright on a shelf or inverted on a rack; what it cannot do on its own is decide which one a given instruction $\ell$ intends. The problem we set out to solve is to identify the mode that satisfies $\ell$ at inference time, without retraining the policy. In \textsc{LangSteer}, we use a VLM we refer to as the \textit{composer} that reads $\ell$ together with the scene and emits whichever of an affordance map $\mathcal{V}_+$, an avoidance map $\mathcal{V}_-$, and a target orientation $\mathcal{V}_r \in \mathrm{SO}(3)$ the instruction requires. We turn these maps into a cost whose gradient enters the denoising loop as a classifier-guidance correction, biasing each sample toward the mode the instruction aligns with and leaving every parameter of the base policy untouched.
\vspace{-2mm}
\subsection{Steering with Value Maps}
\vspace{-2mm}

\label{sec:conditioning}

\textbf{Preliminaries.} The robot observes its environment at each control step $t$ through posed RGB-D cameras that yield a 3D point cloud $P_t \in \mathbb{R}^{N \times 3}$ and the robot's pose $q_t$, collected as $o_t = (P_t, q_t)$. The robot's goal is to complete a task for a person in the environment, specified as a natural-language instruction $\ell$ at the start of the task. The robot acts through a pretrained diffusion policy $p_\phi(\tau_{t:t+H} \mid o_t)$~\cite{3d-diffuser-actor} that samples a horizon-$H$ end-effector trajectory $\tau_{t:t+H} = (a_t, \dots, a_{t+H})$. Each action $a_k \in \mathbb{R}^D$ encodes a 6-DoF pose and a gripper command. To sample a clean trajectory we follow the denoising diffusion probabilistic models (DDPM) formulation~\cite{ddpm}: starting from Gaussian noise $\tau^T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ of matching shape, the learned noise predictor $\epsilon_\phi(\tau^i, i \mid o_t)$ refines $\tau^i$ across diffusion steps $i = T, T{-}1, \dots, 1$ into a clean trajectory $\tau^0$.

\textbf{Decoupling Language from the Policy.} A diffusion policy can condition on an additional input $c$ that shapes the trajectory it samples, $p_\phi(\tau_{t:t+H} \mid o_t, c)$. Language-conditioned policies set $c$ to an encoding of $\ell$, training a single learned mapping from the instruction and the scene to a trajectory. This couples two demands on the same parameters: resolving what an open-ended instruction means against the scene, and generating the motion that satisfies it. In \textsc{LangSteer}, we separate them by processing $\ell$ externally into value maps that ground the instruction in the scene, and condition the policy on the minimal context required to execute a diverse skill set: a skill identifier $\mathbf{a}$ and a target object $\mathbf{o}$. The skill identifier picks out the interaction the policy should run, since the same object admits several interactions (grasp, push, pull); the target object picks out the geometry to act on, since the same skill differs in contact across objects (e.g., grasp block vs. handle).
% The two demands have different structure. Semantic interpretation must handle arbitrary phrasings and scene-dependent referents, while motion generation must produce trajectories within the modes the policy has learned. 
 
\textbf{The VLM Composer.} Selecting the correct mode requires more than the instruction. For example, in \autoref{fig:architecture}, \textit{``Put the block in the cabinet''} does not by itself indicate which block to move when several sit in the workspace. Resolving the object reference draws on knowledge the user did not communicate. VLMs trained on internet-scale image-text data carry common-sense priors over objects, scenes, and the everyday inferences a person performs without articulating them. We therefore give the role of resolving the task specification to a vision-language composer that applies those priors to resolve $\ell$ against the current scene. The composer reads $\ell$ alongside an RGB snapshot of the scene and a set of 3D bounding boxes for objects in the workspace, each labeled with its semantic class. The snapshot grounds the composer's language in the scene, and the bounding boxes give $\ell$'s referents concrete geometry to attach to.

% We give this role to a vision-language composer, which reads $\ell$ alongside an RGB snapshot of the scene and a set of 3D bounding boxes for objects in the workspace, each labeled with its semantic class. We obtain these bounding boxes from an open-vocabulary perception pipeline that segments the workspace point cloud and labels each segment (Section~\ref{sec:impl}). The snapshot grounds the composer's language in the scene, and the bounding boxes give $\ell$'s referents concrete geometry to attach to (Figure~\ref{fig:architecture}).

 From these inputs the composer emits two outputs: the conditioning pair $c = (\mathbf{a}, \mathbf{o})$ for the policy, and short programs that construct the value maps. The programs reference the bounding boxes by name, generating each map from the geometry of the boxes it identifies as relevant. We execute the programs against the current scene to produce $(\mathcal{V}_+, \mathcal{V}_-, \mathcal{V}_r)$, and re-execute them whenever the scene changes so the maps stay current with the workspace. This program structure keeps each map inspectable and dynamic. Additionally, because perception is separate from the policy, the composer can also ground value maps on any object the perception pipeline detects.
 
% The composer also decomposes the instruction into a sequence of contact-rich stages the base policy executes one at a time; ``put the block in the slider'' becomes a \emph{grasp} stage followed by a \emph{place} stage. For each stage the composer emits the conditioning pair $c$ that selects the policy's (action, object) conditioning, together with whichever subset of $\{\mathcal{V}_+, \mathcal{V}_-, \mathcal{V}_r\}$ the stage requires.

\textbf{Constructing Value Maps.} Each value map encodes one kind of constraint over the scene, and a single map can fuse contributions from any number of objects that impose that kind of constraint. The affordance map $\mathcal{V}_+ : \mathcal{W} \to [0,1]$ over a discretized workspace $\mathcal{W} \subset \mathbb{R}^3$ marks regions the end-effector should move toward, with higher values denoting stronger attraction. The avoidance map $\mathcal{V}_- : \mathcal{W} \to [0,1]$ marks regions it should keep clear of, with higher values denoting stronger repulsion. The rotation target $\mathcal{V}_r \in \mathrm{SO}(3)$ guides the end-effector's orientation to a target whenever needed, and is absent otherwise. We combine the positional maps into a scalar cost field $C_{pos} : \mathcal{W} \to \mathbb{R}$ over the workspace, evaluated at a point $\mathbf{x} \in \mathcal{W}$ as
\begin{equation}
    C_{pos}(\mathbf{x}) = 1 - \bar{\mathcal{V}}_+(\mathbf{x}) + \lambda_a \, \bar{\mathcal{V}}_-(\mathbf{x}),
    \label{eq:cost}
\end{equation}
where $\bar{\mathcal{V}}_+, \bar{\mathcal{V}}_-$ are the min-max normalized affordance and avoidance maps, and $\lambda_a$ trades avoidance against affordance. For orientation, $\mathcal{V}_r$ pulls each waypoint toward a per-waypoint target on the geodesic between its current forecast $\hat{R}_k$ and $\mathcal{V}_r$. We construct this target with spherical linear interpolation (SLERP), $R^{\star}_k = \mathrm{SLERP}(\hat{R}_k, \mathcal{V}_r, \alpha_k)$, with $\alpha_k$ ramping along the horizon so early waypoints turn only slightly and later ones approach $\mathcal{V}_r$. We measure deviation from this target in the 6D rotation representation $\rho$ the policy operates in,
\begin{equation}
    C_{\mathrm{rot},k}(R) = \tfrac{1}{2}\,\bigl\lVert \rho(R) - \rho(R^{\star}_k) \bigr\rVert^2 ,
    \label{eq:rotcost}
\end{equation}
which yields the rotation gradient as the residual $\rho(\hat{R}_k) - \rho(R^{\star}_k)$. Targeting a nearby point on the geodesic rather than $\mathcal{V}_r$ itself keeps each step's rotation correction within the policy's training distribution.

\textbf{Forecasting the Clean Trajectory.} The value maps assume a coherent trajectory, but at any intermediate diffusion step $\tau^i$ is still noisy. A cost lookup on the waypoints of $\tau^i$ would amplify that noise into a misleading gradient. Therefore, at each step, we forecast the clean trajectory the policy is converging toward, $\hat\tau^0$. Tweedie's formula~\cite{tweedie} gives this forecast in closed form as the posterior mean of $\tau^0$ conditioned on the current $\tau^i$, $\hat{\tau}^0 = \frac{\tau^i - \sqrt{1 - \bar{\alpha}_i}\;\epsilon_\phi(\tau^i, i \mid o_t, c)}{\sqrt{\bar{\alpha}_i}}$, where $\bar{\alpha}_i$ is the cumulative noise schedule. We invert the base policy's input normalization to map these waypoints back to world coordinates, then read the value-map costs at each one.
 
\textbf{Value-Map Gradient Steering.} We define a trajectory-level cost and inject its gradient into the denoising loop. Given the forecasted trajectory $\hat\tau^0$, the total cost is
\begin{equation}
    C_{\mathrm{total}}(\hat\tau^0) = \sum_{k=t}^{t+H} \left[\, C_{pos}(\hat{p}_k) + \lambda_r\, C_{\mathrm{rot},k}(\hat{R}_k) \,\right],
    \label{eq:total_cost}
\end{equation}
where $\hat{p}_k$ and $\hat{R}_k$ are the world-frame position and rotation of the $k$-th forecasted waypoint, and $\lambda_r$ weighs orientation against position. We apply a classifier-guidance correction to the noise predictor,
\begin{equation}
\hat\epsilon = \epsilon_\phi(\tau^i, i \mid o_t, c) + \eta\, s(i)\, (\sqrt{\bar{\alpha}_i} / \sqrt{1 - \bar{\alpha}_i})\, \nabla_{\hat\tau^0} C_{\mathrm{total}}(\hat\tau^0)
    \label{eq:guided_noise}
\end{equation}
with guidance strength $\eta > 0$ and the Tweedie Jacobian factor converting the $\hat\tau^0$-space gradient into $\epsilon$-space. Because the base policy denoises position and rotation through separate noise heads, we route the rotational component of $\nabla_{\hat\tau^0} C_{\mathrm{total}}$ through the rotation head with the Jacobian factor of its own schedule, and pass $\hat\epsilon$ to the standard DDPM update in place of $\epsilon_\phi$.
 
The forecast $\hat\tau^0$ is unreliable early in denoising, so we gate guidance off until diffusion has crossed the midpoint of its denoising schedule. From there, a linear schedule $s(i) \in [s_{\min}, 1]$ decays the strength to $s_{\min}$ at $i = 0$. The gate avoids steering before a trajectory mode is formed, and the decay yields control back to the base policy as the trajectory approaches its final, contact-rich form.

\subsection{Implementation Details}
\label{sec:impl}

\textbf{Policy Architecture.} We implement \textsc{LangSteer} on 3D Diffuser Actor~\citep{3d-diffuser-actor}, a conditional DDPM with separate position and rotation noise schedules whose published checkpoint conditions on a CLIP~\cite{clip} text embedding of the instruction. To realize the structured conditioning described in \autoref{sec:conditioning}, we retrain the policy once, replacing the CLIP text pathway with a pair of learned embedding tables indexed by the skill identifier $\mathbf{a}$ and target object $\mathbf{o}$. The closed vocabularies for $\mathbf{a}$ and $\mathbf{o}$ are maximally compact and fixed at training time, and the architectural component of the policy remains unchanged.

\textbf{Value Map Construction.} The composer is OpenAI's GPT-5.4-mini~\cite{gpt54mini}, a vision-language model that we prompt with the instruction, an overhead RGB snapshot of the scene, and a set of labeled 3D bounding boxes. We expose the perception module to the composer as a small Python API, paired with helper routines that write values into an empty workspace voxel grid. For each value map, the composer returns a program that queries the named objects it needs and uses these routines to populate sparse voxel assignments, which we smooth into the continuous fields $\mathcal{V}_+$ and $\mathcal{V}_-$.

\textbf{Steering.} We precompute $\nabla C_{pos}$ on the voxel grid at the start of each stage, so the position term at each waypoint reduces to a single trilinear lookup. The rotation term is the closed-form residual to the per-waypoint SLERP target. Position and rotation carry independent Jacobian factors that follow their respective noise schedules, scaled-linear for position and squared-cosine for rotation. Guidance acts on whichever channels the composer marks active for the stage, and the gripper channel uses the unmodified noise prediction throughout. We list values for $\eta$, $s_{\min}$, $\lambda_a$, $\lambda_r$, and $T$ in Appx. \ref{app:hyperparameters}.

\textbf{Multi-stage Tasks.} The composer can decompose multi-step instructions into a sequence of stages, each with its own conditioning pair $c$ and value-map programs. \textit{``Put the block in the slider''} becomes a grasp stage conditioned on \textit{(grasp, block)} followed by a place stage conditioned on \textit{(place, block)}. A stage transition fires once the end-effector reaches the basin of the value map from the previous stage, after which we re-run the composer against the updated scene to emit the next $(c, \mathcal{V}_+, \mathcal{V}_-, \mathcal{V}_r)$.
%===============================================================================
\vspace{-2mm}
\section{Experimental Results}
\vspace{-2mm}
\label{sec:result}

We aim to test three claims motivating the \textsc{LangSteer} model in our experiments: 1) decoupling language understanding from the policy makes execution robust to how an instruction is phrased; 2) the model correctly generates trajectories that follow instructions that leave the task referent under-specified, which requires reasoning about the workspace; and 3) the model can generate trajectories that satisfy continuous preferences over objects and scene elements that a sentence embedding cannot resolve into a correct mode. 
% \autoref{sec:calvin} addresses the first two claims quantitatively on the CALVIN benchmark. \autoref{sec:isaac} demonstrates spatial preferences in Isaac Sim, and \autoref{sec:isaac} demonstrates orientation steering and value-map composition on real hardware.
\vspace{-2mm}
\subsection{CALVIN Benchmark}
\vspace{-2mm}
\label{sec:calvin}

\noindent\textbf{Evaluation Setup.} We test the first two claims on the CALVIN benchmark~\cite{calvin}, a tabletop manipulation environment with 34 tasks spanning block manipulation, articulated objects, and state changes. The benchmark itself does not stress the language interface beyond the original annotations, so we construct a perturbation experiment over the same task set. We hold each task fixed and generate variant instructions along four axes of increasing difficulty:

[\textbf{P1}]~\textbf{Synonym substitution:} Replace the action verb or object descriptor with a synonym while preserving sentence structure.
%
[\textbf{P2}]~\textbf{Syntactic restructuring:} Re-order the sentence while preserving its semantics (e.g., ``push the red block left'' $\to$ ``move the red block toward the left side of the table'').
%
[\textbf{P3}]~\textbf{Verbose overspecification:} Add irrelevant detail without changing the semantics.
%
[\textbf{P4}]~\textbf{Scene-grounded underspecification:} Strip the instruction of disambiguating detail (e.g., ``move the block left'' when several blocks are present), and pair it with a starting scene whose annotation lets a viewer identify the intended execution unambiguously.
We compare three policy variants. \textbf{Base} is the language-conditioned base policy on its own \cite{3d-diffuser-actor}, with native CLIP-text conditioning and no steering. \textbf{+Steering} adds value-map gradient steering (\autoref{sec:conditioning}) on top of the base policy while preserving base conditioning, isolating the contribution of the steering gradient. \textbf{\textsc{LangSteer}} is the full framework: it replaces the base policy's CLIP-text conditioning with the structured $c =(\mathbf{a}, \mathbf{o})$ signal of \autoref{sec:conditioning} and applies steering on top.

\textbf{Results.} \autoref{fig:calvin_combined} reports the breakdown, which separates two distinct stresses on the language interface. At the training distribution (\textbf{P0}), and under mild paraphrase (\textbf{P1}) and restructuring (\textbf{P2}), \textbf{Base}, \textbf{+Steering}, and \textbf{\textsc{LangSteer}} framework perform within a few points of one another, with average success rates clustered around 80\%. Steering injection therefore does not degrade behavior on the instructions the base policy already handles. The three variants diverge sharply once the instruction drifts further from the training distribution (\textbf{P3}, \textbf{P4}). 

\textbf{P3} holds each instruction's semantics fixed and pads it with irrelevant detail. The base policy's success rate falls from 78.8\% at \textbf{P0} to 46.7\% at \textbf{P3}, with the steepest drops on \emph{Rotate} and \emph{Open/Close}. The meaning of the instruction has not changed, but since the base policy is conditioned on the CLIP embedding of the full instruction, it inherits the encoder's brittleness. \textsc{LangSteer} recovers an average of 78.4\% on the same instructions because the composer's output is a short program written against the objects detected in the workspace. \textit{``Put the block in the slider''} and \textit{``gently slide the small red block into the slider compartment''} resolve to the same program, and the policy receives identical conditioning and steering.

In \textbf{P4}, the instruction is ambiguous in isolation and becomes well-posed only once a viewer examines the workspace. The base policy's text encoder does not have workspace context, and its average success rate collapses to 14.2\%, with \emph{Rotate}, \emph{Open/Close}, and \emph{Switch} at floor (2.1\%, 2.0\%, and 4.0\%, respectively). \textsc{LangSteer} recovers an average of 73.5\% on the same instructions, a \textbf{59-point margin}. The composer reads the workspace image alongside the detected bounding boxes and grounds the referent before producing a value map. The value-map gradient then steers the policy toward a single, scene-consistent mode. \textbf{+Steering} tracks the base policy at \textbf{P0} through \textbf{P2}. It cannot close the \textbf{P4} gap: the text encoder still receives the underspecified instruction, and the gradient alone does not override the wrong mode the encoder commits to. Closing the loop requires both signals: the structured conditioning that removes encoder sensitivity, and the value-map that biases denoising toward the grounded region.
\begin{figure}[!t]
    \centering
        \includegraphics[width=\linewidth]{calvin_plots_v2.pdf}
    \caption{ \textbf{Left:} \textbf{Per-group success rate} across language perturbations P0--P4 across 34 tasks in CALVIN, with 25 trials for each condition. \textbf{Right}: average steps to task completion when successful.}
    \label{fig:calvin_combined}

\end{figure}
\vspace{-3mm}
\subsection{Expressing Spatial and Rotational Preferences}
\vspace{-2mm}

\label{sec:isaac}
\begin{wraptable}{r}{0.55\linewidth}
\caption{Preference satisfaction on Isaac scenes.}
\vspace{1.25em}
\label{tab:isaac}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l|cc|cc}
\toprule
 & \multicolumn{2}{c|}{\textbf{Mug}} & \multicolumn{2}{c}{\textbf{Bowl}} \\
\textbf{Method} & R.\ obj. & R.\ geom. & R.\ obj. & R.\ geom. \\
\midrule
VLS                       & 2/10 & 3/10 & 3/10 & 2/10 \\
\textbf{\textsc{LangSteer} (ours)} & \textbf{6/10} & \textbf{5/10} & \textbf{7/10} & \textbf{4/10} \\
\bottomrule
\end{tabular}
\vspace{-1em}
\end{wraptable}
CALVIN scores whether the policy reaches a goal state, but most of its tasks admit a single execution that satisfies the oracle. Many natural instructions name a \emph{region} of the scene rather than a discrete state: \emph{place the mug behind the cutting board} or \emph{set the bowl on the right side of the cabinet} each admit a continuum of valid executions. We therefore construct two simulated scenes in NVIDIA Isaac Sim with a Franka Panda arm and use them to ask whether \textsc{LangSteer}'s value-map interface expresses preferences the base policy cannot resolve from language alone.

\textbf{Isaac Extensions.} The first scene places a mug on a cabinet shelf with distractors; the second places a bowl in a kitchen workspace organized around a cutting board and a cabinet. Both scenes admit many viable placements, and the human's intent disambiguates among them. We train a base policy ~\cite{3d-diffuser-actor} for each scene under the structured conditioning of Section~\ref{sec:conditioning}, using a single skill pair (\emph{grasp} or \emph{place}) and a single object (\emph{mug} or \emph{bowl}). The conditioning is intentionally coarse: it commits the policy to the right skill on the right object and nothing more, leaving the distribution over placements fully multimodal at inference time. We run \textsc{LangSteer} and VLS \cite{vls} (which uses language-derived keypoints to steer) on this same base policy and use our stage transition mechanism, so any gap reflects the steering signal each method constructs.

We evaluate ten instructions per scene, split evenly across two categories. The first category specifies a placement relative to other objects in the workspace (e.g., \emph{behind the cutting board}, \emph{to the left of the bowl}); the second specifies a placement relative to the geometry of a single object or container (e.g., \emph{at the right side of the cabinet}). Table~\ref{tab:isaac} reports per-cell success counts. \textsc{LangSteer} satisfies the preference on roughly half of all instructions across both scenes, while VLS manages around one in four. The gap concentrates on the relational category. VLS resolves \emph{on top of} reliably, but cannot express \emph{behind}, \emph{left of}, or \emph{right of}: its reward attaches to a sparse set of anchor points, so the cost surface is discontinuous over the empty regions where these relations resolve. \textsc{LangSteer}'s affordance field populates those regions with continuous values from object geometry, so the gradient that pulls the end-effector toward a target also distinguishes one side of an object from the other. In the geometry category, a sparse anchor at least sits where some valid placements do, but relations like \emph{edge of} or \emph{right side of the cabinet} pick out continuous sub-regions that remain out of reach of a keypoint reward. \textsc{LangSteer} handles both families with the same mechanism.

\textbf{Real-World Demonstration.} We demonstrate both rotational steering and value-map composition on a Franka FR3 placing a wine glass, the scenario from \autoref{fig:teaser}. We use the same policy backbone \cite{3d-diffuser-actor} under the coarse structured conditioning of \autoref{sec:isaac}, now applied to the skill pair (\emph{grasp}, \emph{place}) and a single object (\emph{wine glass}). The workspace has a wine rack beneath an open cabinet, so two placement modes remain physically accessible after a successful grasp: upright in the cabinet, and inverted on the rack below. We extract semantic bounding boxes for the wine rack, the cabinet surface, and the vase. Training demonstrations cover both modes and the structured conditioning $c = $ (\textit{place}, \textit{wine glass}) carries no signal to distinguish among \emph{place} modes, and interestingly, the policy's prior commits to cabinet-upright on every unsteered trial. 

\textsc{LangSteer} selects the correct mode through a composition of two value maps, without needing to retrain the base policy or rebalance the dataset. The composer reads the instruction \emph{``the cabinet is full, put away the wine glass"} together with the observed RGB-D scene, then emits an affordance map $\mathcal{V}_+$ over the volume of the rack and a rotational target $\mathcal{V}_r$ that points the gripper downward. Both gradients enter the denoising loop simultaneously: position steering pulls the end-effector toward the rack region, and rotation steering turns each waypoint toward the inverted target which overrides the policy's default and commits it to the inverted-rack mode shown in \autoref{fig:teaser}.

%===============================================================================
\vspace{-3mm}
\section{Conclusion}
\label{sec:conclusion}
\vspace{-3mm}

\textsc{LangSteer} separates semantic interpretation from motion generation by lifting each instruction into a set of 3D value maps over the scene and steering a frozen visuomotor policy through the gradient of the value maps. The framework leaves the policy's contact-rich behavior untouched and rebuilds the language interface as an explicit, inspectable structure that the composer recomputes against each new scene. On CALVIN, value maps grounded in detected objects preserve task success under systematic phrasing perturbations. In extended settings, a continuous 3D field expresses spatial preferences that the keypoint reward of VLS cannot localize. 

\textbf{Limitations.} \textsc{LangSteer}'s skills are bounded by the base policy's modes. A target mode the policy has not learned remains unreachable regardless of the steering signal, and the closed-vocabulary assumption on the structured conditioning is a corollary. Stage switching is a second limitation: we trigger transitions between contact-rich stages with a grasp gate and a proximity heuristic, which is not a principled solution to deciding when we are in a valid starting state to begin the next skill. Additionally, the composer's value maps ground only as accurately as the bounding boxes it receives and the code the VLM generates, so perception errors and hallucinations propagate into mode selection.


%===============================================================================

\clearpage
% The acknowledgments are automatically included only in the final and preprint versions of the paper.
\acknowledgments{If a paper is accepted, the final camera-ready version will (and probably should) include acknowledgments. All acknowledgments go at the end of the paper, including thanks to reviewers who gave useful comments, to colleagues who contributed to the ideas, and to funding agencies and corporate sponsors that provided financial support.}

%===============================================================================

% no \bibliographystyle is required, since the corl style is automatically used.
\bibliography{example}  % .bib

\newpage

\appendix

\section{Parameters}\label{app:hyperparameters}

\subsection{Model / Training}

Our base policy is 3D Diffuser Actor~\citep{3d-diffuser-actor}, a conditional DDPM that predicts an $H = 20$-step end-effector trajectory from a 3D scene observation. It denoises position and rotation with separate heads under distinct noise schedules. The observation combines posed RGB-D from a static and a gripper camera, back-projected to a per-pixel point cloud, with a proprioceptive history of the last $n_{\text{hist}} = 3$ end-effector states (position and orientation). The published checkpoint conditions on a CLIP~\citep{clip} text embedding of the instruction. Each predicted waypoint is a 6-DoF pose in a 6D rotation representation together with a gripper command, produced in a normalized gripper-centric frame and mapped back to absolute world poses.

For \textsc{LangSteer} we retrain this policy once, replacing the CLIP text pathway with the skill and object embedding tables of \autoref{sec:conditioning} ($|\mathbf{a}| = 5$, $|\mathbf{o}| = 5$) concatenated into the policy's existing conditioning pathway, and leave the rest of the architecture unchanged. Training uses a reduced single-GPU budget; \autoref{tab:model_params} lists the settings.

\begin{table}[h]
\centering
\footnotesize
\caption{Base policy and retraining settings.}
\label{tab:model_params}
\begin{tabular}{ll}
\toprule
\textbf{Setting} & \textbf{Value} \\
\midrule
Position noise schedule & scaled-linear $\beta$, $\epsilon$-prediction \\
Rotation noise schedule & squared-cosine, $\epsilon$-prediction \\
Training diffusion timesteps $T$ & 25 \\
Prediction horizon $H$ & 20 waypoints \\
Gripper history & 3 \\
Embedding dimension & 192 \\
Rotation parametrization & 6D \\
Skill / object vocabulary & $|\mathbf{a}| = 5$ / $|\mathbf{o}| = 5$ \\
\midrule
Training iterations & 200k \\
Batch size & 16 \\
Learning rate & $3\!\times\!10^{-4}$, cosine decay to $1\%$ \\
Weight decay & $5\!\times\!10^{-3}$ \\
Warmup & 2{,}000 steps \\
Gradient clip & 1.0 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Steering}

Steering adds the value-map gradient of \autoref{eq:guided_noise} to the noise prediction, and \autoref{tab:steering_params} lists its parameters. The global guidance strength is $\eta = 1.0$ for all skills except lift, which uses $\eta = 0.7$. The avoidance weight $\lambda_a = 1$ balances the affordance and avoidance terms of the positional cost in \autoref{eq:cost}. The rotation term in \autoref{eq:total_cost} carries cost weight $\lambda_r = 0.4$, applied through the policy's separate rotation noise head on its own schedule and independent of the per-skill position strength.

\textbf{Adaptive guidance schedule.} The strength factor $s(i)$ of \autoref{eq:guided_noise} anneals the guidance as denoising proceeds, decaying linearly to $s_{\min} = 0.1$ as the diffusion converges, returning control to the base policy for the contact-rich finish.

\textbf{Off-manifold correction.} A large guidance strength can push the sampled trajectory off the policy's learned action manifold. The adaptive decay above is one way to mitigate this failure mode. We further apply two Langevin MCMC corrector steps (step size $10^{-3}$) at each denoising step, which re-score the trajectory under the policy and pull it back toward the manifold.

\begin{table}[h]
\centering
\footnotesize
\caption{Steering parameters.}
\label{tab:steering_params}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Guidance strength $\eta$ & 1.0 (0.7 for lift) \\
Rotation cost weight $\lambda_r$ & 0.4 \\
Avoidance weight $\lambda_a$ & 1.0 \\
Decay floor $s_{\min}$ & 0.1 \\
MCMC corrector steps & 2 \\
Corrector step size & $10^{-3}$ \\
Voxel grid resolution & $100^3$ \\
Stage transition threshold & 0.1 m \\
\bottomrule
\end{tabular}
\end{table}

\section{Model Ablation}

The structured conditioning $c = (\mathbf{a}, \mathbf{o})$ is the component of \textsc{LangSteer} that replaces the base policy's language interface. This ablation isolates its contribution by holding the steering pipeline fixed and varying only the policy's conditioning. We evaluate three conditioning variants across the perturbation tiers P0--P4 under the protocol of \autoref{sec:calvin}, with 25 trials per condition over the CALVIN task set. This isolates the conditioning axis, complementary to the base-versus-steered comparison of \autoref{sec:calvin}.

\textbf{Unconditioned} receives no instruction signal. It can execute the behaviors in its training distribution but cannot select which skill or object a given instruction intends. \textbf{Action-only} conditions on the skill identifier $\mathbf{a}$ alone. It commits the policy to the intended skill but leaves the referent ambiguous whenever several objects in the scene admit that skill. \textbf{Action+Object} is the full conditioning $c = (\mathbf{a}, \mathbf{o})$ of \autoref{sec:conditioning}, which adds the target object and is the system reported in \autoref{sec:calvin}.

\begin{table}[h]
\centering
\footnotesize
\caption{Conditioning ablation: success rate (\%) across perturbation tiers, with the steering pipeline held fixed.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Conditioning} & \textbf{P0} & \textbf{P1} & \textbf{P2} & \textbf{P3} & \textbf{P4} \\
\midrule
Unconditioned    & 32.6 & 35.6 & 27.9 & 34.4 & 26.7 \\
Action-only      & 26.2 & 30.3 & 25.6 & 28.8 & 19.9 \\
Action$+$Object  & 82.0 & 78.7 & 79.6 & 78.4 & 73.5 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Discussion.} The variants trace the role of each conditioning component. Without any instruction signal the policy has no basis to commit to the mode an instruction intends, so its success reflects only how often its prior happens to match the target. Conditioning on the skill identifier recovers the intended interaction, but it cannot disambiguate among objects that share a skill. The scene-grounded under-specification tier (P4) stresses exactly this regime, where the instruction names a skill that several objects in the scene admit and the skill identifier alone does not determine which one. Adding the target object $\mathbf{o}$ resolves the referent, which is why the full framework conditions on the pair $(\mathbf{a}, \mathbf{o})$.

\section{Composer Prompts / APIs}

Following the language-model-program structure of VoxPoser~\citep{voxposer}, the composer is realized as a set of prompted programs. A top-level composer prompt decomposes the instruction into an ordered list of stages, each emitting a conditioning pair $(\mathbf{a}, \mathbf{o})$ together with the value-map programs the stage runs. Separate affordance- and avoidance-map prompts each return a short program that queries the relevant objects and populates the workspace voxel grid. Every prompt supplies a minimal scene context, the exposed helper API, the workspace axis convention, a few in-context query-program pairs, and the live query. VoxPoser hosts its full per-program example sets externally; we instead include condensed, environment-agnostic excerpts inline.

\autoref{lst:valuemap} shows a value-map prompt; the affordance map $\mathcal{V}_+$ and avoidance map $\mathcal{V}_-$ are built by the same program structure. The instructions direct the program to fill a smooth field over the voxel workspace rather than a single target voxel, so the map can express a region beside or along one side of an object and not only its center. The in-context examples cover center, offset, and region-fill patterns, leaving the relational cases the instructions permit for the model to compose. \autoref{lst:composer} shows the composer prompt, which sequences these programs into stages.

\begin{lstlisting}[style=prompt, caption={Condensed value-map prompt}, label={lst:valuemap}]
import numpy as np
from perception_utils import parse_query_obj
from value_map_utils import (get_empty_value_map, set_voxel_by_radius,
                             set_voxel_by_box, cm2index)

# Build a 3D voxel value map: a smooth scalar field over the workspace whose
# high values mark where the end-effector should move. The composer runs one
# such program per stage, and the field becomes the gradient that pulls the
# diffusion policy's predicted trajectory toward the target during denoising.
#
# Helpers:
#   parse_query_obj(name) -> object with .position (center) and .aabb
#       (min and max corners) for a named object in the scene.
#   get_empty_value_map() -> a zeroed voxel grid to write into.
#   set_voxel_by_radius(grid, xyz, radius_cm, value) -> fill a smooth ball.
#   set_voxel_by_box(grid, obj, value) -> fill an object's bounding volume.
#   cm2index(cm, axis) -> convert a centimeter offset to a voxel-index step.
#
# Axes: x = left(-) to right(+), y = front(-) to back(+), z = bottom to top.
#
# Fill a smooth region rather than a single voxel. A target may be an object's
# center, an offset from one of its faces, a margin beside it, or a sub-region
# of its extent such as an edge or one side. These compose to express
# relations like left-of, right-of, in-front-of, behind, or above a referent.
# Assign the finished grid to ret_val.

# Query: near the center of the cup.
value_map = get_empty_value_map()
cup = parse_query_obj('cup')
set_voxel_by_radius(value_map, cup.position, radius_cm=5, value=1)
ret_val = value_map

# Query: just above the plate.
value_map = get_empty_value_map()
plate = parse_query_obj('plate')
cx, cy, cz = plate.position
(min_x, min_y, min_z), (max_x, max_y, max_z) = plate.aabb
top = (cx, cy, max_z + cm2index(5, 'z'))
set_voxel_by_radius(value_map, top, radius_cm=4, value=1)
ret_val = value_map

# Query: over the surface of the shelf.
value_map = get_empty_value_map()
shelf = parse_query_obj('shelf')
set_voxel_by_box(value_map, shelf, value=1)
ret_val = value_map

# Query: within reach of the handle.
value_map = get_empty_value_map()
handle = parse_query_obj('handle')
set_voxel_by_radius(value_map, handle.position, radius_cm=8, value=1)
ret_val = value_map

\end{lstlisting}

\begin{lstlisting}[style=prompt, caption={Condensed composer prompt.}, label={lst:composer}]
import numpy as np
from value_map_utils import get_value_map

# Decompose the instruction into an ordered list of stages that the policy
# runs one at a time. Each stage is a tuple (value_map, skill, object):
#   value_map -- where the end-effector should move this stage, from a
#                value-map program (see the value-map prompt);
#   skill     -- the gripper action the policy is conditioned on, one of
#                grasp, push, pull, place, rotate;
#   object    -- the object whose geometry the policy attends to.
# A stage may also carry an avoidance map to route around an obstacle, or a
# target orientation when the wrist pose matters. Stages run in sequence; the
# next activates once the gripper reaches the current stage's basin. If the
# object is already grasped, emit the place stage alone. Use any scene context
# to resolve ambiguous references before composing. Assign the list to ret_val.

# Query: put the cup on the shelf.
grasp_map = get_value_map('near the center of the cup')
place_map = get_value_map('over the surface of the shelf')
ret_val = [
    (grasp_map, 'grasp', 'cup'),
    (place_map, 'place', 'cup'),
]

# Query: pull the cart toward you by its handle.
grasp_map = get_value_map('near the center of the handle')
pull_map = get_value_map('a short distance in front of the handle')
ret_val = [
    (grasp_map, 'grasp', 'handle'),
    (pull_map, 'pull', 'handle'),
]
\end{lstlisting}


\section{Instruction Perturbations}

The perturbation tiers are defined in \autoref{sec:calvin}, with P0 the original annotation and P1--P4 increasing in difficulty. \autoref{tab:perturbations} gives a representative instruction for each task group under each tier.

\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\caption{Representative instructions across the perturbation tiers P0--P4 for each task group.}
\label{tab:perturbations}
\begin{tabular}{@{}l p{0.125\linewidth} p{0.125\linewidth} p{0.125\linewidth} p{0.125\linewidth} p{0.125\linewidth} p{0.125\linewidth}@{}}
\toprule
\textbf{Tier} & \textbf{Rotate} & \textbf{Lift} & \textbf{Place} & \textbf{Push} & \textbf{Open} & \textbf{Switch} \\
\midrule
P0 (original) & rotate the red block to the right & lift the red block from the table & place the block in the slider & push the red block to the left & open the drawer & turn on the green light \\
\addlinespace
P1 (synonym) & turn the red block right & pick up the red block & put the object in the sliding cabinet & slide the red block to the left & pull the drawer & push down the button to turn on the green light \\
\addlinespace
P2 (restructured) & grasp the red block, then rotate it right & grasp the red block on the table and lift it up & place the grasped object in the slider & go push the red block to the left & grasp the drawer handle and open it & toggle the button to turn on the led \\
\addlinespace
P3 (verbose) & Firmly grip the red block and twist it exactly 90 degrees clockwise. & Smoothly grasp the small red block resting on the flat table and lift it straight up. & Gently set the item you are holding down inside the sliding cabinet. & Carefully nudge the small red block over toward the left side of the workspace. & Carefully pull the metal handle to slide the drawer open completely. & Reach out and firmly press the small button to turn on the bright green LED. \\
\addlinespace
P4 (underspecified) & Rotate the block nearest the drawer clockwise. & Lift the block on the table nearest the sliding door. & Set the held object behind the sliding door. & Push block that is farthest to the right inward. & Pull open the compartment that's currently shut. & Activate the light above the slider. \\
\bottomrule
\end{tabular}
\end{table}


\end{document}
